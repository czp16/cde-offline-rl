import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np
from tqdm import tqdm
import wandb

from utils import TanhNormalPolicy, to_tensor, to_numpy

def get_f_div_fn(f_name):
    """ 
    return `f`, `(f')^{-1}` 
    r(x) = ReLU((f')^{-1}(x))
    g(x) = f(ReLU((f')^{-1}(x)))
    """

    if f_name == 'chi2':
        f_fn = lambda x:0.5*(x-1)**2 
        f_prime_fn_np = lambda x:x-1
        f_prime_inv_fn = lambda x:x+1
        r_fn = lambda x: torch.relu(x+1)
        g_fn = lambda x: 0.5*(torch.relu(x+1)-1)**2
        log_r_fn = lambda x: torch.log(torch.relu(x+1)+1e-10)
    
    elif f_name == 'softchi':
        f_fn = lambda x: torch.where(x<1, x*(torch.log(x+1e-10)-1)+1, 0.5*(x-1)**2)
        f_prime_fn_np = lambda x: np.where(x<1, np.log(x), x-1)
        f_prime_inv_fn = lambda x: torch.where(x<0, torch.exp(torch.clamp_max(x,0)), x+1)
        r_fn = lambda x: torch.where(x<0, torch.exp(torch.clamp_max(x,0)), x+1)
        g_fn = lambda x: torch.where(x<0, (torch.clamp_max(x,0)-1)*torch.exp(torch.clamp_max(x,0))+1, 0.5*x**2)
        log_r_fn = lambda x: torch.where(x<0, x, torch.log(torch.relu(x)+1))

    elif f_name == 'kl':
        f_fn = lambda x: x * torch.log(x+1e-10)
        f_prime_fn_np = lambda x: np.log(x) + 1
        f_prime_inv_fn = lambda x: torch.exp(x-1)
        r_fn = lambda x: torch.exp(x-1)
        g_fn = lambda x: (x-1) * torch.exp(x-1)
        log_r_fn = lambda x: x-1

    else:
        raise NotImplementedError('Not implemented f_fn:', f_name)

    return f_fn, f_prime_fn_np, f_prime_inv_fn, r_fn, g_fn, log_r_fn


class CDELearner:
    def __init__(
            self, 
            gamma,
            alpha,

            policy,
            policy_optim,
            data_policy,
            data_policy_optim,
            v_network,
            v_optim,
            e_network,
            e_optim,
            n_step_per_episode,

            num_repeat_actions,
            zeta_mix_dist,
            ood_eps,
            mix_data_policy,

            e_value_type="q", # determine e_net = q or Adv
            IS_ratio_normalize=True,
            policy_ent_reg=False,
            data_policy_ent_reg=False,
            target_ent=None,
            device=None,

            policy_extract_mode='info_proj',
            normalize_obs=True,
            normalize_rew=True,
            rew_scale=1.0,
            # logger=None,
            eval_policy_fn=None,
        ):

        self._gamma = gamma
        self._alpha = alpha

        self.policy = policy
        self.policy_optim = policy_optim
        self.data_policy = data_policy
        self.data_policy_optim = data_policy_optim
        self.v_network = v_network
        self.v_optim = v_optim
        self.e_network = e_network
        self.e_optim = e_optim
        self.n_step_per_episode = n_step_per_episode

        self.num_repeat_actions = num_repeat_actions
        self.zeta_mix_dist = zeta_mix_dist
        self.ood_eps = ood_eps # corresponds to \tilde{\epsilon} in paper
        self.mix_data_policy = mix_data_policy

        self.e_value_type = e_value_type
        self.IS_ratio_normalize = IS_ratio_normalize
        
        self._lambda_v = torch.tensor(0.0, device=device, requires_grad=True)
        self._lambda_e = torch.tensor(0.0, device=device, requires_grad=True)

        if self.IS_ratio_normalize: # the constraint: E_D[w(s,a)] = 1
            self._lambda_v_optim = torch.optim.Adam([self._lambda_v], 3e-4)
            self._lambda_e_optim = torch.optim.Adam([self._lambda_e], 3e-4)
        self.device = device
        
        self.policy_ent_reg = policy_ent_reg
        self.data_policy_ent_reg = data_policy_ent_reg
        self.target_entropy = target_ent
        if self.policy_ent_reg:
            self.log_ent_coef = torch.tensor(0.0, device=device, requires_grad=True)
            self.ent_coef_optim = torch.optim.Adam([self.log_ent_coef], 3e-4)
        if self.data_policy_ent_reg:
            self.log_data_ent_coef = torch.tensor(0.0, device=device, requires_grad=True)
            self.data_ent_coef_optim = torch.optim.Adam([self.log_data_ent_coef], 3e-4)
            
        self.policy_extract_mode = policy_extract_mode
        self.normalize_obs = normalize_obs
        self.normalize_rew = normalize_rew
        self.rew_scale = rew_scale
        # self.logger = logger
        self.eval_policy_fn = eval_policy_fn

        # self.is_tanh_policy = isinstance(self.policy, TanhNormalPolicy)
        # assert self.is_tanh_policy
        self.f_fn, self.f_prime_fn_np,  self.f_prime_inv_fn, self.r_fn, self.g_fn, self.log_r_fn = get_f_div_fn('softchi')
        

    def learn_w(self, batch, init_batch, batch_size, total_episodes, warmup_episodes):
        for episode in tqdm(range(total_episodes), desc="CDE", ncols=100):
            # below lists are for logging
            v_loss_list = []
            e_loss_list = []
            Df_list = []
            TD_error_list = []
            e_ood_loss_list = []
            data_policy_loss_list = []
            data_policy_entropy_list = []
            policy_loss_list = []
            policy_kl_list = []
            policy_logw_list = []
            policy_entropy_list =[]

            init_batch_len = len(init_batch)

            for _ in range(self.n_step_per_episode):
                b = batch.sample(batch_size)

                init_indices = np.random.randint(0, init_batch_len, batch_size)
                obs_init = init_batch.obs[init_indices]
                v_init = self.v_network(obs_init) #

                v_s = self.v_network(b.obs)
                v_s_next = self.v_network(b.obs_next)
                
                # adv_sa and following variables are computated from `v_network`
                adv_sa = b.rew + self._gamma * (1.0 - b.terminal) * v_s_next - v_s
                adv_over_alpha = (adv_sa - self._lambda_v) / self._alpha
                w_sa = self.r_fn(adv_over_alpha)
                f_w_sa = self.g_fn(adv_over_alpha)

                # adv_e and following variables are computated from `e_network`
                if self.e_value_type == "adv":
                    adv_e = self.e_network(b.obs, b.act)
                elif self.e_value_type == "q":
                    adv_e = self.e_network(b.obs, b.act) - v_s.detach()
                adv_e_over_alpha = (adv_e - self._lambda_v) / self._alpha
                w_e = self.r_fn(adv_e_over_alpha)

                Df = f_w_sa.mean().item() # divergence between distributions of policy & dataset
                if np.isnan(Df):
                    raise RuntimeError("Not converge when minimizing v. Try larger alpha.")
                Df_list.append(Df)

                td_error = adv_sa.pow(2).mean()
                TD_error_list.append(td_error.item())

                # 1. v loss, lambda_v_loss
                v_loss = w_sa*(adv_sa-self._lambda_v.detach()) - self._alpha*f_w_sa
                v_loss = v_loss.mean() + (1-self._gamma)*v_init.mean()
                
                if self.IS_ratio_normalize:
                    lambda_v_loss = w_sa.detach()*(adv_sa.detach()-self._lambda_v) + self._lambda_v
                    lambda_v_loss = lambda_v_loss.mean()

                self.v_optim.zero_grad()
                v_loss.backward() # retain_graph=True
                self.v_optim.step()

                if self.IS_ratio_normalize:
                    self._lambda_v_optim.zero_grad()
                    lambda_v_loss.backward()
                    self._lambda_v_optim.step()

                v_loss_list.append(v_loss.item())
                # lambda_v_loss_list.append(lambda_v_loss.item())


                # 2. e_loss, lambda_e_loss
                e_loss = F.mse_loss(adv_e, adv_sa.detach())
                if 0.0 < self.zeta_mix_dist < 1.0:
                    ood_act = torch.empty(
                        (self.num_repeat_actions, *b.act.shape), device=self.device
                    ).uniform_(-1.0, 1.0)

                    e_ood = self.get_ood_e(b, ood_act)
                    if self.e_value_type == "adv":
                        adv_ood = e_ood
                    elif self.e_value_type == "q":
                        adv_ood = e_ood - v_s.detach()
                    
                    ood_loss = adv_ood - self._lambda_v.detach() - self._alpha * self.f_prime_fn_np(self.ood_eps)
                    ood_loss = torch.square(torch.relu(ood_loss)).mean() # relu: only add loss when A > alpha * f'(\tilde{eps})

                    e_loss = self.zeta_mix_dist * e_loss + (1-self.zeta_mix_dist) * ood_loss

                    e_ood_loss_list.append(ood_loss.item())
                
                
                # if self.IS_ratio_normalize:
                #     w_ood = self.r_fn((adv_ood - self._lambda_e)/ self._alpha).detach().mean()
                #     lambda_e_loss = -(self.zeta_mix_dist * w_e.detach().mean() + (1-self.zeta_mix_dist) * w_ood)*self._lambda_e + self._lambda_e
                #     lambda_e_loss = lambda_e_loss.mean()

                self.e_optim.zero_grad()
                e_loss.backward()
                self.e_optim.step()

                # if self.IS_ratio_normalize:
                #     self._lambda_e_optim.zero_grad()
                #     lambda_e_loss.backward()
                #     self._lambda_e_optim.step()

                
                # 3. data_policy loss, simple BC with entropy regularization
                if self.policy_extract_mode == 'info_proj':
                    dp_dist, _y, _x = self.data_policy(b.obs)
                    dp_log_p, _ = self.data_policy.log_prob(dp_dist, b.act)

                    if self.mix_data_policy and (0.0 < self.zeta_mix_dist < 1.0):
                        dp_ood_log_p, _ = self.data_policy.log_prob(dp_dist, ood_act)
                        assert dp_ood_log_p.shape == (self.num_repeat_actions, batch_size, 1)
                        dp_ood_log_p = dp_ood_log_p.mean(0)
                        data_policy_loss = self.zeta_mix_dist * dp_log_p + (1 - self.zeta_mix_dist) * dp_ood_log_p
                        data_policy_loss = - data_policy_loss.mean()
                    else:
                        data_policy_loss = - dp_log_p.mean()

                    _dp_log_py, _ = self.data_policy.log_prob(dp_dist, _y, _x)
                    data_policy_entropy = - _dp_log_py.mean()

                    if self.data_policy_ent_reg:
                        data_ent_coef = torch.exp(self.log_data_ent_coef).detach()
                        data_policy_loss -= data_ent_coef * data_policy_entropy
                        data_ent_coef_loss = - self.log_data_ent_coef * (self.target_entropy - data_policy_entropy.detach())

                    data_policy_entropy_list.append(data_policy_entropy.item())


                    self.data_policy_optim.zero_grad()
                    data_policy_loss.backward(retain_graph=True)
                    self.data_policy_optim.step()

                    if self.data_policy_ent_reg:
                        # auto update log_data_ent_coef
                        self.data_ent_coef_optim.zero_grad()
                        data_ent_coef_loss.backward()
                        self.data_ent_coef_optim.step()
                    
                    data_policy_loss_list.append(data_policy_loss.item())


                # 4. policy loss
                if episode >= warmup_episodes:
                    # _act: sampled act (not the act in dataset)
                    dist, _act, _pretanh_act = self.policy(b.obs)
                    
                    if self.policy_extract_mode == "weighted_BC":
                        log_p, _ = self.policy.log_prob(dist, b.act)
                        policy_loss = - (w_e.detach() * log_p).mean()
                    
                    elif self.policy_extract_mode == 'info_proj':
                        if self.e_value_type == "adv":
                            sampled_adv_sa = self.e_network(b.obs, _act)
                        elif self.e_value_type == "q":
                            sampled_adv_sa = self.e_network(b.obs, _act) - v_s.detach()
                        _log_we = self.log_r_fn((sampled_adv_sa-self._lambda_v.detach())/self._alpha)
                        
                        with torch.no_grad():
                            data_dist, _, _ = self.data_policy(b.obs)
                        # alternative way: sampling to estimate kl
                        _, _log_px = self.policy.log_prob(dist, _act, _pretanh_act)
                        _, _data_log_px = self.data_policy.log_prob(data_dist, _act, _pretanh_act)
                        
                        if not self.mix_data_policy:
                            kl = _log_px - self.zeta_mix_dist * _data_log_px # here we need zeta on \pi_data to approximate \hat{\pi}_data
                        else:
                            kl = _log_px - _data_log_px
                        # kl = D.kl_divergence(dist, data_dist)

                        policy_loss = - b.is_optimal * (_log_we - kl)
                        policy_loss = policy_loss.mean()

                        policy_kl_list.append(kl.mean().item())
                        policy_logw_list.append(_log_we.mean().item())

                    else:
                        raise NotImplementedError(
                            f'The policy extraction mode`{self.policy_extract_mode}` is not available.'
                        )

                    # policy_ent = dist.entropy().mean()
                    # alternative way: sampling to estimate entropy
                    sampled_log_p, _ = self.policy.log_prob(dist, _act, _pretanh_act)
                    policy_ent = - sampled_log_p.mean()
                    if self.policy_ent_reg:
                        ent_coef = torch.exp(self.log_ent_coef).detach()
                        policy_loss -= ent_coef * policy_ent
                        ent_coef_loss = - self.log_ent_coef * (self.target_entropy - policy_ent.detach())
                    policy_entropy_list.append(policy_ent.item())

                    self.policy_optim.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    self.policy_optim.step()

                    if self.policy_ent_reg:
                        # auto update log_ent_coef
                        self.ent_coef_optim.zero_grad()
                        ent_coef_loss.backward()
                        self.ent_coef_optim.step()
                    
                    policy_loss_list.append(policy_loss.item())

                e_loss_list.append(e_loss.item())

            loss_dict = {
                'loss/v_loss_minimax': np.mean(v_loss_list),
                'loss/td_error': np.mean(TD_error_list),
                'alpha': self._alpha,
            }
            
            loss_dict[f'loss/{self.e_value_type}_loss_mse'] = np.mean(e_loss_list)
            if 0 < self.zeta_mix_dist < 1.0:
                loss_dict[f'loss/{self.e_value_type}_ood_loss_mse'] = np.mean(e_ood_loss_list)

            if self.IS_ratio_normalize:
                loss_dict['lambda_v'] = self._lambda_v.item()
                loss_dict['lambda_e'] = self._lambda_e.item()
            if self.policy_extract_mode == 'info_proj':
                loss_dict['loss/data_policy'] = np.mean(data_policy_loss_list)
                loss_dict['loss/data_policy_entropy'] = np.mean(data_policy_entropy_list)
            
            if episode >= warmup_episodes:
                loss_dict['Df'] = np.mean(Df_list)
                loss_dict['loss/policy'] = np.mean(policy_loss_list)
                if self.policy_extract_mode == 'info_proj':
                    loss_dict['loss/policy_kl'] = np.mean(policy_kl_list)
                    loss_dict['loss/policy_log_w'] = np.mean(policy_logw_list)
                loss_dict['loss/policy_entropy'] = np.mean(policy_entropy_list)

            if episode >= warmup_episodes and self.eval_policy_fn:
                eval_dict = self.eval_policy_fn()
                loss_dict.update(**eval_dict)

            # if episode >= 0:
            #     _obs = batch.obs[999999:1000998]
            #     _act = batch.act[999999:1000998]
            #     v_obs = to_numpy(self.v_network(_obs).squeeze())
            #     e_obs_act = to_numpy(self.e_network(_obs, _act).squeeze())
                
            #     import matplotlib.pyplot as plt
            #     plt.figure(figsize=(8,4))
            #     plt.subplot(1,2,1)
            #     plt.plot(np.arange(1000-1), v_obs)
            #     plt.subplot(1,2,2)
            #     plt.plot(np.arange(1000-1), e_obs_act)
            #     plt.savefig('tmp.png')

            wandb.log(loss_dict, step=episode)
            
    def get_ood_e(self, batch, ood_act):
        bs = len(batch)
        # ood_act : [num_repeat_actions, bs, a_dim]

        repeat_size = [self.num_repeat_actions, 1, 1]
        view_size = [bs * self.num_repeat_actions, batch.obs.shape[-1]]
        tmp_obs = batch.obs.unsqueeze(0).repeat(*repeat_size).view(*view_size)

        e_ood = self.e_network(tmp_obs, ood_act.view(bs * self.num_repeat_actions, -1))
        e_ood = e_ood.reshape(self.num_repeat_actions, bs, 1)

        return e_ood


    def _get_timestep(self, batch):
        timestep = np.zeros_like(batch.rew)
        t = 0
        for _ in range(len(timestep)):
            timestep[_] = t
            t += 1
            if batch.terminal[_]:
                t = 0
        return timestep
    

    def preprocess(self, batch):
        assert hasattr(batch, 'is_init')
        self.init_s_propotion = batch.is_init.mean()
        print(f"data init state proportion:{self.init_s_propotion:.4f}")

        if self.normalize_obs:
            obs_mean, obs_std = batch.obs.mean(0, keepdims=True), batch.obs.std(0, keepdims=True)
            batch.obs = (batch.obs - obs_mean) / (obs_std + 1e-7)
            batch.obs_next = (batch.obs_next - obs_mean) / (obs_std + 1e-7)
        
        rew_mean, rew_std = batch.rew.mean(), batch.rew.std()
        self.rew_std = rew_std
        if self.normalize_rew:
            batch.rew = (batch.rew - rew_mean) / (rew_std + 1e-7)
        if self.rew_scale:
            batch.rew = self.rew_scale * batch.rew

        # if self._merge_timeout_to_terminal:
        #     batch.terminal = np.logical_or(batch.terminal, batch.timeout).astype(batch.terminal.dtype)

        
        batch.obs = to_tensor(batch.obs)
        batch.act = to_tensor(batch.act)
        batch.obs_next = to_tensor(batch.obs_next)
        batch.rew = to_tensor(batch.rew).unsqueeze(-1)
        batch.terminal = to_tensor(batch.terminal).unsqueeze(-1)
        batch.is_init = to_tensor(batch.is_init).unsqueeze(-1)
        if hasattr(batch, "is_optimal"):
            batch.is_optimal = to_tensor(batch.is_optimal).unsqueeze(-1)
        else:
            batch.is_optimal = torch.ones_like(batch.rew)

        return batch


    def learn(self, batch, batch_size, total_episodes, warmup_episodes=0):
        batch = self.preprocess(batch)
        init_batch = batch[batch.is_init[:,0] > 0]

        return self.learn_w(batch, init_batch, batch_size, total_episodes, warmup_episodes)