import numpy as np
import time, os
import wandb
import torch
import torch.distributions as D
import gym
import mujoco_py
import d4rl
from torch.utils.tensorboard import SummaryWriter

from cde import CDELearner
from utils import Batch, Critic, NormalPolicy, MixtureNormalPolicy, \
    TanhNormalPolicy, TanhMixtureNormalPolicy, dice_dataset, \
    to_tensor, to_numpy, set_seed, Config, get_mujoco_ret_thres

def get_args(parser):
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--env_name', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--dataset_ratio', type=float, default=1.0)
    parser.add_argument('--hyperparams', type=str)
    parser.add_argument('--cudaid', type=int, default=-1)
    # parser.add_argument('--alpha_f_div', type=float, default=0.1)
    # parser.add_argument('--num_repeat_actions', type=int, default=5)
    # parser.add_argument('--zeta_mix_dist', type=float, default=0.9)
    # parser.add_argument('--ood_eps', type=float, default=0.3)



def main():
    cfg = Config()
    get_args(cfg.parser)
    cfg.load_hyperparams()
    cfg.select_device()

    HP = cfg.hyperparams

    env = gym.make(HP['env']['name'])

    eval_env = env
    
    # set seed
    set_seed(HP['misc']['seed'])
    eval_env.seed(HP['misc']['seed'])

    # Generate dataset and obs normalizer
    dataset_dict = dice_dataset(
        HP['env']['name'], 
        HP['misc']['dataset_ratio'], 
        # use_sparse_rew=HP['misc']['use_sparse_r'],
    )
    data_batch = Batch(**dataset_dict)

    dataset_size = len(data_batch)
    print("Size of dataset:", dataset_size)

    if HP['preprocess']['normalize_obs']:
        obs_mean = data_batch.obs.mean(0)
        obs_std = data_batch.obs.std(0)
    else:
        obs_mean, obs_std = 0, 1

    s_dim = data_batch.obs.shape[-1]
    a_dim = data_batch.act.shape[-1]

    # Create network
    nn_lr = HP['network']['network_lr']

    if HP['network']['use_tanh_squash']:
        policy = TanhNormalPolicy(s_dim, a_dim, HP['network']['policy_hidden_dim']).to(Config.DEVICE)
        if HP['network']['dp_mixture_n_comp'] > 1:
            data_policy = TanhMixtureNormalPolicy(HP['network']['dp_mixture_n_comp'], s_dim,a_dim, 1.0, HP['network']['dataset_policy_hidden_dim']).to(Config.DEVICE)
        else:
            data_policy = TanhNormalPolicy(s_dim,a_dim, HP['network']['dataset_policy_hidden_dim']).to(Config.DEVICE)
    else:
        policy = NormalPolicy(s_dim, a_dim, HP['network']['policy_hidden_dim']).to(Config.DEVICE)
        if HP['network']['dp_mixture_n_comp'] > 1:
            data_policy = MixtureNormalPolicy(HP['network']['dp_mixture_n_comp'], s_dim,a_dim, 1.0, HP['network']['dataset_policy_hidden_dim']).to(Config.DEVICE)
        else:
            data_policy = NormalPolicy(s_dim,a_dim, HP['network']['dataset_policy_hidden_dim']).to(Config.DEVICE)
            
    policy_optim = torch.optim.Adam(policy.parameters(), lr=nn_lr)
    data_policy_optim = torch.optim.Adam(data_policy.parameters(), lr=nn_lr)

    v_net = Critic(s_dim, hidden_dims=HP['network']['v_net_hidden_dim']).to(Config.DEVICE)
    v_optim = torch.optim.Adam(v_net.parameters(), lr=nn_lr, weight_decay=HP['network']['v_net_l2_regularize'])
    e_net = Critic(s_dim, a_dim, HP['network']['e_net_hidden_dim']).to(Config.DEVICE)
    e_optim = torch.optim.Adam(e_net.parameters(), lr=nn_lr, weight_decay=HP['network']['e_net_l2_regularize'])

    wandb.init(
        project="CDE",
        entity="czp16",
        name=f"{HP['env']['name']}",
        config={
            "env_name": HP['env']['name'],
            "seed": HP['misc']['seed'],
            "dataset_ratio": HP['misc']['dataset_ratio'],
            "alpha_f_div": HP['DICE']['alpha_f_div'],
            "zeta": HP['DICE']['zeta_mix_dist'],
            # "ood_eps": HP['DICE']['ood_eps'],
            # "numood": HP['DICE']['num_repeat_actions'],
        }
    )

    # Create logger to log the training data
    log_dir = os.path.join(HP['learner']['writer_dir'], f"{HP['env']['name']}/")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, 0o777)
    
    # evaluation function
    def eval_policy_fn():
        reward_list = []
        for episode in range(HP['learner']['eval_episode']):
            obs = eval_env.reset()
            total_reward = 0.
            done, i = False, 0
            while not done:
                obs = (obs - obs_mean)/obs_std
                with torch.no_grad():
                    obs = to_tensor(obs).unsqueeze(0) # 1 x dim_obs
                    action = policy.deterministic_action(obs)
                    action = to_numpy(action.squeeze(0))
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward
                i += 1
            reward_list.append(total_reward)

        reward_list = np.array(reward_list)

        reward_mean, reward_std = np.mean(reward_list), np.std(reward_list)
        norm_rew = env.get_normalized_score(reward_mean)
        norm_std = env.get_normalized_score(reward_mean + reward_std)
        norm_std -= norm_rew

        eval_dict = {
            "eval/scores_mean": 100*norm_rew,
            "eval/scores_std": 100*norm_std,
        }

        if any([s in HP['env']['name'] for s in ['halfcheetah', 'hopper', 'walker2d']]):
            return_thres = get_mujoco_ret_thres(HP['env']['name'])
            sparse_reward_list = (reward_list > return_thres)
            eval_dict['eval/success_rate_mean'] = 100*np.mean(sparse_reward_list)
            eval_dict['eval/success_rate_std'] = 100*np.std(sparse_reward_list)

        return eval_dict
    
    
    target_entropy = HP['learner']['target_entropy']
    if target_entropy is None:
        target_entropy = -a_dim
    device = Config.DEVICE

    offline_learner = CDELearner(
        HP['learner']['gamma'], 
        HP['DICE']['alpha_f_div'],
        policy, policy_optim,
        data_policy, data_policy_optim,
        v_net, v_optim, 
        e_net, e_optim,
        HP['learner']['n_step_per_episode'],
        HP['DICE']['num_repeat_actions'],
        HP['DICE']['zeta_mix_dist'],
        HP['DICE']['ood_eps'],
        HP['DICE']['mix_data_policy'],
        HP['learner']['e_value_type'],
        HP['learner']['IS_ratio_normalize'],
        HP['learner']['policy_entropy_reg'],
        HP['learner']['data_policy_entropy_reg'],
        target_entropy,
        device,
        HP['learner']['policy_extract_mode'],
        HP['preprocess']['normalize_obs'],
        HP['preprocess']['normalize_rew'],
        HP['preprocess']['rew_scale'],
        # logger,
        eval_policy_fn,
    )
    offline_learner.learn(
        data_batch, 
        HP['learner']['batch_size'], 
        HP['learner']['epoch'], 
        HP['learner']['warmup_epoch']
    )
    
    # torch.save({
    #     'policy': policy.state_dict(),
    # }, 'exp/model/cde.model')



if __name__ == "__main__":
    main()