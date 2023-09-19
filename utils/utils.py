import numpy as np
import gym
import d4rl


def dice_dataset(env_name, dataset_ratio=1.0, skip_timeout_transition=True):
    """
    trajectory rewighting is for behavior policy learning
    """
    env = gym.make(env_name)
    dataset = env.get_dataset()
    if any([s in env_name for s in ['halfcheetah', 'hopper', 'walker2d']]): # use sparse rewards for mujoco
        dataset = get_sparse_mujoco_dataset(dataset, env_name)
        dataset['rewards'] = dataset['sparse_rewards']
    N = dataset['rewards'].shape[0]

    use_timeouts = ('timeouts' in dataset)

    if dataset_ratio < 1.0:
        """
        Trajectory-wise randomly sampling:
            sample traj until # of sub-dataset > ratio * # of dataset
        """
        if use_timeouts:
            episode_terminals = np.logical_or(dataset["terminals"], dataset["timeouts"])
        else:
            episode_terminals = dataset["terminals"]

        ends = np.where(episode_terminals)[0]
        if episode_terminals[-1] < 1.0: # mark the last as terminal = 1.0
            ends = np.append(ends, N-1)
        starts = np.append(0, ends[:-1]+1)
        lens = np.append(ends[0]+1, np.diff(ends))
        assert N == np.sum(lens), f"N: {N}; lens: {np.sum(lens)}"
        n_traj = len(lens)

        order = np.random.permutation(n_traj)
        N_subdataset = 0
        mask = np.zeros(N, dtype=bool)
        for i in order: # i-th trajectory
            s, e = starts[i], ends[i]
            mask[s:e+1] = True

            N_subdataset += lens[i]
            if N_subdataset >= int(dataset_ratio*N):
                break
        
        print("Sub-sampling N_subdataset", N_subdataset)
        keys = ['observations', 'actions', 'rewards', 'terminals', 'timeouts']
        for k in keys:
            dataset[k] = dataset[k][mask]

        N = N_subdataset

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    is_init_ = []

    episode_step = 0
    last_is_done = True

    for i in range(N-1):
        if last_is_done:
            episode_step = 0
        episode_step += 1

        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        is_init = last_is_done

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps)
        if (not done_bool) and (skip_timeout_transition and final_timestep):
            # Skip this transition and don't apply terminals on the last step of an episode
            last_is_done = True
            continue

        last_is_done = done_bool

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        is_init_.append(is_init)
        episode_step += 1
    
    obs_ = np.array(obs_)
    action_ = np.array(action_)
    next_obs_ = np.array(next_obs_)
    reward_ = np.array(reward_)
    done_ = np.array(done_)
    is_init_ = np.array(is_init_)

    dice_ds =  {
        'obs': obs_,
        'act': action_,
        'obs_next': next_obs_,
        'rew': reward_,
        'terminal': done_,
        'is_init': is_init_,
    }

    # def softmax(x):
    #     y = np.exp(x - np.max(x))
    #     return y / np.sum(y)

    # if traj_weight_temp is not None:
    #     """
    #     Reweight trajectory for behavior policy training, will not influence other parts.
    #         Reweight based on the scores G normalized by d4rl, $ G\in [0,1] $,
    #         then the new weight is $\propto \frac{\exp(G)/temp}{\sum_i \exp(G_i)/temp} $.
    #         See more details in Eq.(10) in https://openreview.net/pdf?id=OhUAblg27z 
    #     """
        
    #     returns = []
    #     weights = np.zeros_like(reward_)
    #     init_idx = np.where(is_init_)[0]
    #     begin_idx, end_idx = init_idx, np.append(init_idx[1:], len(reward_))
    #     for b, e in zip(begin_idx, end_idx):
    #         returns.append(np.sum(reward_[b:e]))
    #     returns = np.array(returns)
    #     returns = np.clip(returns, 0.0, 1.0)
    #     traj_weight = softmax(returns / traj_weight_temp)
    #     for b, e, w in zip(begin_idx, end_idx, traj_weight):
    #         weights[b:e] = w
    #     weights = weights / np.sum(weights) * len(reward_)

    #     dice_ds['weight'] = weights

    returns = []
    is_optimals = np.zeros_like(reward_)
    init_idx = np.where(is_init_)[0]
    begin_idx, end_idx = init_idx, np.append(init_idx[1:], len(reward_))
    for b, e in zip(begin_idx, end_idx):
        returns.append(np.sum(reward_[b:e]))
    returns = np.array(returns)
    for b, e, ret in zip(begin_idx, end_idx, returns):
        is_optimals[b:e] = float(ret >=1.0) # is optimal if succeed
    is_optimals += 1e-6 # in case that all states are not optimal
    is_optimals = is_optimals / np.sum(is_optimals) * len(is_optimals)

    dice_ds['is_optimal'] = is_optimals

    return dice_ds


def get_return_range(dataset):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, ter, tim in zip(dataset['rewards'], dataset['terminals'], dataset['timeouts']):
        ep_ret += float(r)
        ep_len += 1
        if ter or tim:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)

def get_mujoco_ret_thres(env_name):
    all_return_thres = {
        'halfcheetah-medium-expert-v2': 10703.437232539058, 
        'walker2d-medium-expert-v2': 4924.7614947631955, 
        'hopper-medium-expert-v2': 3561.865650832653, 
        'halfcheetah-medium-v2': 4909.088379695488, 
        'walker2d-medium-v2': 3697.807839655783, 
        'hopper-medium-v2': 1621.4686678946018,
    }
    return all_return_thres[env_name]


def get_sparse_mujoco_dataset(dataset, env_name):
    # pre-calculated
    return_thres = get_mujoco_ret_thres(env_name)
    
    dataset['sparse_rewards'] = np.zeros_like(dataset['rewards'])
    ep_ret = 0.0
    for t in range(len(dataset['sparse_rewards'])):
        ep_ret += dataset['rewards'][t]
        if ep_ret > return_thres:
            dataset['sparse_rewards'][t] = 1.0
        if dataset['terminals'][t] or dataset['timeouts'][t]:
            ep_ret = 0.0

    return dataset

# if __name__ == "__main__":
#     N = 20
#     obs = np.zeros(N)
#     act = np.zeros(N)
#     rew = np.arange(N)
#     terminals = np.zeros(N)
#     terminals[np.random.randint(0,N,3)] = 1.0
#     timeouts = np.zeros(N)
#     timeouts[np.random.randint(0,N,3)] = 1.0
    
#     ds = {
#         'observations': obs,
#         'actions': act,
#         'rewards': rew,
#         'terminals': terminals,
#         'timeouts': timeouts,
#     }

#     print("="*10)
#     for k,v in ds.items():
#         print(k, v)

#     ds_with_init = dataset_with_init(ds, N)

#     print("="*10)
#     for k,v in ds_with_init.items():
#         print(k, v)