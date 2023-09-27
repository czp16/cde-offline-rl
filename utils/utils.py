import numpy as np
import gym
import d4rl


def dice_dataset(env_name, dataset_ratio=1.0, skip_timeout_transition=True):
    
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
    is_optimals = np.array([])

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps)
        if (not done_bool) and (skip_timeout_transition and final_timestep):
            # Skip this transition and don't apply terminals on the last step of an episode
            is_optimals = np.append(is_optimals, (reward > 0.0) * np.ones(episode_step))
            episode_step = 0
            continue

        is_init_.append(int(episode_step==0))

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

        if done_bool or final_timestep:
            is_optimals = np.append(is_optimals, (reward > 0.0) * np.ones(episode_step))
            episode_step = 0

    is_optimals = np.append(is_optimals, (reward > 0.0) * np.ones(episode_step))
    
    obs_ = np.array(obs_)
    action_ = np.array(action_)
    next_obs_ = np.array(next_obs_)
    reward_ = np.array(reward_)
    done_ = np.array(done_)
    is_init_ = np.array(is_init_)

    if is_optimals.sum() < 10:
        is_optimals += 1e-6 # in case that all states are not optimal
        is_optimals = is_optimals / np.sum(is_optimals) * len(is_optimals)

    dice_ds =  {
        'obs': obs_,
        'act': action_,
        'obs_next': next_obs_,
        'rew': reward_,
        'terminal': done_,
        'is_init': is_init_,
        'is_optimal': is_optimals,
    }

    # dice_ds['is_optimal'] = is_optimals
    assert dice_ds['is_optimal'].size == dice_ds['rew'].size, f"{dice_ds['is_optimal'].size} v.s. {dice_ds['rew'].size}"

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

if __name__ == "__main__":
    env_names = [
        "halfcheetah-medium-expert-v2", 
        "walker2d-medium-expert-v2", 
        "hopper-medium-expert-v2", 
        "halfcheetah-medium-v2", 
        "walker2d-medium-v2", 
        "hopper-medium-v2",
    ]
    for env_name in env_names:
        print(env_name)
        ds = dice_dataset(env_name)
        print('len', ds['is_optimal'].size, 'n isinit', ds['is_init'].sum())