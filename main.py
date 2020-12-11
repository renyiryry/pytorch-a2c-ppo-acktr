import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo
from arguments import get_args
from envs import make_vec_envs
from model import Policy
from storage import RolloutStorage
from utils import get_vec_normalize
from visualize import visdom_plot

args = get_args()

# print('args.env_name')
# print(args.env_name)

# print('args.num_processes')
# print(args.num_processes)

# sys.exit()

if args.env_name == 'CartPole-v0':
    assert args.num_processes == 1

# assert args.algo in ['a2c', 'ppo', 'acktr']
# assert args.algo in ['a2c', 'ppo', 'acktr', 'kbfgs']
# assert args.algo in ['a2c', 'ppo', 'acktr', 'acktr-homo', 'kbfgs']
# assert args.algo in ['a2c', 'ppo', 'acktr', 'acktr-homo', 'kbfgs-homo']
# assert args.algo in ['a2c', 'ppo', 'acktr', 'acktr-homo', 'kbfgs', 'kbfgs-homo']
assert args.algo in ['a2c', 'ppo', 'acktr', 'acktr-homo', 'acktr-homo-noEigen',
                     'kbfgs', 'kbfgs-homo',
                     'kbfgs-homo-invertA', 'kbfgs-homo-invertA-decoupledDecay',
                     'kbfgs-homo-momentumGrad',
                     'kbfgs-homo-noClip']

if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'
    
print('os.path.join(args.save_dir, args.algo)')
print(os.path.join(args.save_dir, args.algo))
    
print('args.log_interval')
print(args.log_interval)
    
print('args.num_frames')
print(args.num_frames)

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

print('num_updates')
print(num_updates)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

eval_log_dir = args.log_dir + "_eval"

try:
    os.makedirs(eval_log_dir)
except OSError:
    files = glob.glob(os.path.join(eval_log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(port=args.port)
        win = None

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                        args.gamma, args.log_dir, args.add_timestep, device, False)

    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    
    print('args.lr')
    print(args.lr)
    
#     print('args.stat_decay')
#     print(args.stat_decay)
    
#     sys.exit()

    if args.algo == 'a2c':
        
#         print('args.eps')
#         print(args.eps)
        
#         sys.exit()
        
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo in ['acktr']:
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps,
                               acktr=True, stat_decay=args.stat_decay)
    elif args.algo in ['acktr-homo']:
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps,
                               acktr=True,
                               if_homo=True, stat_decay=args.stat_decay)
    elif args.algo in ['acktr-homo-noEigen']:
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps, acktr=True,
                               if_homo=True, stat_decay=args.stat_decay, if_eigen=False)
    elif args.algo in ['kbfgs']:
        
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps,
                               kbfgs=True, stat_decay=args.stat_decay)
    elif args.algo in ['kbfgs-homo']:
        
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps,
                               kbfgs=True, if_homo=True, stat_decay=args.stat_decay)
    elif args.algo in ['kbfgs-homo-invertA']:
        
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps,
                               kbfgs=True, if_homo=True,
                               stat_decay=args.stat_decay, if_invert_A=True)
        
    elif args.algo in ['kbfgs-homo-invertA-decoupledDecay']:
        
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps,
                               kbfgs=True, if_homo=True,
                               stat_decay_A=args.stat_decay_A,
                               stat_decay_G=args.stat_decay_G, 
                               if_invert_A=True,
                               if_decoupled_decay=True)
    elif args.algo in ['kbfgs-homo-momentumGrad']:
        
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps,
                               kbfgs=True, if_homo=True,
                               if_momentumGrad=True, stat_decay=args.stat_decay)
    elif args.algo in ['kbfgs-homo-noClip']:
        
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr, eps=args.eps,
                               kbfgs=True, if_homo=True,
                               if_clip=False, stat_decay=args.stat_decay)
    else:
        print('unknown args.algo for ' + args.algo)
        sys.exit()

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                        envs.observation_space.shape, envs.action_space,
                        actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    
    record_rewards = []
    
    record_num_steps = []
    
    print('num_updates')
    print(num_updates)
    
    total_num_steps = 0

    start = time.time()
    for j in range(num_updates):
        
        print('j')
        print(j)
        
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                
#                 print('info.keys()')
#                 print(info.keys())
                
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                
                    print('info[episode][r]')
                    print(info['episode']['r'])
                    
                    record_rewards.append(info['episode']['r'])
                    
#                     print('total_num_steps')
#                     print(total_num_steps)
                    
#                     print('total_num_steps + (step + 1) * args.num_processes')
#                     print(total_num_steps + (step + 1) * args.num_processes)
                    
                    record_num_steps.append(total_num_steps + (step + 1) * args.num_processes)
                    
#                     sys.exit()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy, update_signal = agent.update(rollouts)
        
        if update_signal == -1:
#             sys.exit()
            break

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          getattr(get_vec_normalize(envs), 'ob_rms', None)]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards), dist_entropy,
                       value_loss, action_loss))

        if (args.eval_interval is not None
                and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            eval_envs = make_vec_envs(
                args.env_name, args.seed + args.num_processes, args.num_processes,
                args.gamma, eval_log_dir, args.add_timestep, device, True)

            vec_norm = get_vec_normalize(eval_envs)
            if vec_norm is not None:
                vec_norm.eval()
                vec_norm.ob_rms = get_vec_normalize(envs).ob_rms

            eval_episode_rewards = []

            obs = eval_envs.reset()
            eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                            actor_critic.recurrent_hidden_state_size, device=device)
            eval_masks = torch.zeros(args.num_processes, 1, device=device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                        obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

                # Obser reward and next obs
                obs, reward, done, infos = eval_envs.step(action)

                eval_masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                                for done_ in done])
                for info in infos:
                    if 'episode' in info.keys():
                        eval_episode_rewards.append(info['episode']['r'])

            eval_envs.close()

            print(" Evaluation using {} episodes: mean reward {:.5f}\n".
                format(len(eval_episode_rewards),
                       np.mean(eval_episode_rewards)))

        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.env_name,
                                  args.algo, args.num_frames)
            except IOError:
                pass
            
    print('record_rewards')
    print(record_rewards)
    
    dir_with_params = args.env_name + '/' +\
    args.algo + '/' +\
    'eps_' + str(args.eps) + '/' +\
    'lr_' + str(args.lr) + '/' +\
    'stat_decay_' + str(args.stat_decay) + '/'
    
#     saving_dir = './result/' + args.env_name + '/' + args.algo + '/'
    saving_dir = './result/' + dir_with_params
    
    if not os.path.isdir(saving_dir):
        os.makedirs(saving_dir)
    
    import pickle
    
    with open(saving_dir + 'result.pkl', 'wb') as handle:
        pickle.dump({'record_rewards': record_rewards, 'record_num_steps': record_num_steps}, handle)
        
    print('args.log_dir')
    print(args.log_dir)
    
    print('os.listdir(args.log_dir)')
    print(os.listdir(args.log_dir))
    
#     saving_dir_monitor = './result_monitor/' + args.env_name + '/' + args.algo + '/'

    

    saving_dir_monitor = './result_monitor/' + dir_with_params
    
    if os.path.isdir(saving_dir_monitor):
        import shutil
        
        shutil.rmtree(saving_dir_monitor)
    
    if not os.path.isdir(saving_dir_monitor):
        os.makedirs(saving_dir_monitor)
        
    print('saving_dir_monitor')
    print(saving_dir_monitor)
        
    print('os.listdir(saving_dir_monitor)')
    print(os.listdir(saving_dir_monitor))
    
    import shutil
    
    for file_name in os.listdir(args.log_dir):
        
        full_file_name = os.path.join(args.log_dir, file_name)
        
        print('full_file_name')
        print(full_file_name)
        
        print('os.path.isfile(full_file_name)')
        print(os.path.isfile(full_file_name))
        
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, saving_dir_monitor)
            
        
        
#     full_file_name = os.path.join(src, file_name)
#     if os.path.isfile(full_file_name):
#         shutil.copy(full_file_name, dest)
    
#     sys.exit()


if __name__ == "__main__":
    main()
