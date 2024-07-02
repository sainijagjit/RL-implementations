import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC
from gymnasium.wrappers import RescaleAction
import sys
# from utils import MyWalkerWrapper


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default='BipedalWalker-v3',
                    help='Mujoco Gym environment (default: LunarLander-v2)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: True)')
parser.add_argument('--seed', type=int, default=12, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=5, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--max_memory_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--grad_clip', type=bool, default=False, metavar='N',
                    help='Gradient clipping (default: True)')
parser.add_argument('--model_saving_frequency', type=int, default=10, metavar='N',
                    help='Model Saving Frequency(default: 500)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--eval_mode', type=bool, default=True, help='eval mode')
parser.add_argument('--checkpoint_name', default='sac_checkpoint_BipedalWalker-v3_500', help='checkpoint name for evaluation')

args = parser.parse_args()


# Environment
env = gym.make(args.env_name)
env = RescaleAction(env, -1, 1)
# env = MyWalkerWrapper(env)
env.reset(seed=args.seed)
env.action_space.seed(args.seed)

env1 = gym.make(args.env_name,  render_mode='human')
env1 = RescaleAction(env1, -1, 1)
env1.reset(seed=args.seed)
env1.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
# writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                              args.policy, "autotune" if args.automatic_entropy_tuning else ""))


# Training Loop
total_numsteps = 0
updates = 0


def evaluate_model(checkpoint_name,episodes):
    agent.load_checkpoint(checkpoint_name,evaluate=True)
    avg_reward = 0
    agent.policy.eval()
    for _  in range(episodes):
        state, info = env1.reset()
        episode_reward_ = 0
        done = False
        trunc = False
        while not done and not trunc:
            # action, _ =  env1.action_space.sample(),1
            action, _ = agent.select_action(state, evaluate=True)
            next_state, reward, done, trunc, info = env1.step(action)
            if not done:
                episode_reward_ += reward
            state = next_state
        avg_reward += episode_reward_
        print("----------------------------------------")
        print("Episode Reward: {}".format(episode_reward_))
        print("----------------------------------------")
    avg_reward /= episodes
    agent.policy.train()
    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")
 
if args.eval_mode:
    evaluate_model(args.checkpoint_name,episodes=15)   
    sys.exit()

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    trunc = False
    state, info = env.reset()

    while not done and not trunc:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action, log_pi = agent.select_action(state)  # Sample action from policy

        if len(agent.memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss = agent.update(updates)
                # if(total_numsteps % 2000 == 0):
                #     print(f"critic_1_loss: {critic_1_loss}, critic_2_loss: {critic_2_loss}, policy_loss: {policy_loss}")

                # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)
                # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, trunc, info = env.step(action) # Step
        if done:
            reward = -5
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if trunc else float(done)
        
        agent.memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    # writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    
    if i_episode % args.model_saving_frequency == 0:
        agent.save_checkpoint(args.env_name,i_episode)
        
    if i_episode % 10 == 0 and args.eval is True:
        episodes = 1
        evaluate_model(episodes)
        # writer.add_scalar('avg_reward/test', avg_reward, i_episode)
env.close()
