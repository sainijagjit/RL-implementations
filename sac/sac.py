import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import CriticNetwork,PolicyNetwork
from replay_memory import ReplayMemory
from torch.distributions import Normal
from utils import soft_update, hard_update_and_freeze_gradient
import numpy as np


class SAC():
    def __init__(self,num_inputs,action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval
        self.batch_size = args.batch_size
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.grad_clip = args.grad_clip
        
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.critic_1 = CriticNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_2 = CriticNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=args.lr)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=args.lr)
        
        self.critic_target_1 = CriticNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        self.critic_target_2 = CriticNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        
        hard_update_and_freeze_gradient(self.critic_target_1,self.critic_1)
        hard_update_and_freeze_gradient(self.critic_target_2,self.critic_2)

        
        self.policy = PolicyNetwork(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
        
        self.memory = ReplayMemory(args.max_memory_size, args.seed)
        
        
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mean, log_std = self.policy(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        # torch.rsample does reparametrization trick automatically (mean + std * N(0,1))
        # normal = Normal(0, 1)
        # epsilon  = normal.sample().to(self.device)
        # action = mean + std * epsilon
        if evaluate:
            action = torch.tanh(mean)
            log_pi = None
        else:
            action = normal.rsample()
            # In the Soft Actor-Critic (SAC) algorithm, when using a reparameterization trick with the Tanh squashing function to bound actions, it's necessary to adjust the calculation of the log probability of the sampled actions. This adjustment is needed because the transformation of the action distribution through a non-linear function (like the Tanh function) changes the original probability distribution of the actions.
            log_pi = normal.log_prob(action)
            log_pi = log_pi - torch.log((1- action.tanh()**2) + 1e-6)
            # The operation log_pi.sum(axis=-1) in the context of your PyTorch code is used to sum the log probabilities across the dimensions of the action space. This is necessary in environments where the agent can take multiple actions simultaneously (i.e., multi-dimensional actions), and you need a single scalar value representing the total log probability of the entire action vector. 
            action = torch.tanh(action) 
            log_pi = log_pi.squeeze(0).sum(axis=-1, keepdim=True)
        action = action.cpu().detach().numpy()[0]
        return action, log_pi
        
        
    def update(self, updates):  
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        # Critic Update
        with torch.no_grad():
            next_action_batch, log_pi = self.select_action(next_state_batch)
            next_action_batch = torch.FloatTensor(next_action_batch).to(self.device)
            Q1_next_target = self.critic_target_1(next_state_batch, next_action_batch)
            Q2_next_target = self.critic_target_2(next_state_batch, next_action_batch)
            Min_Q_next_target = torch.min(Q1_next_target,Q2_next_target) - self.alpha * log_pi
            y = reward_batch + mask_batch * self.gamma * Min_Q_next_target
        Q1_current = self.critic_1(state_batch,action_batch) 
        Q2_current = self.critic_2(state_batch,action_batch) 
        Q1_loss = F.mse_loss(Q1_current,y)
        Q2_loss = F.mse_loss(Q2_current,y)
        
        self.critic_1_optim.zero_grad()
        Q1_loss.backward()
        if self.grad_clip:
            for param in self.critic_1.parameters():
                    param.grad.data.clamp_(-1, 1)
        self.critic_1_optim.step()
        
        self.critic_2_optim.zero_grad()
        Q2_loss.backward()
        if self.grad_clip:
            for param in self.critic_2.parameters():
                    param.grad.data.clamp_(-1, 1)
        self.critic_2_optim.step()
        
        # Policy Network Update
        # We don't want to update critic in actor update, hence saving some computation
        for parameters in self.critic_1.parameters():
            parameters.requires_grad = False
        for parameters in self.critic_2.parameters():
            parameters.requires_grad = False
            
        action_batch_pi, log_pi_pi = self.select_action(state_batch)
        action_batch_pi = torch.FloatTensor(action_batch_pi).to(self.device)
        Q1_pi = self.critic_1(state_batch,action_batch_pi )
        Q2_pi = self.critic_2(state_batch,action_batch_pi)
        Q_min = torch.min(Q1_pi,Q2_pi)
        # -1 for gradient accent
        Policy_loss =  -1 * (Q_min - self.alpha * log_pi_pi).mean()
        self.policy_optim.zero_grad()
        Policy_loss.backward()
        if self.grad_clip:
            for param in self.policy.parameters():
                    param.grad.data.clamp_(-1, 1)
        self.policy_optim.step()
        
        for parameters in self.critic_1.parameters():
            parameters.requires_grad = True
        for parameters in self.critic_2.parameters():
            parameters.requires_grad = True
            
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

            
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target_1, self.critic_1, self.tau)
            soft_update(self.critic_target_2, self.critic_2, self.tau)
            
        return Q1_loss.item(), Q2_loss.item(), Policy_loss.item()
        
        
        
    # Save model parameters
    def save_checkpoint(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_1_state_dict': self.critic_1.state_dict(),
                    'critic_2_state_dict': self.critic_2.state_dict(),
                    'critic_target_1_state_dict': self.critic_target_1.state_dict(),
                    'critic_target_2_state_dict': self.critic_target_2.state_dict(),
                    'critic_1_optimizer_state_dict': self.critic_1_optim.state_dict(),
                    'critic_2_optimizer_state_dict': self.critic_2_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)
        
        
    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(f'checkpoints/{ckpt_path}')
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
            self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
            self.critic_target_1.load_state_dict(checkpoint['critic_target_1_state_dict'])
            self.critic_target_2.load_state_dict(checkpoint['critic_target_2_state_dict'])
            self.critic_1_optim.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
            self.critic_2_optim.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.policy.eval()
                self.critic_1.eval()
                self.critic_2.eval()
                self.critic_target_1.eval()
                self.critic_target_2.eval()
            else:
                self.policy.train()
                self.critic_1.train()
                self.critic_2.train()
                self.critic_target_1.train()
                self.critic_target_2.train()
            
            
             
             
             
    