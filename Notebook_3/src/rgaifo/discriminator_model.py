import torch
import torch.nn as nn
import numpy as np
from utils import log_sum_exp

class Discriminator(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_dim, env):
        super(Discriminator, self).__init__()

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim

        self.env = env

        actv = nn.Tanh
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), actv(),
            #nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, 1, bias=False))

        # log(normalization-constant)
        self.logZ = nn.Parameter(torch.ones(1))
        self.action_one_hot = np.eye(ac_dim)
        input = []

        #std = self.inputs.std(0)
        #print(std.shape)
        #print(self.inputs.shape)
        #self.inputs = (self.inputs - self.inputs.mean(0))/std
        #print(self.inputs.shape, "SHAPE")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        # self.device = device
        # self.to(device)
        self.train()

    def forward(self):
        raise NotImplementedError()

    def update(self, policy_net, expert_states, learner_states, learner_ids, num_grad_steps):
        self.train()
        loss_val = 0
        n = 0
        epsilon_policy = 1e-5
        #mu_learner_actions = torch.FloatTensor(mu_learner_actions).view(self.env.n_states*self.env.n_actions, 1)
        self.policy_torch = torch.FloatTensor((policy + epsilon_policy)/(1 + epsilon_policy))
        self.log_probs = torch.log(self.policy_torch).view(self.env.n_states*self.env.n_actions, 1)

        for _ in range(num_grad_steps):
            
            buffer_logp = self.tower(expert_states_pair)
            learner_logp = self.tower(learner_states_pair)

            learner_logq = learner_log_probs + self.logZ.expand_as(learner_log_probs)
            buffer_logq = buffer_log_probs + self.logZ.expand_as(buffer_log_probs)

            learner_log_pq = torch.cat([learner_logp, learner_logq], dim=1)
            learner_log_pq = log_sum_exp(learner_log_pq, dim=1, keepdim=True)

            buffer_log_pq = torch.cat([buffer_logp, buffer_logq], dim=1)
            buffer_log_pq = log_sum_exp(buffer_log_pq, dim=1, keepdim=True)

            learner_loss = -(learner_logq - learner_log_pq).mean(0)
            buffer_loss = -(buffer_logp - buffer_log_pq).mean(0)

            reward_bias = (-torch.cat([learner_logp, buffer_logp], dim=0)).clamp_(min=0).mean(0)
            loss = buffer_loss + learner_loss + 2*reward_bias
            
            
            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()
        
        return loss_val / n

    def predict_batch_rewards(self, mus):
        with torch.no_grad():
            self.eval()
            f = self.tower(self.inputs)
        f = f.detach().numpy().reshape(mus.shape)
        #log_probs = self.log_probs.detach().numpy().reshape(mus.shape)
        """exp_f = np.exp(f)
        probs = self.policy_torch.detach().numpy().reshape(mus.shape)
        D = exp_f/(exp_f + probs)
        rewards = np.log(D) - np.log(1 - D)
        return rewards #mus*rewards #- log_probs"""
        return f