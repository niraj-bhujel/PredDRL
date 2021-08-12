import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as tdist

from policy.policy_base_torch import OffPolicyAgent


from misc.huber_loss import huber_loss

from policy.ddpg_torch import DDPG, Actor


class Critic(nn.Module):
    def __init__(self, state_shape, action_dim, units=[400, 300], name="Critic"):
        super(Critic, self).__init__()

        # Q1
        self.l1 = nn.Linear(state_shape[0]+action_dim, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], 1)

        #Q2
        self.l4 = nn.Linear(state_shape[0]+action_dim, units[0])
        self.l5 = nn.Linear(units[0], units[1])
        self.l6 = nn.Linear(units[1], 1)
            

    def forward(self, states, actions):
        
        xu = torch.cat([states, actions], axis=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2

    def Q1(self, states, actions):
        sa = torch.cat([states, actions], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3(DDPG):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="TD3",
            actor_update_freq=2,
            policy_noise=0.2,
            noise_clip=0.5,
            actor_units=[400, 300],
            critic_units=[400, 300],
            lr_critic=0.001,
            **kwargs):
        super().__init__(name=name, state_shape=state_shape, action_dim=action_dim,
                         actor_units=actor_units, critic_units=critic_units,
                         lr_critic=lr_critic, **kwargs)

        self.critic = Critic(state_shape, action_dim, critic_units)
        self.critic_target = Critic(state_shape, action_dim, critic_units)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic, eps=1e-4)

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

        self._actor_update_freq = actor_update_freq
        self.total_it = 0

    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        
        self.total_it += 1

        # Compute critic loss
        td_error1, td_error2 = self._compute_td_error_body(states, actions, next_states, rewards, dones)
        critic_loss = torch.mean(huber_loss(td_error1, delta=self.max_grad) * weights) + \
                      torch.mean(huber_loss(td_error2, delta=self.max_grad) * weights)

        # # Optimize the critic
        self.optimization_step(self.critic_optimizer, critic_loss)

        actor_loss = None
        # Delayed policy updates
        if self.total_it % self._actor_update_freq==0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean() # based on original source code
            # actor_loss = -torch.cat(self.critic(states, next_actions)).mean()

            self.optimization_step(self.actor_optimizer,  actor_loss)  

            # Update target networks
            self.soft_update_of_target_network(self.actor, self.actor_target)
            self.soft_update_of_target_network(self.critic, self.critic_target)

        return actor_loss, critic_loss, torch.abs(td_error1) + torch.abs(td_error2)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            td_errors1, td_errors2 = self._compute_td_error_body(states, actions, next_states, rewards, dones)

        return np.squeeze(np.abs(td_errors1.cpu().numpy()) + np.abs(td_errors2.cpu().numpy()))


    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):

        not_dones = 1. - dones

        with torch.no_grad():

            # generate noise
            noise = (torch.randn_like(actions) * self._policy_noise).clamp(-self._noise_clip, self._noise_clip)

            # Get noisy action
            next_action = self.actor_target(next_states)
            next_action = (next_action + noise).clamp(-self.actor_target.max_action, self.actor_target.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_states, next_action)

            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = rewards + not_dones * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(states, actions)


        return target_Q - current_Q1, target_Q - current_Q2
