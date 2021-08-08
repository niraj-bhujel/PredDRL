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

        self.l1 = nn.Linear(state_shape[0]+action_dim, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], 1)

        self.l4 = nn.Linear(state_shape[0]+action_dim, units[0])
        self.l5 = nn.Linear(units[0], units[1])
        self.l6 = nn.Linear(units[1], 1)
            

    def forward(self, inputs):
        states, actions = inputs
        
        xu = torch.cat([states, actions], axis=1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)

        return x1, x2


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
        self.soft_update_of_target_network(self.critic, self.critic_target, self.tau)

        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic, eps=1e-4)

        self._policy_noise = policy_noise
        self._noise_clip = noise_clip

        self._actor_update_freq = actor_update_freq
        # self._it = tf.Variable(0, dtype=tf.int32)
        self._it = 0

    def _train_body(self, states, actions, next_states, rewards, done, weights):

        td_error1, td_error2 = self._compute_td_error_body(states, actions, next_states, rewards, done)

        critic_loss = torch.mean(huber_loss(td_error1, delta=self.max_grad) * weights) + \
                      torch.mean(huber_loss(td_error2, delta=self.max_grad) * weights)

        # optimization step
        self.optimization_step(self.critic_optimizer, self.critic, critic_loss)
        self.soft_update_of_target_network(self.critic, self.critic_target, tau=self.tau)
        
        self._it += 1
        next_actions = self.actor(states)

        actor_loss = -torch.cat(self.critic([states, next_actions])).mean()

        if self._it % self._actor_update_freq==0:
            self.optimization_step(self.actor_optimizer, self.actor, actor_loss)        
            # Update target networks
            self.soft_update_of_target_network(self.actor, self.actor_target, tau=self.tau)

        return actor_loss, critic_loss, torch.abs(td_error1) + torch.abs(td_error2)

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        pass


    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):

        not_dones = 1. - dones

        with torch.no_grad():
            # Get noisy action
            next_action = self.actor_target(next_states)
            noise = torch.empty_like(next_action).normal_(mean=0,std=self._policy_noise)
            noise.clamp_(-self._noise_clip, self._noise_clip)

            next_action = torch.clamp(next_action + noise, -self.actor_target.max_action, self.actor_target.max_action)

            target_Q1, target_Q2 = self.critic_target([next_states, next_action])

            target_Q = torch.min(torch.cat([target_Q1, target_Q2], dim=-1), dim=-1)[0].unsqueeze(-1)

            target_Q = rewards + (not_dones * self.discount * target_Q)


        current_Q1, current_Q2 = self.critic([states, actions])

        return target_Q - current_Q1, target_Q - current_Q2
