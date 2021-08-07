

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import tensorflow as tf

from policy.policy_base_torch import OffPolicyAgent
from misc.huber_loss import huber_loss

# from exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration

class Actor(torch.nn.Module):
    def __init__(self, state_shape, action_dim, max_action, units=[256, 256], name="Actor"):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_shape[0], units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], action_dim)

        self.max_action = max_action

    def forward(self, inputs):
        #DNN
        features = F.relu(self.l1(inputs))
        features = F.relu(self.l2(features))
        features = self.l3(features)

        action = self.max_action * torch.tanh(features)

        return action

class Critic(nn.Module):
    def __init__(self, state_shape, action_dim, units=[256, 256], name="Critic"):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_shape[0]+action_dim, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], 1)


    def forward(self, inputs):
        states, actions = inputs
        features = torch.cat([states, actions], axis=1)
        features = F.relu(self.l1(features))
        features = F.relu(self.l2(features))
        features = self.l3(features)
        return features


class DDPG(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="DDPG",
            max_action=1.,
            lr_actor=0.001,
            lr_critic=0.001,
            actor_units=[400, 300],
            critic_units=[400, 300],
            sigma=0.1,
            tau=0.005,
            n_warmup=int(1e4),
            memory_capacity=int(1e6),
            **kwargs):

        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)


        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau

        # Define and initialize Actor network
        self.actor = Actor(state_shape, action_dim, max_action, actor_units).to(self.device)

        self.actor_target = Actor(state_shape, action_dim, max_action, actor_units).to(self.device)

        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=lr_actor)

        self.actor_target.load_state_dict(self.actor.state_dict())

        # Define and initialize Critic network
        self.critic = Critic(state_shape, action_dim, critic_units).to(self.device)

        self.critic_target = Critic(state_shape, action_dim, critic_units).to(self.device)

        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic, eps=1e-4)

        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(self, state, test=False, tensor=False):

        is_single_state = len(state.shape) == 1

        if not tensor:
            assert isinstance(state, np.ndarray)

        state = np.expand_dims(state, axis=0).astype(np.float32) if is_single_state else state

        action = self._get_action_body(torch.from_numpy(state).to(self.device), 
                                       self.sigma * (1. - test), 
                                       self.actor.max_action)
        if tensor:
            return action
        else:
            return action.cpu().numpy()[0] if is_single_state else action.numpy()


    def _get_action_body(self, state, sigma, max_action):

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)

        self.actor.train()

        return torch.clamp(action, -max_action, max_action)

    def train(self, states, actions, next_states, rewards, done, weights=None):

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        done = torch.from_numpy(done).to(self.device)

        # print(states.shape, actions.shape, next_states.shape, rewards.shape)

        if weights is not None:
            weights = torch.ones_like(rewards).to(self.device)

        actor_loss, critic_loss, td_errors = self._train_body(states, actions, next_states, rewards, done, weights)

        # optimization step
        self.optimization_step(self.actor_optimizer, self.actor, actor_loss)
        self.optimization_step(self.critic_optimizer, self.critic, critic_loss)

        # Update target networks
        self.soft_update_of_target_network(self.actor, self.actor_target, tau=self.tau)
        self.soft_update_of_target_network(self.critic, self.critic_target, tau=self.tau)

        return actor_loss.item(), critic_loss.item(), td_errors.detach().cpu().numpy()

    def optimization_step(self, optimizer, network, loss, clip_norm=None, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm) #clip gradients to help stabilise training

        optimizer.step()

    def _train_body(self, states, actions, next_states, rewards, done, weights):

        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, done)

        critic_loss = torch.mean(huber_loss(td_errors, delta=self.max_grad) * weights)

        next_action = self.actor(states)
        actor_loss = -torch.mean(self.critic([states, next_action]))

        return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        
        with torch.no_grad():
            td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)

        return np.abs(np.ravel(td_errors.cpu().numpy()))

    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):

        not_dones = 1. - dones

        with torch.no_grad():
            target_Q = self.critic_target([next_states, self.actor_target(next_states)])
            target_Q = rewards + (not_dones * self.discount * target_Q)

        current_Q = self.critic([states, actions])

        td_errors = target_Q - current_Q

        return td_errors

    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


