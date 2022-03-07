import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dgl
from dgl.heterograph import DGLHeteroGraph

from policy.policy_base import OffPolicyAgent
from misc.huber_loss import huber_loss


# from exploration_strategies.OU_Noise_Exploration import OU_Noise_Exploration

class Actor(nn.Module):
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


    def forward(self, states, actions):
        # states, actions = inputs
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
            net_params=None,
            args=None,
            **kwargs):

        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau
        self.action_dim = action_dim
        self.state_shape = state_shape

        # Define and initialize Actor network
        self.actor = Actor(state_shape, action_dim, max_action, actor_units).to(self.device)
        self.actor_target = Actor(state_shape, action_dim, max_action, actor_units).to(self.device)
        self.soft_update_of_target_network(self.actor, self.actor_target,)


        # Define and initialize Critic network
        self.critic = Critic(state_shape, action_dim, critic_units).to(self.device)
        self.critic_target = Critic(state_shape, action_dim, critic_units).to(self.device)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        # define optimizers
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic, eps=1e-4)

        self.iteration=n_warmup

    def get_action(self, state, test=False, tensor=False):

        if isinstance(state, DGLHeteroGraph):
            state = dgl.batch([state]).to(self.device) 
        else:
            state = torch.Tensor(state).to(self.device)

        action = self._get_action_body(state, 
                                       self.sigma * (1. - test), 
                                       self.actor.max_action)
        if tensor:
            return action
        else:
            return dgl.unbatch(state)[0].cpu(), action.cpu().numpy()


    def _get_action_body(self, state, sigma, max_action):
        self.eval()
        with torch.no_grad():
            action = self.actor(state)
            # action += torch.empty_like(action).normal_(mean=0,std=sigma)
        self.train()
        # return torch.clamp(action, -max_action, max_action)
        return action.squeeze(0)

    def optimization_step(self, optimizer, loss, clip_norm=None, model=None, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm) #clip gradients to help stabilise training

        optimizer.step()

    def soft_update_of_target_network(self, local_model, target_model):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def train_step(self, states, actions, next_states, rewards, dones, weights):
        # print('states:', states.number_of_nodes(), 'actions:', actions.shape, 'rewards:', rewards.shape, 'dones:', dones.shape, 'weights', weights.shape)

        self.iteration +=1 

        actor_loss, critic_loss, td_errors = self._train_body(states, actions, next_states, rewards, dones, weights)

        actor_loss = actor_loss.item() if actor_loss is not None else actor_loss
        critic_loss = critic_loss.item()
        td_errors = td_errors.detach().cpu().numpy()

        if actor_loss is not None:

            self.writer.add_scalar(self.policy_name + "/actor_loss", actor_loss, self.iteration)
            self.writer.add_scalar(self.policy_name + "/critic_loss", critic_loss, self.iteration)
            self.writer.add_scalar(self.policy_name + "/batch_rewards", rewards.mean().item(), self.iteration)

            print('STEP:{}, batch_rewards:{:.2f}, actor_loss:{:.5f}, critic_loss:{:.5f}'.format(self.iteration, 
                                                                                                rewards.mean().item(),
                                                                                                actor_loss,
                                                                                                critic_loss,))

        return actor_loss, critic_loss, td_errors

    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        # Compute critic loss

        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)
        critic_loss = torch.mean(huber_loss(td_errors, delta=self.max_grad) * weights)

        # Optimize the critic
        self.optimization_step(self.critic_optimizer, critic_loss)

        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Optimize the actor 
        self.optimization_step(self.actor_optimizer, actor_loss)

        # Update target networks
        self.soft_update_of_target_network(self.actor, self.actor_target)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones):
        
        with torch.no_grad():
            td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)

        return np.abs(np.ravel(td_errors.cpu().numpy()))

    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):

        not_dones = 1. - dones

        with torch.no_grad():
            target_Q = self.critic_target(next_states, self.actor_target(next_states))

            target_Q = rewards + (not_dones * self.discount * target_Q)

        current_Q = self.critic(states, actions)

        td_errors = target_Q - current_Q

        return td_errors
