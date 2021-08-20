

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl

from policy.policy_base import OffPolicyAgent
from misc.huber_loss import huber_loss
from networks.gated_gcn_net import GatedGCNNet
from utils.graph_utils import node_type_list, state_dims

class Actor(nn.Module):
    def __init__(self, config, args, action_dim, max_action, **kwargs):
        # super(Actor, self).__init__()
        super(Actor, self).__init__()

        config['net']['in_dim_node'] = sum([state_dims[s] for s in args.input_states])
        config['net']['in_dim_edge'] = sum([state_dims[s] for s in args.input_edges])
        config['net']['out_dim_node'] = action_dim
        self.net = GatedGCNNet(config['net'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)

        self.max_action = max_action
        self.input_states = args.input_states
        self.input_edges = args.input_edges

    def forward(self, g):
        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)
        g, h, e = self.net(g, h, e)

        h = self.max_action * torch.tanh(h)

        return h

class Critic(nn.Module):
    def __init__(self, config, args, action_dim, **kwargs):
        super(Critic, self).__init__()

        self.input_states = args.input_states
        self.input_edges = args.input_edges

        config['net']['in_dim_node'] = sum([state_dims[s] for s in args.input_states]) + action_dim
        config['net']['in_dim_edge'] = sum([state_dims[s] for s in args.input_edges])
        config['net']['out_dim_node'] = 1
        self.net = GatedGCNNet(config['net'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)

    def forward(self, g, a):
        
        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        h = torch.cat([h, a], dim=-1)
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)
        _, h, _ = self.net(g, h, e)
        # g.ndata['qvalue'] = h
        return h


class GraphDDPG(OffPolicyAgent):
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
            config=None,
            args=None,
            **kwargs):

        super().__init__(name=name, memory_capacity=memory_capacity, n_warmup=n_warmup, **kwargs)

        # Set hyperparameters
        self.sigma = sigma
        self.tau = tau
        self.action_dim = action_dim

        # Define and initialize Actor network
        self.actor = Actor(config,  args, action_dim, max_action, **kwargs)
        self.actor_target = Actor(config, args, action_dim, max_action, **kwargs)
        self.soft_update_of_target_network(self.actor, self.actor_target)

        # Define and initialize Critic network
        self.critic = Critic(config, args, action_dim, **kwargs)
        self.critic_target = Critic(config, args, action_dim, **kwargs)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        # define optimizers
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic, eps=1e-4)

        self.step = 0

    def get_action(self, state, test=False, tensor=False):
        state = state.to(self.device)
        action = self._get_action_body(state, 
                                       self.sigma * (1. - test), 
                                       self.actor.max_action)
        if tensor:
            return action
        else:
            return action.cpu().numpy()

    def _get_action_body(self, state, sigma, max_action):

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)

            action += torch.empty_like(action).normal_(mean=0,std=sigma)

        self.actor.train()

        return torch.clamp(action, -max_action, max_action)

    def train(self, states, actions, next_states, rewards, dones, weights, step=0):
        self.step += 1
        states = dgl.batch(states).to(self.device)
        next_states = dgl.batch(next_states).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).view(-1, self.action_dim).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).view(-1, 1).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).view(-1, 1).to(self.device)
        weights = torch.from_numpy(np.array(weights)).to(self.device)

        # print(actions.shape, rewards.shape, dones.shape, weights.shape)
        # torch.Size([100, 24]) torch.Size([100, 2]) torch.Size([100, 24]) torch.Size([100, 1]) torch.Size([100, 1]) torch.Size([100])
        
        actor_loss, critic_loss, td_errors = self._train_body(states, actions, next_states, rewards, dones, weights)

        actor_loss = actor_loss.item() if actor_loss is not None else actor_loss

        return actor_loss , critic_loss.item(), td_errors.detach().cpu().numpy()

    def optimization_step(self, optimizer, loss, clip_norm=None, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm) #clip gradients to help stabilise training

        optimizer.step()

    def _train_body(self, states, actions, next_states, rewards, dones, weights):

        # Compute critic loss
        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)
        critic_loss = torch.mean(huber_loss(td_errors, delta=self.max_grad) * weights)

        # Optimize the critic
        self.optimization_step(self.critic_optimizer,  critic_loss)

        # Compute actor loss
        next_action = self.actor(states)
        actor_loss = -self.critic(states, next_action).mean()

        # Optimize the actor 
        self.optimization_step(self.actor_optimizer, actor_loss)

        # Update target networks
        self.soft_update_of_target_network(self.actor, self.actor_target)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        self.writer.add_scalar(self.policy_name + "/actor_loss", actor_loss, self.step)
        self.writer.add_scalar(self.policy_name + "/critic_loss", critic_loss, self.step)

        return actor_loss, critic_loss, td_errors

    def compute_td_error(self, states, actions, next_states, rewards, dones, phase='test'):
        states = dgl.batch(states).to(self.device)
        next_states = dgl.batch(next_states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device).view(-1, self.action_dim)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).view(-1, 1)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device).view(-1, 1)
        
        with torch.no_grad():
            td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones, phase)

        return td_errors.cpu().numpy()

    def _compute_td_error_body(self, states, actions, next_states, rewards, dones, phase='train'):

        not_dones = 1. - dones

        with torch.no_grad():
            next_actions = self.actor_target(next_states)

            target_Q = self.critic_target(next_states, next_actions)

            # q value for robot only
            target_Q = target_Q[next_states.ndata['cid']==node_type_list.index('robot')]
            
            target_Q = rewards + (not_dones * self.discount * target_Q)

        current_Q = self.critic(states, actions)

        # q value of robot
        current_Q = current_Q[states.ndata['cid']==node_type_list.index('robot')]

        td_errors = target_Q - current_Q

        # NOTE adding writer here will also add while called from compute_td_error
        if phase=='train':
            self.writer.add_histogram(self.policy_name + "/next_actions", next_actions, self.step)
            self.writer.add_scalar(self.policy_name + '/current_Q', current_Q.mean().item(), self.step)
            self.writer.add_scalar(self.policy_name + '/target_Q', target_Q.mean().item(), self.step)

        return td_errors

    def soft_update_of_target_network(self, local_model, target_model):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


