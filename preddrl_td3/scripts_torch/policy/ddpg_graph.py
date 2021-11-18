

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

import dgl
from dgl.heterograph import DGLHeteroGraph

from policy.ddpg import DDPG
from policy.policy_base import OffPolicyAgent
from misc.huber_loss import huber_loss

from networks.gated_gcn_net import GatedGCNNet
from utils.graph_utils import node_type_list, state_dims
from layers.mlp_layer import MLP

class Actor(nn.Module):
    def __init__(self, net_params, args, state_shape, action_dim, max_action, **kwargs):
        # super(Actor, self).__init__()
        super(Actor, self).__init__()
        pred_len = kwargs.get('pred_len')
        
        net_params['net']['in_dim_node'] = sum([state_dims[s] for s in args.input_states])
        net_params['net']['in_dim_edge'] = sum([state_dims[s] for s in args.input_edges])

        # net_params['net']['out_dim_node'] = action_dim
        self.net = GatedGCNNet(net_params['net'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)

        self.out = MLP(net_params['net']['hidden_dim'], 2, hidden_size=net_params['mlp']['hidden_size'])
        
        # self.l1 = nn.Linear(net_params['net']['hidden_dim'] + state_shape[0], net_params['mlp']['hidden_size'][0])
        # self.l2 = nn.Linear(net_params['mlp']['hidden_size'][0], net_params['mlp']['hidden_size'][1])

        # self.l3 = nn.Linear(net_params['mlp']['hidden_size'][1], 1)
        # self.l4 = nn.Linear(net_params['mlp']['hidden_size'][1], 1)

        self.max_action = max_action
        self.input_states = args.input_states
        self.input_edges = args.input_edges

    def forward(self, state):
        g = state
        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)

        g, h, e = self.net(g, h, e)
        
        h = self.out(h)
        h = -1.2*torch.tanh(h)

        # h = dgl.readout_nodes(g, 'h', op='mean') # (bs, hdim)
        
        # h = torch.cat([h, state], dim=-1)
        # h = F.relu(self.l1(h))
        # h = F.relu(self.l2(h))

        # v = self.max_action[0]*torch.sigmoid(self.l3(h))
        # w = self.max_action[1]*torch.tanh(self.l4(h))
        # h = torch.cat([v, w], dim=-1)
        
        return h

class Critic(nn.Module):
    def __init__(self, net_params, args, state_shape, action_dim, **kwargs):
        super(Critic, self).__init__()

        self.input_states = args.input_states
        self.input_edges = args.input_edges

        net_params['net']['in_dim_node'] = sum([state_dims[s] for s in args.input_states])
        net_params['net']['in_dim_edge'] = sum([state_dims[s] for s in args.input_edges])
        # net_params['net']['out_dim_node'] = 1
        self.net = GatedGCNNet(net_params['net'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)
        
        self.out = MLP(net_params['net']['hidden_dim'] + action_dim, 1, hidden_size=net_params['mlp']['hidden_size'])

        # self.l1 = nn.Linear(net_params['net']['hidden_dim'] + state_shape[0] + action_dim, net_params['mlp']['hidden_size'][0])
        # self.l2 = nn.Linear(net_params['mlp']['hidden_size'][0], net_params['mlp']['hidden_size'][1])
        # self.l3 = nn.Linear(net_params['mlp']['hidden_size'][1], 1)

    def forward(self, state, action):
        g = state
        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)
        
        g, h, _ = self.net(g, h, e)

        # h = dgl.readout_nodes(g, 'h', op='mean') # (bs, 1)

        # h = torch.cat([h, state, action], dim=-1)
        # h = F.relu(self.l1(h))
        # h = F.relu(self.l2(h))
        # h = self.l3(h)

        h = torch.cat([h, action], dim=-1)
        h = self.out(h)
        g.ndata['h'] = h
        h = dgl.readout_nodes(g, 'h', op='mean') # (bs, 1)

        return h


class GraphDDPG(DDPG):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="GraphDDPG",
            max_action=(1., 1.),
            lr_actor=1e-4,
            lr_critic=1e-4,
            net_params=None,
            args=None,
            **kwargs):

        super().__init__(state_shape, action_dim, name=name, **kwargs)

        # Define and initialize Actor network
        self.actor = Actor(net_params,  args, state_shape, action_dim, max_action, **kwargs)
        self.actor_target = Actor(net_params, args, state_shape, action_dim, max_action, **kwargs)
        self.soft_update_of_target_network(self.actor, self.actor_target)

        # Define and initialize Critic network
        self.critic = Critic(net_params, args, state_shape, action_dim, **kwargs)
        self.critic_target = Critic(net_params, args, state_shape, action_dim, **kwargs)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        # define optimizers
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic, eps=1e-4)


    def _train_body(self, states, actions, next_states, rewards, dones, weights):
        # Compute critic loss

        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)
        critic_loss = torch.mean(huber_loss(td_errors, delta=self.max_grad) * weights)

        # Optimize the critic
        self.optimization_step(self.critic_optimizer, critic_loss)

        # Compute actor loss
        action = self.actor(states)
        actor_loss = -self.critic(states, action).mean()

        # compute correct action reward for agents
        ped_mask = (states.ndata['cid']==node_type_list.index('pedestrian')).unsqueeze(1)
        action_error = torch.norm((states.ndata['action'] - action)*ped_mask)
        self.writer.add_scalar("Common/action_error", action_error, self.iteration)

        actor_loss += action_error


        # Optimize the actor 
        self.optimization_step(self.actor_optimizer, actor_loss)

        # Update target networks
        self.soft_update_of_target_network(self.actor, self.actor_target)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        return actor_loss, critic_loss, td_errors
