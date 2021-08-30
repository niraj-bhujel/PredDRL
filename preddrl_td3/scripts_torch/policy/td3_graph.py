import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as tdist

import dgl

from misc.huber_loss import huber_loss
from policy.td3 import TD3
from networks.gated_gcn_net import GatedGCNNet
from utils.graph_utils import node_type_list, state_dims
from layers.mlp_layer import MLP

class Actor(nn.Module):
    def __init__(self, net_params, args, action_dim, max_action, **kwargs):
        # super(Actor, self).__init__()
        super(Actor, self).__init__()

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

        self.out = MLP(net_params['net']['hidden_dim'], action_dim, hidden_size=net_params['mlp']['hidden_size'])

        self.max_action = max_action
        self.input_states = args.input_states
        self.input_edges = args.input_edges

        self.relu = nn.ReLU()

    def forward(self, g):
        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)

        g, h, e = self.net(g, h, e)
        h = dgl.readout_nodes(g, 'h', op='mean') # (bs, hdim)
        h = self.out(h)

        h = self.max_action * torch.tanh(h)
        return h

class Critic(nn.Module):
    def __init__(self, net_params, args, action_dim, **kwargs):
        super(Critic, self).__init__()

        self.input_states = args.input_states
        self.input_edges = args.input_edges

        net_params['net']['in_dim_node'] = sum([state_dims[s] for s in args.input_states])
        net_params['net']['in_dim_edge'] = sum([state_dims[s] for s in args.input_edges])

        self.net1 = GatedGCNNet(net_params['net'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)

        self.net2 = GatedGCNNet(net_params['net'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)

        self.out1 = MLP(net_params['net']['hidden_dim'] + action_dim, 1, hidden_size=net_params['mlp']['hidden_size'])
        self.out2 = MLP(net_params['net']['hidden_dim'] + action_dim, 1, hidden_size=net_params['mlp']['hidden_size'])

    def forward(self, g, a):
        '''a: (bs, 1)'''

        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)
        
        g1, h1, _ = self.net1(g, h, e)
        h1 = dgl.readout_nodes(g1, 'h', op='mean') # (bs, 1)
        h1 = torch.cat([h1, a], dim=-1)
        h1 = self.out1(h1) 

        g2, h2, _ = self.net2(g, h, e)
        h2 = dgl.readout_nodes(g2, 'h', op='mean') # (bs, 1)
        h2 = torch.cat([h2, a], dim=-1)
        h2 = self.out2(h2)       

        return h1, h2

    def Q1(self, g, a):
        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)
        
        g, h, _ = self.net1(g, h, e)
        h = dgl.readout_nodes(g, 'h', op='mean') # (bs, 1)
        h = torch.cat([h, a], dim=-1)
        h = self.out1(h)

        return h

class GraphTD3(TD3):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="GraphTD3",
            max_action=1.,
            actor_update_freq=2,
            lr_actor=0.001,
            lr_critic=0.001,
            net_params=None,
            args=None,
            **kwargs):

        super().__init__(state_shape, action_dim, name=name, actor_update_freq=actor_update_freq, **kwargs)

        # Define and initialize Actor network
        self.actor = Actor(net_params,  args, action_dim, max_action, **kwargs)
        self.actor_target = Actor(net_params, args, action_dim, max_action, **kwargs)
        self.soft_update_of_target_network(self.actor, self.actor_target)

        # Define and initialize Critic network
        self.critic = Critic(net_params, args, action_dim, **kwargs)
        self.critic_target = Critic(net_params, args, action_dim, **kwargs)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        # define optimizers
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic, eps=1e-4)

        self._actor_update_freq = actor_update_freq



