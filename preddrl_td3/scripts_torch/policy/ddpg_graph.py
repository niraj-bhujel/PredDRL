

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

import dgl

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

        # self.out = MLP(net_params['net']['hidden_dim'] + state_shape[0], net_params['net']['hidden_dim'], hidden_size=net_params['mlp']['hidden_size'])
        self.l1 = nn.Linear(net_params['net']['hidden_dim'] + state_shape[0], net_params['mlp']['hidden_size'][0])
        self.l2 = nn.Linear(net_params['mlp']['hidden_size'][0], net_params['mlp']['hidden_size'][1])

        self.l3 = nn.Linear(net_params['mlp']['hidden_size'][1], 2)
        # self.l4 = nn.Linear(net_params['mlp']['hidden_size'][1], 1)

        self.max_action = max_action
        self.input_states = args.input_states
        self.input_edges = args.input_edges

        # self.relu = nn.ReLU()

    def forward(self, state):
        state, g = state

        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([1/g.edata[s] for s in self.input_edges], dim=-1)

        g, h, e = self.net(g, h, e)
        h = dgl.readout_nodes(g, 'h', op='mean') # (bs, hdim)
        
        h = torch.cat([h, state], dim=-1)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))

        # h = self.l3(h)
        h = self.max_action*torch.tanh(self.l3(h))

        # v = F.relu(self.l3(h))
        # w = self.l4(h)
        # w = self.max_action*torch.tanh(self.l4(h))
        # h = torch.cat([v, w], -1)
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

        self.l1 = nn.Linear(net_params['net']['hidden_dim'] + state_shape[0] + action_dim, net_params['mlp']['hidden_size'][0])
        self.l2 = nn.Linear(net_params['mlp']['hidden_size'][0], net_params['mlp']['hidden_size'][1])
        self.l3 = nn.Linear(net_params['mlp']['hidden_size'][1], 1)

        # self.out = MLP(net_params['net']['hidden_dim'] + state_shape[0] + action_dim, 1, hidden_size=net_params['mlp']['hidden_size'])

    def forward(self, state, action):

        state, g = state

        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([1/g.edata[s] for s in self.input_edges], dim=-1)
        
        g, h, _ = self.net(g, h, e)
        h = dgl.readout_nodes(g, 'h', op='mean') # (bs, 1)

        h = torch.cat([h, state, action], dim=-1)
        # h = self.out(h)
        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        h = self.l3(h)

        return h


class GraphDDPG(DDPG):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="GraphDDPG",
            max_action=1.,
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



    def get_action(self, state, test=False, tensor=False):
        
        state, g = state

        state = torch.Tensor(state).view(1, -1).to(self.device)
        g = dgl.batch([g]).to(self.device)

        action = self._get_action_body([state, g], 
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

        # action = torch.clamp(action, -max_action, max_action)
        return action.squeeze(0)
