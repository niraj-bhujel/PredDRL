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



class GatedGCN(OffPolicyAgent):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="DDPG",
            max_action=1.,
            lr = 0.001,
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
        self.max_action = max_action

        self.input_states = args.input_states
        self.input_edges = args.input_edges

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

        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)

        self.step = 0

    def prepare_inputs(self, g, a=None):

        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        # h = torch.cat([h, a], dim=-1)
        e = torch.cat([g.edata[e] for e in self.input_edges], dim=-1)

        return g, h, e

    def get_action(self, state, test=False, tensor=False):
        state = state.to(self.device)
        action = self._get_action_body(state, 
                                       self.sigma * (1. - test), 
                                       self.max_action)
        if tensor:
            return action
        else:
            return action.cpu().numpy()


    def _get_action_body(self, state, sigma, max_action):

        self.net.eval()
        with torch.no_grad():
            g, h, e = self.prepare_inputs(state)
            _, action, _ = self.net(g, h, e)

            action += torch.empty_like(action).normal_(mean=0,std=sigma)

        self.net.train()

        return torch.clamp(action, -max_action, max_action)

    def train(self, states):
        self.step += 1

        node_loss = 0
        edge_loss = 0

        # Compute node loss
        states = dgl.batch(states).to(self.device)
        # next_states = dgl.batch(next_states).to(self.device)

        # actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        # next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        # rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).view(-1, 1).to(self.device)
        # dones = torch.from_numpy(np.array(dones, dtype=np.float32)).view(-1, 1).to(self.device)
        # weights = torch.from_numpy(np.array(weights)).to(self.device)

        g, h, e = self.prepare_inputs(states)
        _, h, e  = self.net(g, h, e)

        action = self.max_action * torch.tanh(h)
        action += torch.empty_like(action).normal_(mean=0, std=self.sigma)

        v, w = action.split(1, -1)

        pred_yaw = states.ndata['yaw'] + w*states.ndata['time_step']

        pred_vel = torch.cat([v*torch.cos(pred_yaw), v*torch.sin(pred_yaw)], dim=-1)
        pred_pos = states.ndata['pos'] + pred_vel * states.ndata['time_step']

        goal_angle = torch.atan2(pred_pos[:, 1:2], pred_pos[:, 0:1])

        heading = pred_yaw - goal_angle

        heading = torch.where(heading>np.pi, heading-2*np.pi, heading)
        heading = torch.where(heading<np.pi, heading+2*np.pi, heading)

        node_mask = torch.logical_or(states.ndata['cid']==node_type_list.index('robot'), 
                                     states.ndata['cid']==node_type_list.index('pedestrian'))
        
        node_loss = torch.mean(((pred_pos - states.ndata['goal'])**2 + heading**2) * node_mask.unsqueeze(-1))


        # edge loss
        src_nodes, dst_nodes = states.edges()
        src_nodes_pos = pred_pos.index_select(dim=0, index=src_nodes) #[num_samples, nodes, 2]
        dst_nodes_pos = pred_pos.index_select(dim=0, index=dst_nodes) #[num_samples, nodes, 2]

        edge_dist = torch.sqrt(torch.sum((src_nodes_pos - dst_nodes_pos)**2, dim=-1, keepdims=True)) #[edges, 1]

        edge_loss = torch.mean((torch.clamp(0.5 - edge_dist, min=0))**2)

        # total loss
        loss = node_loss + 0.01*edge_loss

        # Optimize
        self.optimization_step(self.optimizer, loss)

        self.writer.add_scalar(self.policy_name + "/node_loss", node_loss, self.step)
        self.writer.add_scalar(self.policy_name + "/edge_loss", edge_loss, self.step)


        return h

    def optimization_step(self, optimizer, loss, clip_norm=None, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)

        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(net.parameters(), clip_norm) #clip gradients to help stabilise training

        optimizer.step()


