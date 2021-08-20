import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl

from policy.policy_base import OffPolicyAgent
from misc.huber_loss import huber_loss

from networks.gated_gcn_net import GatedGCNNet
from utils.graph_utils import node_type_list


class MLP(torch.nn.Module):
    def __init__(self, state_shape, action_dim, max_action, units=[256, 256]):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(state_shape, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], action_dim)

        self.max_action = max_action

    def forward(self, inputs):
        #DNN
        features = F.relu(self.l1(inputs))
        features = F.relu(self.l2(features))
        features = self.l3(features)

        # features = self.max_action * torch.tanh(features)

        return features


class SimpleMLP(OffPolicyAgent):
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
        self.state_shape = state_shape
        self.input_states = args.input_states
        self.input_edges = args.input_edges


        self.net = MLP(state_shape[0], action_dim, max_action, actor_units).to(self.device)

        self.optimizer = optim.Adam(params=self.net.parameters(), lr=lr)

        self.step = 0


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

        
        states = torch.from_numpy(np.array(states, dtype=np.float32)).view(-1, self.state_shape[0]).to(self.device)


        a = self.net(states)

        # h = self.max_action * torch.tanh(h)
        v, w = a.split(1, -1)

        pred_yaw = w + states.ndata['yaw']
        pred_vel = torch.cat([v*torch.cos(w), v*torch.sin(w)], dim=-1)

        pred_pos = states.ndata['pos'] + pred_vel * 0.25

        goal_angle = torch.atan2(pred_pos[:, 1:2], pred_pos[:, 0:1])

        heading = pred_yaw - goal_angle

        heading = torch.where(heading>np.pi, heading-2*np.pi, heading)
        heading = torch.where(heading<np.pi, heading+2*np.pi, heading)

        node_mask = torch.logical_or(states.ndata['cid']==node_type_list.index('robot'), 
                                     states.ndata['cid']==node_type_list.index('pedestrian'))
        
        node_loss = torch.mean(((pred_pos - states.ndata['goal'])**2 + heading**2) * node_mask.unsqueeze(-1))


        # # edge loss
        # src_nodes, dst_nodes = states.edges()
        # src_nodes_pos = pred_pos.index_select(dim=0, index=src_nodes) #[num_samples, nodes, 2]
        # dst_nodes_pos = pred_pos.index_select(dim=0, index=dst_nodes) #[num_samples, nodes, 2]

        # edge_dist = torch.sqrt(torch.sum((src_nodes_pos - dst_nodes_pos)**2, dim=-1, keepdims=True)) #[edges, 1]

        # edge_loss = torch.mean((torch.clamp(0.5 - edge_dist, min=0))**2)

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


