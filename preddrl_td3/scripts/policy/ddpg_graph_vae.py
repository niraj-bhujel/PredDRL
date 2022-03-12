
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dgl
from dgl.heterograph import DGLHeteroGraph

from policy.ddpg import DDPG

from networks.gated_gcn_net import GatedGCNNet
from networks.layers.mlp_layer import MLP

from misc.huber_loss import huber_loss

class EncoderX(nn.Module):
    def __init__(self, net_params, args, **kwargs):
        super().__init__()

        net_params['encoder']['in_dim_node'] = args.state_dims['history_vel']
        net_params['encoder']['in_dim_edge'] = args.state_dims['history_dist']

        self.main = GatedGCNNet(net_params['encoder'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)

        self.head_mu = nn.Linear(net_params['encoder']['hidden_dim'], net_params['z_dim'])
        self.head_logvar = nn.Linear(net_params['encoder']['hidden_dim'], net_params['z_dim'])

    def forward(self, g, h, e):
        g, h, e = self.main(g, h, e)
        
        return self.head_mu(h), self.head_logvar(h)

class EncoderXY(nn.Module):
    def __init__(self, net_params, args, **kwargs):
        super().__init__()

        net_params['encoder']['in_dim_node'] = args.state_dims['future_vel'] + args.state_dims['history_vel']
        net_params['encoder']['in_dim_edge'] = args.state_dims['future_dist'] + args.state_dims['history_dist']

        self.main = GatedGCNNet(net_params['encoder'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)

        self.head_mu = nn.Linear(net_params['encoder']['hidden_dim'], net_params['z_dim'])
        self.head_logvar = nn.Linear(net_params['encoder']['hidden_dim'], net_params['z_dim'])

    def forward(self, g, h, e):
        g, h, e = self.main(g, h, e)
        
        return self.head_mu(h), self.head_logvar(h)


class Actor(nn.Module):
    def __init__(self, net_params, args, action_dim, max_action, **kwargs):
        super(Actor, self).__init__()

        net_params['actor']['in_dim_node'] = sum([args.state_dims[s] for s in args.input_states]) + net_params['z_dim']
        net_params['actor']['in_dim_edge'] = sum([args.state_dims[s] for s in args.input_edges])
        
        self.gcn = GatedGCNNet(net_params['actor'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)

        self.out = MLP(net_params['actor']['hidden_dim'], action_dim*args.pred_steps, hidden_size=net_params['mlp'])
        
        self.max_action = max_action
        self.input_states = args.input_states
        self.input_edges = args.input_edges

    def forward(self, state):

        g = state
        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        h = torch.cat([h, g.ndata['z']], dim=-1) 
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)

        g, h, e = self.gcn(g, h, e)

        h = self.out(h)
        h = g.ndata['max_action']*torch.tanh(h)

        return h

class Critic(nn.Module):
    def __init__(self, net_params, args, action_dim, **kwargs):
        super(Critic, self).__init__()

        self.input_states = args.input_states
        self.input_edges = args.input_edges

        net_params['critic']['in_dim_node'] = sum([args.state_dims[s] for s in args.input_states]) + action_dim*args.pred_steps 
        net_params['critic']['in_dim_edge'] = sum([args.state_dims[s] for s in args.input_edges])

        self.gcn = GatedGCNNet(net_params['critic'], 
                               in_feat_dropout=args.in_feat_dropout,
                               dropout=args.dropout, 
                               batch_norm=args.batch_norm,
                               residual=args.residual,
                               activation=args.activation,
                               layer=args.layer,)
        
        self.out = MLP(net_params['critic']['hidden_dim'], 1, hidden_size=net_params['mlp'])


    def forward(self, state, action):

        g = state
        h = torch.cat([g.ndata[s] for s in self.input_states], dim=-1)
        e = torch.cat([g.edata[s] for s in self.input_edges], dim=-1)

        h = torch.cat([h, action], dim=-1)
        g, h, _ = self.gcn(g, h, e)

        h = self.out(h) 
        
        return h

class CVAE(nn.Module):
    def __init__(self, net_params, args):
        super().__init__()
        self.z_dim = net_params['z_dim']
        # Define ecoders
        self.enc_x = EncoderX(net_params, args)
        self.enc_xy = EncoderXY(net_params, args)

        self.p_x = getattr(torch.distributions, 'Normal')
        self.q_xy = getattr(torch.distributions, 'Normal')

    def encoder_dist(self, mean, logvar, d):

        # mean = mean - mean.mean(dim=-1, keepdim=True)
        if d.__name__ == 'StudentT':
            logvar = torch.sqrt(logvar.size(-1) * F.softmax(logvar, dim=1))
            return d(10, mean, logvar)

        logvar = logvar.mul(0.5).exp_() + 1e-6

        return d(mean, logvar)

    def kl_divergence(self, p, q, samples=None):

        if (type(p), type(q)) in torch.distributions.kl._KL_REGISTRY:
            kld = torch.distributions.kl_divergence(p, q)
        else:
            if samples is None:
                K = 12
                samples = p.rsample(torch.Size([K])) if p.has_rsample else p.sample(torch.Size([K]))

            ent = -p.log_prob(samples)
            kld = (-ent - q.log_prob(samples)).mean(0)

        return kld.sum()

    def reparameterize(self, mean, logvar):
        var = logvar.mul(0.5).exp_()    
        eps = torch.Tensor(var.size()).normal_()
        eps = eps.to(var.device)
        z = eps.mul(var).add_(mean)
        return z

    def _kl_divergence(self, mean1, logvar1, mean2, logvar2):
        x1 = torch.sum((logvar2 - logvar1), dim=1)
        x2 = torch.sum(torch.exp(logvar1 - logvar2), dim=1)
        x3 = torch.sum((mean1 - mean2).pow(2) / (torch.exp(logvar2)), dim=1)
        kld_element = x1 - mean1.size(1) + x2 + x3
        return torch.mean(0.5 * kld_element)

    def forward(self, g, num_samples=1):
        
        xx = g.ndata['history_vel']
        ex = g.edata['history_dist']

        p_mean, p_logvar = self.enc_x(g, xx, ex)
        pz_x = self.encoder_dist(p_mean, p_logvar, self.p_x)

        KLD=0
        if self.training:
            yy = g.ndata['future_vel']
            ey = g.edata['future_dist']
            
            yy = torch.cat([xx, yy], dim=-1)
            ey = torch.cat([ex, ey], dim=-1)

            q_mean, q_logvar = self.enc_xy(g, yy, ey)

            qz_xy = self.encoder_dist(q_mean, q_logvar, self.q_xy)
            z = qz_xy.rsample((num_samples, ))

            KLD = self.kl_divergence(qz_xy, pz_x, z)
            # qz_xy = self.reparameterize(q_mean, q_logvar, self.q_xy)
            # KLD = self.kl_divergence(q_mean, q_logvar, p_mean, p_logvar)            

        else:
            z = pz_x.sample((num_samples, )) #(num_samples, num_node, zdim)
            # z = self.reparameterize(p_mean, p_logvar, self.p_x)

        return {'z':z, 'kld':KLD}


class GraphVAE(DDPG):
    def __init__(
            self,
            state_shape,
            action_dim,
            name="GraphVAE",
            max_action=1.0,
            lr_actor=1e-4,
            lr_critic=1e-4,
            net_params=None,
            args=None,
            **kwargs):

        super().__init__(state_shape, action_dim, name=name, **kwargs)

        self.input_states = args.input_states
        self.pred_states = args.pred_states
        self.input_edges = args.input_edges

        # Define and initialize Actor network
        self.actor = Actor(net_params, args, action_dim, max_action, **kwargs)
        self.actor_target = Actor(net_params, args, action_dim, max_action, **kwargs)
        self.soft_update_of_target_network(self.actor, self.actor_target)

        # Define and initialize Critic network
        self.critic = Critic(net_params, args, action_dim, **kwargs)
        self.critic_target = Critic(net_params, args, action_dim, **kwargs)
        self.soft_update_of_target_network(self.critic, self.critic_target)

        # define cvae
        self.cvae = CVAE(net_params, args)
        self.cvae_target = CVAE(net_params, args)
        self.soft_update_of_target_network(self.cvae, self.cvae_target)

        # define optimizers
        self.cvae_optimizer = optim.Adam(params=self.cvae.parameters(), lr=3e-4)
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=lr_critic, eps=1e-4)

    def _get_action_body(self, state, sigma, max_action):
        self.eval()
        with torch.no_grad():
            state.ndata['z'] = self.cvae(state)['z'][0]
            action = self.actor(state)
            # action += torch.empty_like(action).normal_(mean=0,std=sigma)
        self.train()
        # return torch.clamp(action, -max_action, max_action)
        return action.squeeze(0)


    def train_step(self, states, actions, next_states, rewards, dones, weights):
        # print('states:', states.number_of_nodes(), 'actions:', actions.shape, 'rewards:', rewards.shape, 'dones:', dones.shape, 'weights', weights.shape)

        self.iteration +=1 

        actor_loss, critic_loss, cvae_loss = self._train_body(states, actions, next_states, rewards, dones, weights)

        actor_loss = actor_loss.item() if actor_loss is not None else actor_loss
        critic_loss = critic_loss.item()
        kld = cvae_loss['kld'].item()
        latent = cvae_loss['z'].detach().cpu().numpy()
        # td_errors = td_errors.detach().cpu().numpy()


        self.writer.add_scalar(self.policy_name + "/actor_loss", actor_loss, self.iteration)
        self.writer.add_scalar(self.policy_name + "/critic_loss", critic_loss, self.iteration)
        self.writer.add_scalar(self.policy_name + "/KLD", kld, self.iteration)
        self.writer.add_histogram(self.policy_name + "/train_latent_z", latent, self.iteration)
        self.writer.add_scalar(self.policy_name + "/batch_rewards", rewards.mean().item(), self.iteration)

        print('STEP:{}, batch_rewards:{:.2f}, actor_loss:{:.5f}, critic_loss:{:.5f}, KLD:{:.5f}'.format(self.iteration, 
                                                                                            rewards.mean().item(),
                                                                                            actor_loss,
                                                                                            critic_loss,
                                                                                            kld))

        return actor_loss, critic_loss, cvae_loss['kld']

    def _train_body(self, states, actions, next_states, rewards, dones, weights):

        # Sample latent and compute vae loss
        cvae_loss = self.cvae(states)

        # optimize cvae
        self.optimization_step(self.cvae_optimizer, cvae_loss['kld'])

        # Compute critic loss
        td_errors = self._compute_td_error_body(states, actions, next_states, rewards, dones)
        critic_loss = torch.mean(huber_loss(td_errors, delta=self.max_grad) * weights)

        # Optimize the critic
        self.optimization_step(self.critic_optimizer, critic_loss)

        # Update latents on states
        with torch.no_grad():
            states.ndata['z'] = self.cvae_target(states)['z'].view(-1, self.cvae.z_dim)
            # states.ndata['z'] = cvae_loss['z'].view(-1, self.cvae.z_dim)
        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        # Optimize the actor 
        self.optimization_step(self.actor_optimizer, actor_loss)

        # Update target networks
        self.soft_update_of_target_network(self.actor, self.actor_target)
        self.soft_update_of_target_network(self.critic, self.critic_target)
        self.soft_update_of_target_network(self.cvae, self.cvae_target)

        return actor_loss, critic_loss, cvae_loss

    def _compute_td_error_body(self, states, actions, next_states, rewards, dones):

        not_dones = 1. - dones

        with torch.no_grad():
            next_states.ndata['z'] = self.cvae_target(next_states)['z'].view(-1, self.cvae.z_dim)
            target_Q = self.critic_target(next_states, self.actor_target(next_states))

            target_Q = rewards + (not_dones * self.discount * target_Q)

        current_Q = self.critic(states, actions)

        td_errors = target_Q - current_Q

        return td_errors