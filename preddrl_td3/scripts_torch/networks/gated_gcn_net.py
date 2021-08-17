#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.gated_gcn_layer_old import GatedGCNLayer, GatedGCNLayerEdgeFeatOnly, GatedGCNLayerIsotropic, CustomGatedGCNLayer
from layers.mlp_layer import MLPReadout, MLP

class GatedGCNNet(nn.Module):
    
    def __init__(self, net_params, in_feat_dropout=0, dropout=0, batch_norm=False, 
                 residual=False,  activation='ReLU', layer='gated_gcn'):
        
        super(GatedGCNNet, self).__init__()

        self.in_dim_node = net_params['in_dim_node']
        self.in_dim_edge = net_params['in_dim_edge']
        self.hidden_dim = net_params['hidden_dim']
        self.n_layers = net_params['num_layers']
        self.out_dim_node = net_params['out_dim_node']
        self.out_dim_edge = net_params['out_dim_edge']
        self.embed =  net_params['embed']
        self.mlp_readout_node =  net_params['mlp_readout_node']
        self.mlp_readout_edge = net_params['mlp_readout_edge']

        self.in_feat_dropout = in_feat_dropout
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.activation = activation
        self.layer = layer
        
        if self.layer =='gcn':
            Layer = GatedGCNLayerIsotropic

        elif self.layer=='gated_gcn':
            Layer = GatedGCNLayer

        elif self.layer=='edge_gcn':
            Layer = GatedGCNLayerEdgeFeatOnly
            
        elif self.layer=='custom_gcn':
            Layer = CustomGatedGCNLayer
            
        if self.embed:
            self.embedding_h = nn.Linear(self.in_dim_node, self.hidden_dim)
            self.embedding_e = nn.Linear(self.in_dim_edge, self.hidden_dim)
        
        self.layers = nn.ModuleList([Layer(input_dim=self.hidden_dim,
                                           output_dim=self.hidden_dim,
                                           dropout=self.dropout,
                                           batch_norm=self.batch_norm,
                                           activation=self.activation,
                                           residual=self.residual)
                                    for _ in range(self.n_layers)])
        if self.mlp_readout_node:
            self.MLP_nodes = MLPReadout(self.hidden_dim, self.out_dim_node, self.activation)
            
        if self.mlp_readout_edge:
            self.MLP_edges = MLPReadout(self.hidden_dim*2, self.out_dim_edge, self.activation)
            
    def forward(self, g, h, e):
        
        # input embedding
        if self.embed:
            h = self.embedding_h(h)
            e = self.embedding_e(e)
        
        # input features dropout
        h = F.dropout(h, self.in_feat_dropout, training=self.training)
        e = F.dropout(e, self.in_feat_dropout, training=self.training) # added by niraj
        
        # res gated convnets
        for conv in self.layers:
            h, e = conv(g, h, e)
        
        # update graph
        g.ndata['h'] = h
        g.edata['e'] = e
        
        #edge output
        if self.mlp_readout_edge:
            def _edge_feat(edges):
                e = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
                e = self.MLP_edges(e)
                return {'e': e}
            g.apply_edges(_edge_feat)
            e = g.edata['e']
            # e = self.MLP_edges(e)
            # g.edata['e'] = e
        # node output
        if self.mlp_readout_node:
            h = self.MLP_nodes(h)
            g.ndata['h'] = h
        
        return g, h, e

    def __repr__(self):
        rep = '{0}(in_dim_node={1}, in_dim_edge={2}, out_dim_node={3}, out_dim_edge={4}, readout_node={5}, readout_edge={6}, \
        dropout={7}, in_feat_dropout={8}, batch_norm={9}, residual={10}, activation={11}, layer={12}'.format(self.__class__.__name__,
                                                                                                             self.in_dim_node,
                                                                                                            self.in_dim_edge, 
                                                                                                            self.out_dim_node,
                                                                                                            self.out_dim_edge,
                                                                                                            self.mlp_readout_node,
                                                                                                            self.mlp_readout_edge,
                                                                                                            self.in_feat_dropout,
                                                                                                            self.dropout,
                                                                                                            self.batch_norm,
                                                                                                            self.residual,
                                                                                                            self.activation,
                                                                                                            self.layer)
        return rep

