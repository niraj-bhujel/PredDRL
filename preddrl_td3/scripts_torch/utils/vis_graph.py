#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 18:21:14 2020

@author: dl-asoro
"""
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
import shutil
import torch
def network_draw(g, show_node_label=True, node_label='nid', show_edge_labels=False, edge_label='id', 
                 show_legend=False, pos_attr='pos', edge_attr='dist', node_size=300, font_size=6, 
                 rad=0.04,  save_dir=None, fprefix=None, fsuffix=None, frame=None, counter=0,
                 pad=(0, 0, 0, 0), extent=None, **kwargs):
    '''
    Parameters
    ----------
    g : TYPE
        DESCRIPTION.
    show_node_label : TYPE, optional
        DESCRIPTION. The default is False.
    show_edge_labels : TYPE, optional
        DESCRIPTION. The default is False.
    edge_label : TYPE, can also be 'id' after converting dgl graph 'g' to networkx graph 'G'.
                    The id ordering could be different from original 'g'
        DESCRIPTION. The default is 'dist'.
    pos_attr : TYPE, optional
        DESCRIPTION. The default is 'pos'.
    node_size : TYPE, optional
        DESCRIPTION. The default is 300.
    show_legend : BOOL, if True, show the node number and corresponding tid 
        DESCRIPTION. The default is False.
    save_dir : TYPE, optional
        DESCRIPTION. The default is None.
    fprefix : TYPE, optional
        DESCRIPTION. The default is None.
    counter : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    '''
    # overwride from kwargs
    show_node_label = kwargs.get('show_node_label', show_node_label)
    show_edge_labels= kwargs.get('show_edge_labels', show_edge_labels)
    node_size = kwargs.get('node_size', node_size)
    font_size = kwargs.get('font_size', font_size)
    rad = kwargs.get('rad', rad)
    show_legend = kwargs.get('show_legend', show_legend)
    frame = kwargs.get('frame', frame)
    counter = kwargs.get('counter', counter)
    pad = kwargs.get('pad', pad)
    extent = kwargs.get('extent', extent)
    save_dir = kwargs.get('save_dir', save_dir)
    fig_name = kwargs.get('fig_name', '')
    
    if edge_label=='id' or edge_label=='eid': # edge label 'id' is only available after converting to netwokx
        G = g.cpu().to_networkx(node_attrs=[node_label, pos_attr, 'tid'], edge_attrs=[edge_attr, 'spatial_mask'])
    else:
        G = g.cpu().to_networkx(node_attrs=[node_label, pos_attr, 'tid'], edge_attrs=[edge_label, edge_attr, 'spatial_mask'])    


    unique_tid = g.ndata['tid'].unique().cpu().numpy()
    ped_colors = np.random.random((3, len(unique_tid)))
    
    # node
    pos = {}
    node_colors = []
    node_labels = {}
    
    for u, u_data in G.nodes(data=True):
        
        pos[u] = u_data[pos_attr].numpy()

        traj_id = u_data['tid'].numpy()
        
        tid_idx = np.where(unique_tid==traj_id)[0][0]
        node_colors.append(ped_colors[:, tid_idx])        
        
        node_labels[u] = u_data[node_label].numpy().round(2)
        # node_labels[u]=traj_id
        # node_labels[u]=u_data['tid'].numpy()-unique_tid.min()+1

    temporal_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["spatial_mask"]==0]
    spatial_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d["spatial_mask"]==1]
    spatial_edges_w = [min(1, 1/d[edge_attr].numpy()[0]) for u, v, d in G.edges(data=True) if d["spatial_mask"]==1]
    
    plt.clf()
    # plt.close('all')
    fig = plt.figure(fig_name, figsize=(12, 8), dpi=100)
    ax = fig.subplots()
    ax.axis('off')

    if extent is not None:
        x_min, x_max = extent['x_min']-pad[0], extent['x_max']+pad[1]
        y_min, y_max = extent['y_min']-pad[2], extent['y_max']+pad[3]
            
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
            
    if frame is not None:
        if extent is not None:
            ax.imshow(frame, aspect='auto', extent=[x_min, x_max, y_min, y_max]) #extents = (left, right, bottom, top)
        else:
            print('Showing frame without extent, may not render properly. Provide x_min, x_max, y_min, y_max to define the extent of the frame')
            ax.imshow(frame, aspect='auto')
        
    # draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.5, ax=ax)

    # draw node labels
    if show_node_label:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size)
    
    # draw edges
    # Be sure to include node_size as a keyword argument; arrows are drawn considering the size of nodes.
    nx.draw_networkx_edges(G, pos, edgelist=temporal_edges, node_size=node_size, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=spatial_edges,  alpha=0.66, width=spatial_edges_w,
                                            # connectionstyle='arc3, rad=%s'%float(rad),
                                            node_size=node_size, ax=ax) 

    # draw edge labels
    if show_edge_labels:
        edge_labels = {}
        for u, v, e in G.edges(data=True):
            elabel = e[edge_label]
            if isinstance(elabel, torch.Tensor):
                if elabel.requires_grad:
                    elabel = elabel.detach().numpy().round(2)
                else:
                    elabel = elabel.numpy().round(2)

            # edge_labels[(u, v)] = np.around(elabel, decimals=3)

            # use this for curved edges
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x12, y12 = (x1 + x2) / 2., (y1 + y2) / 2. # center of the edge
            dx, dy = x2 - x1, y2 - y1 # x and y offset
            cx, cy = x12 + rad * dy, y12 - rad * dx
            ax.text(cx, cy, s='{}'.format(elabel), fontsize=8, zorder=1, clip_on=True)
            
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.2, font_size=8)
        
    if show_legend:
        # custom legends
        legend_elements = []
        for tid, color in zip(unique_tid, ped_colors.T):
            legend_elements.append(Line2D([0], [0], marker='o', color=color, markerfacecolor=color, markersize=4,
                                            # label='{}:{}'.format(tid, g.nodes()[g.ndata['tid']==tid].tolist()),
                                           # label='{}'.format(tid-unique_tid.min()+1) # indexed to 0
                                           label='tid-{}, cid-{}'.format(tid, g.ndata['cid'][g.ndata['tid']==tid][0].item())
                                           ))
        ax.legend(handles=legend_elements, fontsize='small', handlelength=1)
    
    plt.tight_layout()
    if save_dir is not None:
        if not os.path.exists(save_dir):
            # shutil.rmtree(save_dir)
            os.makedirs(save_dir)

        if fprefix is not None:
            file_name = '{}_step_{}'.format(fprefix, counter)

        elif fsuffix is not None:
            file_name = 'step_{}_{}'.format(counter, fsuffix)

        else:
            file_name = 'step_{}'.format(counter)
        
        plt.title(file_name)

        fig.savefig(save_dir + '/' + file_name + '.png', bbox_inches='tight',dpi=100)
    
if __name__=='__main__':
    import torch
    show_node_label=True
    show_edge_labels=False
    node_label='nid'
    edge_label='id'
    pos_attr='pos'
    edge_attr='dist'
    node_size=300
    rad=0.04
    show_legend=False
    save_dir=None
    fprefix=None
    frame=None
    counter=0
    pad=(0, 0, 0, 0)
    extent=None
    
    # g = raw_data['trgt_graphs']
    # network_draw(g, show_node_label=True, show_edge_labels=False, edge_label='id', show_legend=False)
