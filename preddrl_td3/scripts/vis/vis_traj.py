#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:16:21 2020

@author: dl-asoro
"""
import os
import sys
import time
import pickle
import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgb, LinearSegmentedColormap

def get_color(idx):
    idx = idx * 3
    color = [(37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255]
    color = [c/255 for c in color]
    return color

#x_min, x_max is actually y_min y_max in eth and hotel
data_stats = {'eth': {'y_max': 15.613, 'y_min': -7.69, 'x_max': 13.943, 'x_min': -10.31},
                'hotel': {'y_max': 15.613, 'y_min': -7.69, 'x_max': 13.943, 'x_min': -10.31},
                'univ': {'x_max': 15.613, 'x_min': -7.69, 'y_max': 13.943, 'y_min': -10.31},
                'zara1': {'x_max': 15.613, 'x_min': -7.69, 'y_max': 13.943, 'y_min': -10.31},
                'zara2': {'x_max': 15.613, 'x_min': -7.69, 'y_min': 13.943, 'y_max': -10.31}}


def plot_traj(obsv_traj, trgt_traj, pred_traj=None, ped_ids=None, K=1, extent=None, pad=(1, 1, 1, 1), 
              counter=0, frame=None, save_dir=None, dtext='', fprefix=None, legend=False, ax=None, 
              axis_off=False, limit_axes=False, arrow=False, ticks_off=False,
              lw=1, lm='o', ms=2, mw=1):
    '''
    Parameters
    ----------
    obsv_traj : List of N arrays each with shape [ped_obsv_len, 2]
    trgt_traj : List of N arrays each with shape [ped_trgt_len, 2]
    pred_traj : List of N arrays each with shape [K, ped_trgt_len, 2]
    ped_ids: List of N id
    K: number of prediction to plot
    counter : TYPE, optional
        DESCRIPTION. The default is 0.
    frame : TYPE, optional
        DESCRIPTION. The default is None.
    save_dir : TYPE, optional
        DESCRIPTION. The default is './plots'.
    legend : TYPE, optional
        DESCRIPTION. The default is False.
    axis_off : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    '''
        
    if extent is not None:
        x_min, x_max = extent['x_min']-pad[0], extent['x_max']+pad[1]
        y_min, y_max = extent['y_min']-pad[2], extent['y_max']+pad[3]
    else:
        seq_traj = np.concatenate(obsv_traj + trgt_traj)
        x_min, y_min = seq_traj.min(axis=0) - pad[:2]
        x_max, y_max = seq_traj.max(axis=0) + pad[2:]
        
    #create canvass
    plt.close('all')
    if ax is None:
        w, h = 8, 5
        fig = plt.figure(frameon=True, figsize=(w, h))
        ax = plt.axes()
        if axis_off:
            ax.axis('off')
        fig.add_axes(ax)
    
    if frame is not None:
        if extent is not None:
            ax.imshow(frame, aspect='auto', extent=[x_min, x_max, y_min, y_max]) #extents = (left, right, bottom, top)
        else:
            print('Showing frame without extent, may not render properly. Provide x_min, x_max, y_min, y_max to define the extent of the frame')
            ax.imshow(frame, aspect='auto')
        
    # set limit
    if limit_axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    # ax.plot(0, 0, 'o', color='black')
    # ax.plot((x_max + x_min)/2, (y_max+y_min)/2, 's', color='blue')
    
    num_peds = len(obsv_traj)
    cmap = plt.cm.get_cmap(name='tab20')

    legend_handles = []
    legend_labels = []    
    for p in range(num_peds):
        color = cmap(p)

        if ped_ids is not None:
            color = matplotlib.colors.to_rgba(get_color(ped_ids[p]))

        # obsv tracks
        xs, ys = obsv_traj[p][:, 0], obsv_traj[p][:, 1]
        
        # start markers
        start_mark = ax.scatter(xs[0:1], ys[0:1], c=[color], label='Start', marker=lm, edgecolors='k', s=lw**3, zorder=3)
        
        # plot obsv tracks Never Walk Alone: Mod­e
        obsv_line, = ax.plot(xs, ys, color=color, linestyle='solid', linewidth=lw, zorder=2, 
                             marker=lm, markersize=ms, fillstyle='full', mfc='w', mec=color, mew=mw,
                             )
    
        #target tracks
        xs, ys = trgt_traj[p][:, 0], trgt_traj[p][:, 1]
        target_line, = ax.plot(xs, ys, color=color, label='Target', linestyle='solid', linewidth=lw, zorder=3,
                               marker=lm, markersize=ms, fillstyle='full', mfc='w', mec=color, mew=mw,
                               )
        # quiver requires at least two points
        if arrow and len(trgt_traj[p])>1:
            # end marker
            ax.quiver(xs[-2], ys[-2], (xs[-1]-xs[-2])+0.001, (ys[-1]-ys[-2])+0.001, color=color, zorder=3, 
                        angles='xy', scale_units='xy', scale=1, width=0.02*(ys[-1]-y_min)/(y_max-y_min),
                        # headwidth=3, headlength=4, headaxislength=3,
                        )
        
        if pred_traj is not None:
            preds = pred_traj[p][:, :len(trgt_traj[p]), :]
            # plot top k predicted traj
            for k in range(K):
                xs, ys = preds[k][:, 0], preds[k][:, 1]
                pred_line, = ax.plot(xs, ys, color=color, label='Predictions', linestyle='--', linewidth=lw, zorder=10,
                                     marker=lm, markersize=ms, fillstyle='full', mfc=color, mec=color, mew=mw,
                                     )
                if arrow and len(trgt_traj[p])>1:
                    # end arrow
                    ax.quiver(xs[-2], ys[-2], (xs[-1]-xs[-2])+0.001, (ys[-1]-ys[-2])+0.001, color=color, zorder=10,
                              # angles='xy', scale_units='xy', scale=2, width=0.015*(ys[-1]-y_min)/(y_max-y_min),
                              width=0.01*(ys[-1]-y_min)/(y_max-y_min), headwidth=3, headlength=4, headaxislength=3,
                              )
                    
        if ped_ids is not None:
            legend_handles.append(target_line)
            legend_labels.append('{}:ID{}'.format(p, int(ped_ids[p])))
            
    if legend:
        legend_handles.extend([start_mark, target_line])
        legend_labels.extend(['Start', 'GT'])

        if pred_traj is not None:
            legend_handles.extend([pred_line])
            legend_labels.extend(['Pred'])
        
        ax.legend(legend_handles, legend_labels, handlelength=4)
        
    if dtext is not None:
        # plt.text(0.05, 0.96, '%s, frame:%s'%(dtext, counter), transform=ax.transAxes, fontsize=16, color='blue', va='top',)
        plt.text(0.05, 0.96, '%s'%dtext, transform=ax.transAxes, fontsize=16, color='blue', va='top',)
    
    if ticks_off:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                        labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    
    plt.tight_layout()
    
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if fprefix is not None:
            file_path = save_dir + '{}_frame_{}.png'.format(fprefix, counter)
        else:
            file_path = save_dir + 'frame_{}.png'.format(counter)
        fig.savefig(file_path , bbox_inches='tight',dpi=100)