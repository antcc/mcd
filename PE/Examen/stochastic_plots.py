# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:50:45 2020

@author: alberto
"""

# Load packages
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(t, X, 
                      max_trajectories=20, 
                      fig_num=1, fig_size=(8,4), font_size=10, mean_color='k'):

    M, N = np.shape(X)
    # Plot trajectories 
    M = np.min((M, max_trajectories))
    fig = plt.figure(fig_num)
    fig.clf()
    plt.plot(t, X[:M,:].T, linewidth=1)
    plt.xlabel('t', fontsize=font_size)
    plt.ylabel('X(t)', fontsize=font_size)
    plt.title('Simulation', fontsize=font_size)
  
    plt.plot(t, np.mean(X, axis=0), linewidth=3, color=mean_color)
  
    
  
    
def plot_pdf(X, pdf,
             max_bins=50,
             fig_num=1, fig_size=(4,4), font_size=10):


    # Plot histogram
    fig = plt.figure(fig_num)
    fig.clf()
    n_bins = np.min((np.int(np.round(np.sqrt(len(X)))), max_bins))
       
    plt.hist(X, bins=n_bins, density=True)
    plt.xlabel('x', fontsize=font_size)
    plt.ylabel('pdf(x)', fontsize=font_size)
    
    # Compare with exact distribution
    n_plot = 1000
    x_plot = np.linspace(np.min(X), np.max(X),n_plot)
    y_plot = pdf(x_plot)
    plt.plot(x_plot, y_plot, linewidth=2, color='r')
    