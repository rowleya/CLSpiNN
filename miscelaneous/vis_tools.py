import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import time
import os



def plot_spikes(spikes, title, xlim):

    # Set figure size 
    nsll = 0.4  # neuron-spike line-length
    extra = 0
    
    n = len(spikes)
    ri = (n+2)*(nsll)+extra
    
    height = (ri)+2
    width = 15    
    
    fig, axs = plt.subplots(1, figsize=(width,height), gridspec_kw={'height_ratios': [ri]})
    fig.tight_layout(pad=5.0)
    fig.suptitle(title)

    for i in range(n):
#         indexes = np.where(spikes[i]>0)
        axs.eventplot(spikes[i], linewidths=2, colors='k', lineoffsets=i+1, linelengths=nsll) 
    axs.set_xlabel("Time [ms]")
    axs.set_ylabel("Spikes in IL")
    axs.set_xlim((0,xlim))
    axs.set_ylim((0,n+0.5))
#     axs.set_xticks(np.arange(0, xlim+1, 10))
    axs.set_yticks(np.arange(0, n+2, 1))
    axs.grid(visible=True, which='both', axis='y')
    plt.savefig("last_sim/" + title+".png")


def plot_voltages(voltages, title, xlim):
    
    # Set figure size 
    nsll = 0.4  # neuron-spike line-length
    extra = 0
    
    rv = 4    
    
    height = (rv)+2
    width = 15    
    
    fig, axs = plt.subplots(1, figsize=(width,height), gridspec_kw={'height_ratios': [rv]})
    fig.tight_layout(pad=5.0)
    fig.suptitle(title)  

    n = len(voltages)
    for i in range(n):
        c = 'C{}'.format(i)
        axs.plot(voltages[i], linewidth=2)
    axs.set_title("Output Layer")
    axs.set_xlabel("Time [ms]")
    axs.set_ylabel("Voltage [ms]")
    axs.set_xlim((0,xlim))
    
    
def plot_ma_from_spikes(win_size, nb_steps, m_indexes, m_labels, colors, title):


    fig, axs = plt.subplots(2, figsize=(15,6))
    fig.tight_layout(pad=5.0)
    fig.suptitle("Motor Output")  


    # print("{:.1f} %".format(np.round(t/(duration*1000),3)*100))
    max_y = 0
    for j in range(4):
        neuron_nb = j
        count = np.zeros(nb_steps-win_size)
        for i in range(win_size, nb_steps, 1):
            
            a = np.squeeze(np.array(np.where((m_indexes[neuron_nb]>=(i-win_size)) & (m_indexes[neuron_nb]<=(i)))), axis=0)
            count[i-win_size] = len(a)
            if count[i-win_size] > max_y:
                max_y = count[i-win_size]

        subplot_idx = int(j/2)
        axs[subplot_idx].plot(count, label=m_labels[j], color=colors[j])
        axs[subplot_idx].legend()

        
    axs[0].set_ylim([0, int(max_y*1.2)])
    axs[1].set_ylim([0, int(max_y*1.2)])
    plt.legend()
    plt.savefig("last_sim/" + title + ".png")


def plot_trajectories(coor, title):

    fig, axs = plt.subplots(1, figsize=(15,6))
    fig.tight_layout(pad=5.0)
    fig.suptitle("Trajectories")  
    axs.plot(coor[0,:], label="x", color="b")
    axs.plot(coor[1,:], label="y", color="y")
    axs.plot(np.round(coor[0,:],0), label="x", color="b")
    axs.plot(np.round(coor[1,:],0), label="y", color="y")
    axs.set_ylim([0, int(1.2*max(coor[0,:]))])
    plt.legend() 
    plt.savefig("last_sim/" + title + ".png")