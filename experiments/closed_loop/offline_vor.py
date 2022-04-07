import spynnaker8 as p
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

import time
import numpy as np
import math
import pdb


import sys
sys.path.insert(1, '../../miscelaneous')
from vis_tools import plot_spikes, plot_voltages, plot_ma_from_spikes, plot_trajectories
from stimulator import produce_data

##############################################################################################################################################
#                                                                  STIMULI                                                                   #
##############################################################################################################################################

def create_stimuli(duration, l_px):

    w_px = round(l_px*3/4)
    vx =  l_px/80
    vy = -w_px/160
    r_ball = min(8, int(l_px*7/637+610/637))
    mat, coor = produce_data(l_px, w_px, r_ball, vx, vy, duration)

    return mat, coor, w_px


##############################################################################################################################################
#                                                                    SETUP                                                                   #
##############################################################################################################################################

def run_spinnaker_sim(mat, coor, l_px, w_px, duration):

    dt = 1           # (ms) simulation timestep

    cell_params = {'tau_m': 20.0,
                'tau_syn_E': 5.0,
                'tau_syn_I': 5.0,
                'v_rest': -65.0,
                'v_reset': -65.0,
                'v_thresh': -50.0,
                'tau_refrac': 0.0, # 0.1 originally
                'cm': 1,
                'i_offset': 0.0
                }

    w = 1.0
    w_fovea = 10



    #SpiNNaker Setup

    nb_steps = duration*1000
    delay = dt
    node_id = p.setup(timestep=dt, min_delay=delay, max_delay=delay)     
    p.set_number_of_neurons_per_core(p.IF_curr_exp, 100) #  100 neurons per core


    celltype = p.IF_curr_exp

    cells_l0 = []  
    cells_l1 = []   
    con_l0l1 = []  
    con_move = []
    w_x = np.zeros((2,l_px))
    w_y = np.zeros((2,w_px))

    idx = -1

    m_labels = ["go_right", "go_left", "go_up", "go_down"]
    colors = ["g", "r", "g", "r"]

    move_neurons = []
    for i in range(len(m_labels)):
        move_neurons.append(p.Population(1, celltype(**cell_params), label=m_labels[i]) )
        move_neurons[i].record(["v","spikes"])


    for y in range(w_px):
            
        for x in range(l_px):
            
            idx += 1
            i_spikes = mat[y, x, :]
            i_indexes = np.where(i_spikes > 0)
            spike_trains = p.SpikeSourceArray(spike_times=(i_indexes))

            cur_label = "N_{:d}_{:d}".format(x, y)
            
            # Populations        
            cells_l0.append(p.Population(1,spike_trains))        
    #         cells_l1.append(p.Population(1, celltype(**cell_params), label=cur_label))


            # Connectivity
            cell_conn = p.AllToAllConnector()
            cur_label = "i2l_{:d}_{:d}".format(x, y)         
    #         con_l0l1.append({ 'i1l1': p.Projection(cells_l0[idx], cells_l1[idx], cell_conn,
    #                                 receptor_type='excitatory',
    #                                 synapse_type=p.StaticSynapse(weight=w, delay=delay))})
            
            # Move right (when stimulus on the left 'hemisphere')    
            if x < l_px/2:
                w_motor_x = w_fovea*(abs(x-l_px/2)/l_px )
                con_move.append({ 'l1lm': p.Projection(cells_l0[idx], move_neurons[0], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w_motor_x, delay=delay))})
        
            # Move Left (when stimulus on the right 'hemisphere')
            if x > l_px/2:
                w_motor_x = w_fovea*(abs(x-l_px/2)/l_px )
                con_move.append({ 'l1lm': p.Projection(cells_l0[idx], move_neurons[1], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w_motor_x, delay=delay))})
            
            # Move up (when stimulus on the bottom 'hemisphere')     
            if y < w_px/2: # higher pixel --> bottom of image
                w_motor_y = w_fovea*(abs(y-w_px/2)/w_px )
                con_move.append({ 'l1lm': p.Projection(cells_l0[idx], move_neurons[2], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w_motor_y, delay=delay))})
            
            # Move down (when stimulus on the top 'hemisphere') 
            if y > w_px/2: # lower pixel --> top of image
                w_motor_y = w_fovea*(abs(y-w_px/2)/w_px )
                con_move.append({ 'l1lm': p.Projection(cells_l0[idx], move_neurons[3], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w_motor_y, delay=delay))})
            
            
    #         print("N_{:d}_{:d} --> ({:.3f},{:.3f})".format(x, y, w_motor_x, w_motor_y))
            
            # Setup recording 
            cells_l0[idx].record(["spikes"])
    #         cells_l1[idx].record(["v","spikes"])

    # print("\n\n\n\n\n\n\n\n")        
            
    # Run simulation 
    p.run(nb_steps)



            

    i_indexes = []
    o_indexes = []
    v_arrays = []
    for neuron_nb in range(0,w_px*l_px,1):
        
        in_spikes = cells_l0[neuron_nb].get_data("spikes")
        i_indexes.append(np.asarray(in_spikes.segments[0].spiketrains[0]))   



    m_indexes = []
    m_v_arrays = []
    for i in range(len(m_labels)):    
        spikes = move_neurons[i].get_data("spikes")
        voltage = move_neurons[i].get_data("v")
        m_indexes.append(np.asarray(spikes.segments[0].spiketrains[0]))
        m_v_arrays.append(np.array(voltage.segments[0].filter(name="v")[0]).reshape(-1))

    p.end()


    ##############################################################################################################################################


    xlim = nb_steps       
    for i in range(4):
        plot_spikes(m_indexes[i:i+1], "spikes_" + m_labels[i], xlim)
    # for i in range(4): 
    #     plot_voltages(m_v_arrays[i:i+1], "voltage_" + m_labels[i], xlim)


    win_size = 20
    title = "motor_neurons_ma"
    plot_ma_from_spikes(win_size, nb_steps, m_indexes, m_labels, colors, title)

    # pdb.set_trace()
    plot_trajectories(coor, "ball_trajectories")


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("python3 closed_loop_no_spif.py <duration> <width>")
        quit()
    else:
        try:
            duration = int(sys.argv[1])
            l_px = int(sys.argv[2])
        except:
            print("Something went wrong with the arguments")
            quit()

    mat, coor, w_px = create_stimuli(duration, l_px)
    run_spinnaker_sim(mat, coor, l_px, w_px, duration)