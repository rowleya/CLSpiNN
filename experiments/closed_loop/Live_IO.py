import spynnaker8 as p
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

import time
import numpy as np
import math
import cv2
import pdb


import multiprocessing 

import sys
sys.path.insert(1, '../../miscelaneous')
from spinnorse_tools import fetch_files, plot_in_v_out, plot_spikes, plot_voltages, plot_ma_from_spikes, plot_trajectories
from stimulator import produce_data



def send_spikes_to_sim(label, connection):
    if label == "one":
        rate = 0
    if label == "two":        
        rate = 0
    if label == "three":       
        rate = 10        
    if label == "four":       
        rate = 10
        
    connection.set_rates(label, [(i,  rate) for i in range(1)])
        
def receive_spikes_from_sim(label, time, neuron_ids):
    print("1 for " + label)
    
def brain(input_q, output_q, stop_i_q, stop_o_q):
    
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

    w = 4

    nb_steps = 3000


    #SpiNNaker Setup
    delay = dt
    node_id = p.setup(timestep=dt, min_delay=delay, max_delay=delay)     
    p.set_number_of_neurons_per_core(p.IF_curr_exp, 100) #  100 neurons per core


    celltype = p.IF_curr_exp
    
    con_move = []

    port = 15600


    p_labels = ["one", "two", "three", "four"]
    m_labels = ["right", "left", "up", "down"]

    cell_conn = p.OneToOneConnector()

    px_neurons = []
    for i in range(len(m_labels)):
        px_neurons.append(p.Population(1, p.SpikeSourcePoisson(rate=0), label=p_labels[i]))
    #     px_neurons[i].record(["v","spikes"])

    # Spike emission (from CPU to SpiNNaker)        
    external_input_control = p.external_devices.SpynnakerPoissonControlConnection(poisson_labels=p_labels, local_port=port+1)
    for i in range(4):
        p.external_devices.add_poisson_live_rate_control(px_neurons[i], database_notify_port_num=port+1)
        external_input_control.add_start_resume_callback(p_labels[i], send_spikes_to_sim)
        
    move_neurons = []
    for i in range(len(m_labels)):
        move_neurons.append(p.Population(1, celltype(**cell_params), label=m_labels[i]))
    #     move_neurons[i].record(["v","spikes"])

    # Spike reception (from SpiNNaker to CPU)
    live_spikes_receiver = p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=m_labels, local_port=port-1)
    for i in range(4):
        p.external_devices.activate_live_output_for(move_neurons[i], database_notify_port_num=live_spikes_receiver.local_port)
        live_spikes_receiver.add_receive_callback(m_labels[i], receive_spikes_from_sim)

            
    #################
    #       #       #
    #   1   #   2   #
    #       #       #
    #################
    #       #       #
    #   3   #   4   #
    #       #       #
    #################


    con_move.append({'e_one_right': p.Projection(px_neurons[0], move_neurons[0], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'e_one_down': p.Projection(px_neurons[0], move_neurons[3], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'e_two_left': p.Projection(px_neurons[1], move_neurons[1], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'e_two_down': p.Projection(px_neurons[1], move_neurons[3], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'e_three_right': p.Projection(px_neurons[2], move_neurons[0], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'e_three_up': p.Projection(px_neurons[2], move_neurons[2], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'e_four_left': p.Projection(px_neurons[3], move_neurons[1], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'e_four_up': p.Projection(px_neurons[3], move_neurons[2], cell_conn,
                                                receptor_type='excitatory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})




    con_move.append({'i_one_left': p.Projection(px_neurons[0], move_neurons[1], cell_conn,
                                                receptor_type='inhibitory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'i_one_up': p.Projection(px_neurons[0], move_neurons[2], cell_conn,
                                                receptor_type='inhibitory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'i_two_right': p.Projection(px_neurons[1], move_neurons[0], cell_conn,
                                                receptor_type='inhibitory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'i_two_up': p.Projection(px_neurons[1], move_neurons[2], cell_conn,
                                                receptor_type='inhibitory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'i_three_left': p.Projection(px_neurons[2], move_neurons[1], cell_conn,
                                                receptor_type='inhibitory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'i_three_down': p.Projection(px_neurons[2], move_neurons[3], cell_conn,
                                                receptor_type='inhibitory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'i_four_right': p.Projection(px_neurons[3], move_neurons[0], cell_conn,
                                                receptor_type='inhibitory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})

    con_move.append({'i_four_down': p.Projection(px_neurons[3], move_neurons[3], cell_conn,
                                                receptor_type='inhibitory',
                                                synapse_type=p.StaticSynapse(weight=w, delay=delay))})        


    # Run simulation 
    # p.run(nb_steps)
    # p.end()

    stop_i_q.put([True])
    stop_o_q.put([True])

def set_inputs(input_q, stop_i_q):

    stop = False
    while True:
        
        if stop:
            print("No more inputs to be sent")
            break

        while not stop_i_q.empty():
            stop = stop_i_q.get(False)
    
        input_q.put([0, 9])
        input_q.put([1, 0])
        input_q.put([2, 0])
        input_q.put([3, 0])

        time.sleep(0.010)



def get_outputs(output_q, stop_o_q):

    stop = False
    while True:

        while not stop_o_q.empty():
            stop = stop_o_q.get(False)

        if stop:
            print("No more outputs to be received")
            break

        while not output_q.empty():
            spike = output_q.get(False)

        time.sleep(0.005)



if __name__ == '__main__':
    

    input_q = multiprocessing.Queue()
    output_q = multiprocessing.Queue()
    stop_i_q = multiprocessing.Queue()
    stop_o_q = multiprocessing.Queue()


    p_i_data = multiprocessing.Process(target=set_inputs, args=(input_q,stop_i_q,))
    p_o_data = multiprocessing.Process(target=get_outputs, args=(output_q,stop_o_q,))
    p_brainy = multiprocessing.Process(target=brain, args=(input_q, output_q, stop_i_q,stop_o_q,))
    
    p_i_data.start()
    p_o_data.start()
    p_brainy.start()


    p_i_data.join()
    p_o_data.join()
    p_brainy.join()