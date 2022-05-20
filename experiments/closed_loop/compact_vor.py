
import multiprocessing 
import socket
import pdb
import math
import sys
import os
import datetime
import time
import numpy as np
import random
from struct import pack

# SpiNNaker imports
import pyNN.spiNNaker as p
from pyNN.space import Grid2D

# Input imports
import pygame
sys.path.insert(1, '../../miscelaneous')
from stimulator import update_pixel_frame, BouncingBall

from visualization import *
from stimulation import *

global end_of_sim


#################################################################################################################################
#                                                           SPIF SETUP                                                          #
#################################################################################################################################


script_name = os.path.basename(__file__)

if len(sys.argv) != 3:
    print(f"python3 {script_name} <duration> <width>")
    quit()
else:
    try:
        RUN_TIME = 1000*int(sys.argv[1])
        USER_WIDTH = int(sys.argv[2])
        USER_HEIGTH = int(int(sys.argv[2])*3/4)
    except:
        print("Something went wrong with the arguments")
        quit()
print("\n About to start things ... \n")

# Device parameters are "pipe", "chip_coords", "ip_address", "port"
DEVICE_PARAMETERS = (0, (0, 0), "172.16.223.98", 3333)
PORT_SPIN2CPU = int(random.randint(12000,15000))

send_fake_spikes = True


if send_fake_spikes:
    WIDTH = USER_WIDTH
    HEIGHT = USER_HEIGTH
else:
    WIDTH = 346
    HEIGHT = round(WIDTH*3/4)
# Creates 512 neurons per core
SUB_HEIGHT = max(2,2**math.ceil(math.log(math.ceil(math.log(max(2,int(HEIGHT*WIDTH/256)),2)),2)))
SUB_WIDTH = 2*SUB_HEIGHT

print(f"Creating {SUB_WIDTH}*{SUB_HEIGHT}={SUB_WIDTH*SUB_HEIGHT} neurons per core")

# Weight of connections between "layers"
WEIGHT = 10



# Used if send_fake_spikes is True
SLEEP_TIME = 0.005
N_PACKETS = 1000



print(f"SPIF : {DEVICE_PARAMETERS[2]}:{DEVICE_PARAMETERS[3]}")
print(f"Pixel Matrix : {WIDTH}x{HEIGHT} (Real={not send_fake_spikes})")
print(f"Run Time : {RUN_TIME}")
# time.sleep(10)
# pdb.set_trace()

#################################################################################################################################
#                                                                                                                               #
#################################################################################################################################

'''
This is what's done whenever the CPU receives a spike sent by SpiNNaker
'''
def receive_spikes_from_sim(label, time, neuron_ids):

    global output_q
    
    for n_id in neuron_ids:
        # print(f"Spike --> MN[{n_id}]")
        output_q.put(n_id, False)

''' 
This function creates a list of weights to be used when connecting pixels to motor neurons
'''
def create_conn_list(w, h, n=0):
    w_fovea = 2
    conn_list = []
    

    delay = 1 # 1 [ms]
    nb_col = math.ceil(w/n)
    nb_row = math.ceil(h/n)

    pre_idx = -1
    for h_block in range(nb_row):
        for v_block in range(nb_col):
            for row in range(n):
                for col in range(n):
                    x = v_block*n+col
                    y = h_block*n+row
                    if x<w and y<h:
                        print(f"{pre_idx} -> ({x},{y})")
                        pre_idx += 1

                        for post_idx in range(4):

                            weight = 0.000001
                            x_weight = 2*w_fovea*(abs((x+0.5)-w/2)/(w-1))
                            y_weight = 2*w_fovea*(abs((y+0.5)-h/2)/(h-1))

                            # Move right (when stimulus on the left 'hemisphere')    
                            if post_idx == 0:
                                if (x+0.5) < w/2:
                                    weight = x_weight
                                        
                            # Move Left (when stimulus on the right 'hemisphere')
                            if post_idx == 1:
                                if (x+0.5) > w/2:
                                    weight = x_weight
                                                
                            # Move up (when stimulus on the bottom 'hemisphere')    
                            if post_idx == 2: 
                                if (y+0.5) > h/2: # higher pixel --> bottom of image
                                    weight = y_weight
                            
                            # Move down (when stimulus on the top 'hemisphere') 
                            if post_idx == 3:
                                if (y+0.5) < h/2: # lower pixel --> top of image
                                    weight = y_weight
                            
                            conn_list.append((pre_idx, post_idx, weight, delay))
        
    return conn_list


        


#################################################################################################################################
#                                                        SPINNAKER SETUP                                                        #
#################################################################################################################################

def run_spinnaker_sim(run_time, nb_out_n):

    global output_q

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

    celltype = p.IF_curr_exp

    # Set up PyNN


    p.setup(timestep=1.0, n_boards_required=1)     

    # Set the number of neurons per core 
    p.set_number_of_neurons_per_core(p.IF_curr_exp, SUB_HEIGHT*SUB_WIDTH)

    capture_conn = p.ConvolutionConnector([[WEIGHT]])

    # These are our external retina devices connected to SPIF devices
    pipe = DEVICE_PARAMETERS[0]
    chip_coords = DEVICE_PARAMETERS[1]

    dev = p.Population(None, p.external_devices.SPIFRetinaDevice(
        pipe=pipe, width=WIDTH, height=HEIGHT, sub_width=SUB_WIDTH,
        sub_height=SUB_HEIGHT, input_x_shift=X_SHIFT, input_y_shift=Y_SHIFT, 
        chip_coords=chip_coords, base_key=None, board_address=None))

    # Create a population that captures the spikes from the input
    capture = p.Population(WIDTH * HEIGHT, p.IF_curr_exp(), structure=Grid2D(WIDTH / HEIGHT), label=f"Capture for device SPIF")

    p.Projection(dev, capture, capture_conn, p.Convolution())

    motor_neurons = p.Population(4, celltype(**cell_params), label="motor_neurons")
    conn_list = create_conn_list(WIDTH, HEIGHT, SUB_HEIGHT*SUB_WIDTH)
    cell_conn = p.FromListConnector(conn_list, safe=True)      
    con_move = p.Projection(capture, motor_neurons, cell_conn, receptor_type='excitatory')
        
    # Spike reception (from SpiNNaker to CPU)
    live_spikes_receiver = p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=["motor_neurons"], local_port=PORT_SPIN2CPU)
    _ = p.external_devices.activate_live_output_for(motor_neurons, database_notify_port_num=live_spikes_receiver.local_port)
    live_spikes_receiver.add_receive_callback("motor_neurons", receive_spikes_from_sim)


    # Run the simulation for long enough for packets to be sent
    p.run(run_time)
    # p.external_devices.run_forever(sync_time=0)

    # Tell the software we are done with the board
    p.end()







    

if __name__ == '__main__':

    
    labels = ["go_right", "go_left", "go_up", "go_down"]
    
    manager = multiprocessing.Manager()
    end_of_sim = manager.Value('i', 0)

    output_q = multiprocessing.Queue() # events


    dev_prms = ("172.16.223.98", 3333, HEIGHT, WIDTH)

    with Stimulator(dev_prms, end_of_sim) as view:
        with Oscilloscope(labels, output_q, end_of_sim) as osci:
        


            run_spinnaker_sim(RUN_TIME, len(labels))
            
            # Let other processes know that spinnaker simulation has come to an ned
            end_of_sim.value = 1
            time.sleep(5)
            