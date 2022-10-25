
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
import argparse
from struct import pack

# SpiNNaker imports
import pyNN.spiNNaker as p
from pyNN.space import Grid2D

PORT_SPIN2CPU = int(random.randint(12000,15000))


''' 
This function creates a list of weights to be used when connecting pixels to motor neurons
'''
def create_conn_list(w_fovea, w, h, n=0):
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
                        # print(f"{pre_idx} -> ({x},{y})")
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

class Computer:

    def __init__(self, args):
        self.run_time = int(args.runtime)*1000 # in [ms]
        self.w_fovea = 8
        self.width = args.width
        self.height = self.width
        self.pipe = args.port-3333
        self.chip_coords = (0,0)
        self.x_shift = 16
        self.y_shift = 0
        self.subheight = max(2,2**math.ceil(math.log(math.ceil(math.log(max(2,int(self.height*self.width/256)),2)),2)))
        self.subwidth = 2*self.subheight
        self.nb_neurons_core = args.npc
        self.dimensions = args.dimensions
        self.celltype = p.IF_curr_exp
        self.cell_params = {'tau_m': 20.0,
                            'tau_syn_E': 5.0,
                            'tau_syn_I': 5.0,
                            'v_rest': -65.0,
                            'v_reset': -65.0,
                            'v_thresh': -50.0,
                            'tau_refrac': 0.0, # 0.1 originally
                            'cm': 1,
                            'i_offset': 0.0
                            }
        self.labels = ["go_right", "go_left", "go_up", "go_down"]

    def __enter__(self):


        time.sleep(2)

        # Set up PyNN
        p.setup(timestep=1.0, n_boards_required=1)     

        # Set the number of neurons per core 

        if self.dimensions == 1:
            print(f"\n\n\n\n{self.dimensions}D: {self.nb_neurons_core}\n\n\n")
            p.set_number_of_neurons_per_core(p.IF_curr_exp, self.nb_neurons_core)
            
        if self.dimensions == 2:

            print(f"\n\n\n\n{self.dimensions}D: ({self.nb_neurons_core},{self.nb_neurons_core})\n\n\n")
            p.set_number_of_neurons_per_core(p.IF_curr_exp, (self.nb_neurons_core, self.nb_neurons_core))

        # Set SPIF
        dev = p.Population(None, p.external_devices.SPIFRetinaDevice(
            pipe=self.pipe, width=self.width, height=self.height, sub_width=self.subwidth,
            sub_height=self.subheight, input_x_shift=self.x_shift, input_y_shift=self.y_shift, 
            chip_coords=self.chip_coords, base_key=None, board_address=None))

        # Create a population that captures the spikes from the input
        capture = p.Population(self.width * self.height, p.IF_curr_exp(), structure=Grid2D(self.width / self.height), label=f"Capture for device SPIF")
        capture_conn = p.ConvolutionConnector([[1]])
        p.Projection(dev, capture, capture_conn, p.Convolution())


        motor_neurons = p.Population(len(self.labels), self.celltype(**self.cell_params), label="motor_neurons")
        # cell_conn = p.FromListConnector(create_conn_list(self.w_fovea, self.width, self.height, self.nb_neurons_core), safe=True)      
        cell_conn = p.AllToAllConnector(allow_self_connections=False) 
        con_move = p.Projection(capture, motor_neurons, cell_conn, receptor_type='excitatory')
            

    def __exit__(self, e, b, t): 
        p.end()


    def run_sim(self):
        p.run(self.run_time)
        # p.external_devices.run_forever(sync_time=0)



def parse_args():

    
    parser = argparse.ArgumentParser(description='SpiNNaker-SPIF Simulation with Artificial Data')

    parser.add_argument('-r','--runtime', type=int, help="Run Time, in seconds", default=10)
    parser.add_argument('-i', '--ip', type= str, help="SPIF's IP address", default="172.16.223.98")
    parser.add_argument('-p', '--port', type=int, help="SPIF's port", default=3333)
    parser.add_argument('-w', '--width', type=int, help="Image size (in px)", default=128)
    parser.add_argument('-n', '--npc', type=int, help="# Neurons Per Core", default=24)
    parser.add_argument('-d', '--dimensions', type=int, help="Dimensions (1D, 2D)", default=1)
    
    return parser.parse_args()
   

if __name__ == '__main__':

    args = parse_args()

    spin = Computer(args)

    with spin:
            
        spin.run_sim()