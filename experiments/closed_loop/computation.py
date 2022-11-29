
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

PORT_SPIN2CPU = int(random.randint(12000,15000))




'''
This function creates a list of weights to be used when connecting pixels to motor neurons
'''
def create_weight_list(w_fovea, w, h):
    weight_list = []
    for y in range(h):
        for x in range(w):

            for post_idx in range(4):

                weight = 0.0
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

                weight_list.append(weight)

    return weight_list


class Computer:

    def __init__(self, args, output_q, database_port):
        self.run_time = int(args.runtime)*1000 # in [ms]
        self.w_fovea = 8
        self.width = args.width
        self.height = math.ceil(self.width*3/4)
        self.pipe = args.port-3333
        self.chip_coords = (0,0)
        self.x_shift = 16
        self.y_shift = 0
        self.output_q = output_q
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
        self.database_port = database_port
        self.use_spif = not args.simulate_spif

    def __enter__(self):


        print(f"\n\n\n\n{self.nb_neurons_core}\n\n\n\n")
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
        if self.use_spif:
            dev = p.Population(None, p.external_devices.SPIFRetinaDevice(
                pipe=self.pipe, width=self.width, height=self.height, sub_width=self.subwidth,
                sub_height=self.subheight, input_x_shift=self.x_shift, input_y_shift=self.y_shift,
                chip_coords=self.chip_coords, base_key=None, board_address=None))
        else:
            dev = p.Population(self.width * self.height, p.external_devices.SpikeInjector(
                database_notify_port_num=self.database_port), label="retina",
                structure=Grid2D(self.width / self.height))

        # Create a population that captures the spikes from the input
        # capture = p.Population(self.width * self.height, p.IF_curr_exp(), structure=Grid2D(self.width / self.height), label=f"Capture for device SPIF")
        # capture_conn = p.ConvolutionConnector([[1]])
        # p.Projection(dev, capture, capture_conn, p.Convolution())

        pool_shape = (4, 3)
        post_w, post_h = p.PoolDenseConnector.get_post_pool_shape((self.width, self.height), pool_shape)
        weights = np.array(create_weight_list(self.w_fovea, post_w, post_h))
        motor_conn = p.PoolDenseConnector(weights, pool_shape)
        motor_neurons = p.Population(len(self.labels), self.celltype(**self.cell_params), label="motor_neurons")
        con_move = p.Projection(dev, motor_neurons, motor_conn, p.PoolDense())

        # Spike reception (from SpiNNaker to CPU)
        live_spikes_receiver = p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=["motor_neurons"], local_port=PORT_SPIN2CPU)
        _ = p.external_devices.activate_live_output_for(motor_neurons, database_notify_port_num=live_spikes_receiver.local_port)
        live_spikes_receiver.add_receive_callback("motor_neurons", self.receive_spikes_from_sim)

    def __exit__(self, e, b, t):
        p.end()

    def receive_spikes_from_sim(self, label, time, neuron_ids):

        for n_id in neuron_ids:
            # print(f"Spike --> MN[{n_id}]")
            self.output_q.put(n_id, False)

    def run_sim(self):
        p.run(self.run_time)
        # p.external_devices.run_forever(sync_time=0)

    def wrap_up(self):
        time.sleep(1)
        # Get recordings from populations (in case they exist)