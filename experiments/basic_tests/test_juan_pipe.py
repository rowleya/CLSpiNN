import spynnaker8 as p
from pyNN.space import Grid2D
import socket
import pdb
import numpy as np
from random import randint, random
from struct import pack
from time import sleep
from spinn_front_end_common.utilities.database import DatabaseConnection
from threading import Thread

# pipe, coords, ip_address, port
DEVICE_PARAMETERS = [(0, (0, 0), "172.16.223.90", 3333)]

RUN_TIME = 2*1000

# Constants
P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16

WIDTH = 8
HEIGHT = int(WIDTH*3/4)

# Creates N neurons per core
SUB_HEIGHT = max(2,int(HEIGHT/20))
SUB_WIDTH = 2*SUB_HEIGHT

# Set up PyNN
p.setup(1.0, n_boards_required=1)

# Set the number of neurons per core to a rectangle
p.set_number_of_neurons_per_core(p.IF_curr_exp, (SUB_WIDTH, SUB_HEIGHT))

# These are our external retina devices connected to SPIF devices
for i, (pipe, chip_coords, _, _) in enumerate(DEVICE_PARAMETERS):
    dev = p.Population(None, p.external_devices.SPIFRetinaDevice(
        pipe=pipe, width=WIDTH, height=HEIGHT, sub_width=SUB_WIDTH,
        sub_height=SUB_HEIGHT, input_x_shift=X_SHIFT, input_y_shift=Y_SHIFT))

# Run the simulation for long enough for packets to be sent
p.run(RUN_TIME)

# Tell the software we are done with the board
p.end()

