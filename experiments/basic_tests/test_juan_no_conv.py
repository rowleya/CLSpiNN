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

# Device parameters are "pipe", "chip_coords", "ip_address", "port"
# Note: IP address and port are used to send in spikes when send_fake_spikes
#       is True

DEVICE_PARAMETERS = [(0, (0, 0), "172.16.223.98", 3333)]
# DEVICE_PARAMETERS = [(0, (0, 0), "172.16.223.2", 10000),
#                      (0, (32, 16), "172.16.223.122", 10000),
#                      (0, (16, 8), "172.16.223.130", 10000)]
#DEVICE_PARAMETERS = [(0, (16, 8), "172.16.223.130", 10000)]


def receive_spikes_from_sim(label, time, neuron_ids):
    print("1 for " + label)

send_fake_spikes = True

# Constants
P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16

if send_fake_spikes:
    WIDTH = 8
    HEIGHT = int(WIDTH*3/4)
else:
    WIDTH = 346
    HEIGHT = 260
# Creates 512 neurons per core
SUB_HEIGHT = max(2,int(HEIGHT/20))
SUB_WIDTH = 2*SUB_HEIGHT

print(f"Creating {SUB_WIDTH*SUB_HEIGHT} neurons per core")

# Weight of connections between "layers"
WEIGHT = 100



# Used if send_fake_spikes is True
sleep_time = 0.1
n_packets = 10

# Run time if send_fake_spikes is False
run_time = 10000 #[ms]

if send_fake_spikes:
    run_time = (2*n_packets + 1) * sleep_time * 1000



def send_retina_input(ip_addr, port):
    """ This is used to send random input to the Ethernet listening in SPIF
    """
    global X_SHIFT, Y_SHIFT, P_SHIFT
    NO_TIMESTAMP = 0x80000000
    min_x = 0
    min_y = 0
    max_x = WIDTH - 1
    max_y = HEIGHT - 1
    polarity = 1

    sleep(0.5 + (random() / 4.0))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    count = 0
    x = int(WIDTH*3/4)
    y = int(HEIGHT*1/4)
    for _ in range(n_packets):
        n_spikes = 10
        data = b""
        for _ in range(n_spikes):
            packed = (
                NO_TIMESTAMP + (polarity << P_SHIFT) +
                (y << Y_SHIFT) + (x << X_SHIFT))
            print(f"Sending x={x}, y={y}, polarity={polarity}, "
                  f"packed={hex(packed)}")
            count+= 1

            data += pack("<I", packed)
        sock.sendto(data, (ip_addr, port))
        sleep(sleep_time)
    print(f"count is --> {count}")


def start_fake_senders():
    global DEVICE_PARAMETERS

    for _, _, ip_addr, port in DEVICE_PARAMETERS:
        t = Thread(target=send_retina_input, args=(ip_addr, port))
        t.start()




# Set up PyNN
p.setup(1.0, n_boards_required=24)

# Set the number of neurons per core to a rectangle
# (creates 512 neurons per core)
p.set_number_of_neurons_per_core(p.IF_curr_exp, (SUB_WIDTH, SUB_HEIGHT))

if send_fake_spikes:
    # This is only used with the above to send data to the Ethernet
    connection = DatabaseConnection(start_fake_senders, local_port=None)

    # This is used with the connection so that it starts sending when the
    # simulation starts
    p.external_devices.add_database_socket_address(None, connection.local_port, None)

capture_conn = p.ConvolutionConnector([[WEIGHT]])

# These are our external retina devices connected to SPIF devices
devices = list()
captures = list()
for i, (pipe, chip_coords, _, _) in enumerate(DEVICE_PARAMETERS):
    dev = p.Population(None, p.external_devices.SPIFRetinaDevice(
        pipe=pipe, width=WIDTH, height=HEIGHT, sub_width=SUB_WIDTH,
        sub_height=SUB_HEIGHT, input_x_shift=X_SHIFT, input_y_shift=Y_SHIFT))

    # Create a population that captures the spikes from the input
    capture = p.Population(
        WIDTH * HEIGHT, p.IF_curr_exp(), structure=Grid2D(WIDTH / HEIGHT),
        label=f"Capture for device {i}")

    p.Projection(dev, capture, capture_conn, p.Convolution())
  
    # Record the spikes so we know what happened
    capture.record("spikes")

    # Save for later use
    devices.append(dev)
    captures.append(capture)



# Run the simulation for long enough for packets to be sent
p.run(run_time)


# # Get out the spikes
capture_spikes = np.asarray(capture.get_data("spikes").segments[0].spiketrains)

in_spikes = capture.get_data("spikes")

in_spikes = capture.get_data("spikes")
for h in range(HEIGHT):
    for w in range(WIDTH):
        l = len(np.asarray(in_spikes.segments[0].spiketrains[h*WIDTH+w]))
        print(f"({w},{h})-->{l}")


pdb.set_trace()



# Tell the software we are done with the board
p.end()

