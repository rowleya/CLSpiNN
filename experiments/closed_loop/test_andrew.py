import spynnaker8 as p
from pyNN.space import Grid2D
import socket
from random import randint, random
from struct import pack
from time import sleep
from spinn_front_end_common.utilities.database import DatabaseConnection
from threading import Thread

# Device parameters are "pipe", "chip_coords", "ip_address", "port"
# Note: IP address and port are used to send in spikes when send_fake_spikes
#       is True

DEVICE_PARAMETERS = [(0, (0, 0), "172.16.223.98", 10000)]
# DEVICE_PARAMETERS = [(0, (0, 0), "172.16.223.2", 10000),
#                      (0, (32, 16), "172.16.223.122", 10000),
#                      (0, (16, 8), "172.16.223.130", 10000)]
#DEVICE_PARAMETERS = [(0, (16, 8), "172.16.223.130", 10000)]
send_fake_spikes = True

# Used if send_fake_spikes is True
sleep_time = 0.1
n_packets = 100

# Run time if send_fake_spikes is False
run_time = 60000

if send_fake_spikes:
    run_time = (n_packets + 1) * sleep_time * 1000

# Constants
P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16
WIDTH = 640
HEIGHT = 480
# Creates 512 neurons per core
SUB_WIDTH = 32
SUB_HEIGHT = 16
# Weight of connections between "layers"
WEIGHT = 5


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
    for _ in range(n_packets):
        n_spikes = randint(10, 100)
        data = b""
        for _ in range(n_spikes):
            x = randint(min_x, max_x)
            y = randint(min_y, max_y)
            packed = (
                NO_TIMESTAMP + (polarity << P_SHIFT) +
                (y << Y_SHIFT) + (x << X_SHIFT))
            print(f"Sending x={x}, y={y}, polarity={polarity}, "
                  f"packed={hex(packed)}")
            data += pack("<I", packed)
        sock.sendto(data, (ip_addr, port))
        sleep(sleep_time)


def start_fake_senders():
    global DEVICE_PARAMETERS

    for _, _, ip_addr, port in DEVICE_PARAMETERS:
        t = Thread(target=send_retina_input, args=(ip_addr, port))
        t.start()


def find_next_spike_after(spike_times, time):
    for index, spike_time in enumerate(spike_times):
        if spike_time >= time:
            return index, spike_time
    return None, None


def find_square_of_spikes(x, y, time, spikes, s_label, t_label):
    found_spikes = list()
    last_target_time = None
    for x_t in range(x - 1, x + 2):
        if x_t < 0 or x_t >= WIDTH:
            continue
        for y_t in range(y - 1, y + 2):
            if y_t < 0 or y_t >= HEIGHT:
                continue
            target_neuron = (y_t * WIDTH) + x_t
            index, target_time = find_next_spike_after(
                spikes[target_neuron], time)
            if index is None:
                print(
                    "WARNING: "
                    f"Spike in source {s_label}: {x}, {y} at time {time} not"
                    f" found in target {t_label}: {x_t}, {y_t}: {spikes[target_neuron]}")
                continue
            if last_target_time is not None:
                if last_target_time != target_time:
                    print(
                        "WARNING: "
                        f"Spike in source {s_label}: {x}, {y} at time {time}"
                        " does not have matching time in all surrounding"
                        " targets")
                    continue
            found_spikes.append((x_t, y_t, target_time))
            print(f"Spike in source {s_label}: {x}, {y} at time {time} matches"
                  f" target {t_label}: {x_t}, {y_t} at time {target_time}")
    return spikes, found_spikes


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
    p.external_devices.add_database_socket_address(
        None, connection.local_port, None)

# This is our convolution connector.  This one doesn't do much!
conn = p.ConvolutionConnector([[WEIGHT, WEIGHT, WEIGHT],
                               [WEIGHT, WEIGHT, WEIGHT],
                               [WEIGHT, WEIGHT, WEIGHT]], padding=(1, 1))

capture_conn = p.ConvolutionConnector([[WEIGHT]])


# These are our external retina devices connected to SPIF devices
devices = list()
captures = list()
layer_1s = list()
layer_2s = list()
for i, (pipe, chip_coords, _, _) in enumerate(DEVICE_PARAMETERS):
    dev = p.Population(None, p.external_devices.SPIFRetinaDevice(
        pipe=pipe, width=WIDTH, height=HEIGHT, sub_width=SUB_WIDTH,
        sub_height=SUB_HEIGHT, input_x_shift=X_SHIFT, input_y_shift=Y_SHIFT,
        chip_coords=chip_coords))

    # Create a population that captures the spikes from the input
    capture = p.Population(
        WIDTH * HEIGHT, p.IF_curr_exp(), structure=Grid2D(WIDTH / HEIGHT),
        label=f"Capture for device {i}")
    p.Projection(dev, capture, capture_conn, p.Convolution())

    # Create some convolutional "layers" (just 2, with 1 convolution each here)
    pop = p.Population(
        WIDTH * HEIGHT, p.IF_curr_exp(), structure=Grid2D(WIDTH / HEIGHT),
        label=f"Layer One {i}")
    pop_2 = p.Population(
        WIDTH * HEIGHT, p.IF_curr_exp(), structure=Grid2D(WIDTH / HEIGHT),
        label=f"Layer Two {i}")

    # Record the spikes so we know what happened
    capture.record("spikes")
    pop.record("spikes")
    pop_2.record("spikes")

    # Create convolution connections from the device -> first pop -> second pop
    # These use the same connector, but could be different if desired
    p.Projection(dev, pop, conn, p.Convolution())
    p.Projection(pop, pop_2, conn, p.Convolution())

    # Save for later use
    devices.append(dev)
    captures.append(capture)
    layer_1s.append(pop)
    layer_2s.append(pop_2)

# Run the simulation for long enough for packets to be sent
p.run(run_time)

# Get out the spikes
capture_spikes = list()
for capture in captures:
    capture_spikes.append(capture.get_data("spikes").segments[0].spiketrains)
layer_1_spikes = list()
for pop in layer_1s:
    layer_1_spikes.append(pop.get_data("spikes").segments[0].spiketrains)
layer_2_spikes = list()
for pop_2 in layer_2s:
    layer_2_spikes.append(pop_2.get_data("spikes").segments[0].spiketrains)

# Tell the software we are done with the board
p.end()

# Check if the data looks OK!
for i in range(len(devices)):

    # Go through the capture devices
    for neuron_id in range(len(capture_spikes[i])):
        # Work out x and y of neuron
        x = neuron_id % WIDTH
        y = neuron_id // WIDTH

        if len(capture_spikes[i][neuron_id]) > 0 or len(layer_1_spikes[i][neuron_id]) > 0 or len(layer_2_spikes[i][neuron_id]) > 0:
            print(f"{x}, {y}: {capture_spikes[i][neuron_id]}, {layer_1_spikes[i][neuron_id]}, {layer_2_spikes[i][neuron_id]}")

    # Go through the capture devices
    for neuron_id in range(len(capture_spikes[i])):
        # Work out x and y of neuron
        x = neuron_id % WIDTH
        y = neuron_id // WIDTH

        # Go through the spikes one by one and check they match up
        for spike_time in capture_spikes[i][neuron_id]:
            layer_1_spikes[i], found_times = find_square_of_spikes(
                x, y, spike_time, layer_1_spikes[i], f"device {i}",
                f"layer 1 {i}")

            # Check that the next layer matches as well
            for x_1, y_1, time_1 in found_times:
                layer_2_spikes[i], _ = find_square_of_spikes(
                    x_1, y_1, time_1, layer_2_spikes[i], f"layer 1 {i}",
                    f"layer 2 {i}")
