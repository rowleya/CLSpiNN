import logging
import sys
from durin.actuator import *
from durin.durin import *
from durin.visuals import *

import pyNN.spiNNaker as p
from pyNN.space import Grid2D
from spinn_front_end_common.utilities.database import DatabaseConnection
import socket
import pdb
import math
import collections

import cv2

import datetime
import time
import numpy as np
from random import randint, random
from struct import pack
from time import sleep
import sys
sys.path.insert(1, '../../miscelaneous')
import os
from threading import Thread

import matplotlib
matplotlib.use("TkAgg")
matplotlib.rcParams['toolbar'] = 'None' 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stimulator import update_pixel_frame, BouncingBall
import multiprocessing 

import random
import pygame
from pygame.locals import *
import sys

global end_of_sim, input_q, output_q, spike_q


#################################################################################################################################
#                                                           SPIF SETUP                                                          #
#################################################################################################################################


script_name = os.path.basename(__file__)

if len(sys.argv) != 5:
    print(f"python3 {script_name} <duration> <user_width> <nb_cols>")
    quit()
else:
    try:
        RUN_TIME = 1000*int(sys.argv[1])
        USER_WIDTH = int(sys.argv[2])
        USER_HEIGTH = 1
        NB_COLS = int(sys.argv[3])
        DURIN_IP = sys.argv[4]
    except:
        print("Something went wrong with the arguments")
        quit()
print("\n About to start things ... \n")

# Device parameters are "pipe", "chip_coords", "ip_address", "port"
DEVICE_PARAMETERS = (0, (0, 0), "172.16.223.11", 3333)
PORT_SPIN2CPU = int(random.randint(12000, 16000))

send_fake_spikes = True

# An event frame has 32 bits <t[31]><x [30:16]><p [15]><y [14:0]>
P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16

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
This function returns a port number to be used during SpikeLiveConnection
'''
def get_port(first_val):
    count = 0
    while True:
        count += 1
        yield first_val + count

'''
This is what's done whenever the CPU receives a spike sent by SpiNNaker
'''
def receive_spikes_from_sim(label, time, neuron_ids):

    global end_of_sim, input_q, output_q
    
    for n_id in neuron_ids:
        # print(f"Spike --> MN[{n_id}]")
        output_q.put(n_id, False)

''' 
This function creates a list of weights to be used when connecting pixels to motor neurons
'''
def generate_w_list(w, h, n=0):
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

                        pix_per_col = int(USER_WIDTH/NB_COLS)
                        for post_idx in range(NB_COLS):

                            weight = 0.000001
                            x_weight = 2*w_fovea*(abs((x+0.5)-w/2)/(w-1))

                            # Move right (when stimulus on the left 'hemisphere')    
                            if (x >= (post_idx+0)*pix_per_col) and (x < (post_idx+1)*pix_per_col):
                                weight = 4                                             
                            
                            conn_list.append((pre_idx, post_idx, weight, delay))
        
    return conn_list
'''
This function launches the input generator and packs generated events to send them to SpiNNaker
'''
def launch_input_handler():

    global DEVICE_PARAMETERS

    global end_of_sim, input_q

    global X_SHIFT, Y_SHIFT, P_SHIFT, SLEEP_TIME, N_PACKETS

    p_i_data = multiprocessing.Process(target=set_inputs, args=())
    p_i_data.start()
    

    ip_addr = DEVICE_PARAMETERS[2]
    port = DEVICE_PARAMETERS[3]

    NO_TIMESTAMP = 0x80000000
    polarity = 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


    while True:

        # Check if the spinnaker simulation has ended
        if end_of_sim.value == 1:
            time.sleep(1)
            print("No more events to be created")
            break
        
        events = []
        while not input_q.empty():
            events = input_q.get(False)


        data = b""
        for e in events:        
            x = e[0]
            y = e[1]
        
            packed = (
                NO_TIMESTAMP + (polarity << P_SHIFT) +
                (y << Y_SHIFT) + (x << X_SHIFT))

            data += pack("<I", packed)
        sock.sendto(data, (ip_addr, port))


    p_i_data.join()
        


#################################################################################################################################
#                                                        SPINNAKER SETUP                                                        #
#################################################################################################################################

def run_spinnaker_sim():

    global end_of_sim, input_q, output_q
    global DEVICE_PARAMETERS, P_SHIFT, Y_SHIFT, X_SHIFT, SUB_HEIGHT, SUB_WIDTH, WEIGHT, RUN_TIME

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

    if send_fake_spikes:
        # This is only used with the above to send data to the Ethernet
        connection = DatabaseConnection(launch_input_handler, local_port=None)

        print(f"DB Connection port = {connection.local_port}")
        # time.sleep(5)

        # This is used with the connection so that it starts sending when the simulation starts
        p.external_devices.add_database_socket_address(None, connection.local_port, None)

    capture_conn = p.ConvolutionConnector([[WEIGHT]])

    # These are our external retina devices connected to SPIF devices
    devices = list()
    captures = list()
    pipe = DEVICE_PARAMETERS[0]
    chip_coords = DEVICE_PARAMETERS[1]

    dev = p.Population(None, p.external_devices.SPIFRetinaDevice(
        pipe=pipe, width=WIDTH, height=HEIGHT, sub_width=SUB_WIDTH,
        sub_height=SUB_HEIGHT, input_x_shift=X_SHIFT, input_y_shift=Y_SHIFT, 
        chip_coords=chip_coords, base_key=None, board_address=None))

    # Create a population that captures the spikes from the input
    capture = p.Population(WIDTH * HEIGHT, p.IF_curr_exp(), 
                            structure=Grid2D(WIDTH / HEIGHT),
                            label=f"Capture for device SPIF")

    p.Projection(dev, capture, capture_conn, p.Convolution())

    # Record the spikes so we know what happened
    # capture.record("spikes")

    # Save for later use
    devices.append(dev)
    captures.append(capture)


        


    motor_neurons = p.Population(NB_COLS, celltype(**cell_params), label="motor_neurons")


    conn_list = generate_w_list(WIDTH, HEIGHT, SUB_HEIGHT*SUB_WIDTH)


    cell_conn = p.FromListConnector(conn_list, safe=True)  
    
    con_move = p.Projection(capture, motor_neurons, cell_conn, receptor_type='excitatory')
        
    # Spike reception (from SpiNNaker to CPU)
    port = PORT_SPIN2CPU
    live_spikes_receiver = p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=["motor_neurons"], local_port=port)
    _ = p.external_devices.activate_live_output_for(motor_neurons, database_notify_port_num=live_spikes_receiver.local_port)
    live_spikes_receiver.add_receive_callback("motor_neurons", receive_spikes_from_sim)


    # Run the simulation for long enough for packets to be sent
    p.run(RUN_TIME)
    # p.external_devices.run_forever(sync_time=0)

    # Let other processes know that spinnaker simulation has come to an ned
    end_of_sim.value = 1

    # Tell the software we are done with the board
    p.end()



 

def set_inputs():

    global end_of_sim, input_q, obj_coor_q, control_q, shared_pix_mat

    dt = 1 #ms
    l = WIDTH
    w = HEIGHT
    r = 0 #min(8, int(WIDTH*7/637+610/637))
    print(r)
    cx = int(l*1/4)
    cy = int(w*2/4)
    vx = -WIDTH/1000
    vy = 0
    duration = 60


    
    bball = BouncingBall(dt, w, l, r, cx, cy, vx, vy)

    fps = 30
    LED_f = 200
    ball_update = 0
    LED_update = 0
    LED_on = 1
    t = 0

    while True:

        # Check if the spinnaker simulation has ended
        if end_of_sim.value == 1:
            time.sleep(1)
            print("No more inputs to be sent")
            try:
                cv2.destroyAllWindows()
            except:
                pass
            break
    
        if LED_update >= 1000/LED_f:
            LED_update = 0
            LED_on = 1 - LED_on

        mat, events = update_pixel_frame(bball, LED_on)
        if LED_on:
            pix_mat_mp, pix_mat_np = shared_pix_mat
            pix_mat_np[:,:] = np.transpose(mat)

        if ball_update >= 1000/fps:
            ball_update = 0      
            dx = 0
            dy = 0
            while not control_q.empty():
                command = control_q.get(False)
                if command == 'up':
                    dy = -1   
                if command == 'down':
                    dy = 1   
                if command == 'left':
                    dx = -1
                if command == 'right':
                    dx = 1
            bball.update_c(True, dx, dy)


            # im_final = cv2.resize(mat*255,(640,480), interpolation = cv2.INTER_NEAREST)
            # cv2.imshow("Pixel Space", mat*255)
            # cv2.waitKey(1) 
        

        coor = [bball.cx, bball.cy]
        # obj_coor_q.put(coor)
        input_q.put(events)

        ball_update += 1
        LED_update += 1
        t += 1


        time.sleep(0.0001)


'''
Get spike count based on received spikes
'''
def get_outputs():

    global end_of_sim, output_q, spike_q
    
    dt = 0.040
    
    start = time.time()
    current_t = time.time()
    next_check = current_t + dt

    spike_times = []
    spike_count = np.zeros(NB_COLS,)
    nb_spikes_max = 100
    for i in range(NB_COLS):  # 4 motor neurons          
        spike_times.append(collections.deque(maxlen=nb_spikes_max))
        
    while True:

        # Check if the spinnaker simulation has ended
        if end_of_sim.value == 1:
            time.sleep(1)
            print("No more outputs to be received")
            break

        while not output_q.empty():
            out = output_q.get(False)
            current_t = time.time()
            elapsed_t = (current_t - start)*1000
            spike_times[out].append(elapsed_t)
        
        if current_t >= next_check:
            # print(f"Checking @ t={current_t}")
            next_check = current_t + dt
            for i in range(NB_COLS):  # 4 motor neurons
                train = np.array(spike_times[i])
                short_train = train[train>(current_t-dt-start)*1000]
                # print(f"For MN[{i}]: {len(train)} >= {len(short_train)} (Train vs Short Train)")
                spike_count[i] = len(short_train)
            spike_q.put(spike_count, False)

        time.sleep(0.005)

def rt_plot(i, fig, axs, t, cols, spike_count):

    global spike_q, obj_coor_q, end_of_sim

    while not spike_q.empty():
        spike_count = spike_q.get(False)

    # Add x and y to lists
    t.append(datetime.datetime.now().strftime('%H:%M:%S.%f'))

    t = t[-40:]

    for i in range(NB_COLS):
        cols[i].append(spike_count[i]) 
        cols[i] = cols[i][-40:] # Limit x and y lists to 100 items



    txt_col = ""
    

    # Draw x and y lists

    max_y = int(1.2*max(WIDTH, HEIGHT))


    max_y = 40

    for i in range(NB_COLS):

        axs[i].clear()
        axs[i].plot(t, cols[i], color='g')
        # axs[i].text(t[0], 0.75*max_y, txt_col, fontsize='xx-large')
        axs[i].xaxis.set_visible(False)
        axs[i].yaxis.set_visible(False)
        # axs[i].set_ylabel(f'col_{i}')
        axs[i].set_ylim([0,max_y])


    if end_of_sim.value == 1:
            time.sleep(1)
            print("No more data to be shown.")
            plt.savefig('EndOfSim.png')
            plt.close(fig)


def oscilloscope():


    print("Starting Oscilloscope")

    # Create figure for plotting
    fig, axs = plt.subplots(NB_COLS, figsize=(8, 8))
    fig.canvas.manager.set_window_title('World Space')


    t = []
    cols = []
    for j in range(NB_COLS):
        cols.append([])

    i = 0
    spike_count = np.ones(NB_COLS, )*(-40)

    # Set up plot to call rt_xyz() function periodically
    ani = animation.FuncAnimation(fig, rt_plot, fargs=(fig, axs, t, cols, spike_count), interval=1)
    plt.show()


class Viewer:
    def __init__(self, update_func, control_q, display_size):
        self.display_size = display_size
        self.update_func = update_func
        self.control_q = control_q
        pygame.init()
        self.display = pygame.display.set_mode(display_size)
    
    def set_title(self, title):
        pygame.display.set_caption(title)
    
    def start(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        print(u"\u2191")
                        control_q.put('up')
                    if event.key == pygame.K_s:
                        print(u"\u2193")
                        control_q.put('down')
                    if event.key == pygame.K_a:
                        print(u"\u2190")
                        control_q.put('left')
                    if event.key == pygame.K_d:
                        print(u"\u2192")
                        control_q.put('right')

            Z = self.update_func()
            surf = pygame.surfarray.make_surface(Z)
            surf = pygame.transform.scale(surf, self.display_size)

            self.display.blit(surf, (0, 0))

            pygame.display.update()
            # time.sleep(1/25)

        pygame.quit()

def update_viewer():

    global shared_pix_mat

    pix_mat_mp, pix_mat_np = shared_pix_mat
    
    image = np.zeros((pix_mat_np.shape[0],pix_mat_np.shape[1],3))
    image[:,:,1] = pix_mat_np*255

    return image.astype('uint8')

def run_screen():

    global control_q


    viewer = Viewer(update_viewer, control_q, (640, 64))
    viewer.start()


if __name__ == '__main__':

    global end_of_sim, input_q, output_q, spike_q, obj_coor_q, control_q, shared_pix_mat
    

    manager = multiprocessing.Manager()

    end_of_sim = manager.Value('i', 0)
    pix_mat_mp = multiprocessing.Array('I', int(np.prod((WIDTH, HEIGHT))), lock=multiprocessing.Lock())
    pix_mat_np = np.frombuffer(pix_mat_mp.get_obj(), dtype='I').reshape((WIDTH, HEIGHT))

    shared_pix_mat = (pix_mat_mp, pix_mat_np)

    input_q = multiprocessing.Queue() # events
    control_q = multiprocessing.Queue() # commands
    output_q = multiprocessing.Queue() # events
    spike_q = multiprocessing.Queue() # signals
    obj_coor_q = multiprocessing.Queue() # visualization




    p_o_data = multiprocessing.Process(target=get_outputs, args=())
    p_rt_osc = multiprocessing.Process(target=oscilloscope, args=())
    p_screen = multiprocessing.Process(target=run_screen, args=())
    p_spiNN = multiprocessing.Process(target=run_spinnaker_sim, args=())

    


    # run_spinnaker_sim()

    printing = True
    print(DURIN_IP)
    with Durin(DURIN_IP, stream_command=StreamOn("172.16.223.87", 4501, 100)) as durin:
        
        p_o_data.start()
        p_rt_osc.start()
        p_screen.start()
        p_spiNN.start()

        
        
        while True:            
           time.sleep(1)

            



    p_spiNN.join()
    p_o_data.join()
    p_rt_osc.join()
    p_screen.join()