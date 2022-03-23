import spynnaker8 as p
from pyNN.space import Grid2D
import socket
import pdb
import math
import datetime
import time
import numpy as np
from random import randint, random
from struct import pack
from time import sleep
from spinn_front_end_common.utilities.database import DatabaseConnection
from threading import Thread

import multiprocessing 

#################################################################################################################################
#                                                           SPIF SETUP                                                          #
#################################################################################################################################

# Device parameters are "pipe", "chip_coords", "ip_address", "port"
DEVICE_PARAMETERS = (0, (0, 0), "172.16.223.98", 3333)

send_fake_spikes = True

# An event frame has 32 bits <t[31]><x [30:16]><p [15]><y [14:0]>
P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16

if send_fake_spikes:
    WIDTH = 8
    HEIGHT = int(WIDTH*3/4)*0+8
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
SLEEP_TIME = 0.1
N_PACKETS = 10

# Run time if send_fake_spikes is False
RUN_TIME = 10000 #[ms]

if send_fake_spikes:
    RUN_TIME = (2*4*N_PACKETS + 1) * SLEEP_TIME * 1000


print(f"SPIF : {DEVICE_PARAMETERS[2]}:{DEVICE_PARAMETERS[3]}")
print(f"Pixel Matrix : {WIDTH}x{HEIGHT} (Real={not send_fake_spikes})")
print(f"Run Time : {RUN_TIME}")
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
    print("1 for " + label)
    # print(neuron_ids)

''' 
This function creates a list of weights to be used when connecting pixels to motor neurons
'''
def create_conn_list(w, h, post_idx):
    w_fovea = 100
    conn_list = []
    
    # print(f"\n\n\n\n\n\n\\n\n\n\n {post_idx}\n\n\n\n\n\n\n\n\n\n")
    # time.sleep(5)

    delay = 1 # 1 [ms]
    for y in range(h):
        for x in range(w):
            pre_idx = y*w+x

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
        

    # print(conn_list)
    return conn_list


def send_retina_input(ip_addr, port):
    """ This is used to send random input to the Ethernet listening in SPIF
    """
    global X_SHIFT, Y_SHIFT, P_SHIFT, SLEEP_TIME, N_PACKETS
    NO_TIMESTAMP = 0x80000000
    polarity = 1

    sleep(0.5 + (random() / 4.0))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    count = 0

    combi = [(3,3),(1,3),(3,1),(1,1)]
    for k in range(4):
        x = int(WIDTH*combi[k][0]/4)
        y = int(HEIGHT*combi[k][1]/4)
        for _ in range(N_PACKETS):
            n_spikes = 10
            data = b""
            for _ in range(n_spikes):
                packed = (
                    NO_TIMESTAMP + (polarity << P_SHIFT) +
                    (y << Y_SHIFT) + (x << X_SHIFT))
                # print(f"Sending x={x}, y={y}, polarity={polarity}, "
                #     f"packed={hex(packed)}")
                count+= 1

                data += pack("<I", packed)
            sock.sendto(data, (ip_addr, port))
            sleep(SLEEP_TIME)
        
    print(f"count is --> {count}")


def start_fake_senders():

    global DEVICE_PARAMETERS

    ip_addr = DEVICE_PARAMETERS[2]
    port = DEVICE_PARAMETERS[3]
    
    t = multiprocessing.Process(target=send_retina_input, args=(ip_addr, port,))
    t.start()
        


#################################################################################################################################
#                                                        SPINNAKER SETUP                                                        #
#################################################################################################################################

def run_spinnaker_sim(end_of_sim, input_q, output_q):

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
        connection = DatabaseConnection(start_fake_senders, local_port=None)

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
        sub_height=SUB_HEIGHT, input_x_shift=X_SHIFT, input_y_shift=Y_SHIFT))

    # Create a population that captures the spikes from the input
    capture = p.Population(WIDTH * HEIGHT, p.IF_curr_exp(), 
                            structure=Grid2D(WIDTH / HEIGHT),
                            label=f"Capture for device SPIF")

    p.Projection(dev, capture, capture_conn, p.Convolution())

    # Record the spikes so we know what happened
    capture.record("spikes")

    # Save for later use
    devices.append(dev)
    captures.append(capture)


    m_labels = ["go_right", "go_left", "go_up", "go_down"]
        

    motor_neurons = []
    for mn_i in range(4):
        motor_neurons.append(p.Population(1, celltype(**cell_params), label=m_labels[mn_i]))

    
    # motor_neurons.record("spikes")

    conn_list_0 = create_conn_list(WIDTH, HEIGHT,0)
    conn_list_1 = create_conn_list(WIDTH, HEIGHT,1)
    conn_list_2 = create_conn_list(WIDTH, HEIGHT,2)
    conn_list_3 = create_conn_list(WIDTH, HEIGHT,3)



    for conn in conn_list_0:
        x = conn[0]%WIDTH
        y = math.floor(conn[0]/WIDTH)
        print(f"({x},{y}) : {conn[0]}\t-->\t{conn[1]} : w={conn[2]}")

    for conn in conn_list_1:
        x = conn[0]%WIDTH
        y = math.floor(conn[0]/WIDTH)
        print(f"({x},{y}) : {conn[0]}\t-->\t{conn[1]} : w={conn[2]}")

    for conn in conn_list_2:
        x = conn[0]%WIDTH
        y = math.floor(conn[0]/WIDTH)
        print(f"({x},{y}) : {conn[0]}\t-->\t{conn[1]} : w={conn[2]}")

    for conn in conn_list_3:
        x = conn[0]%WIDTH
        y = math.floor(conn[0]/WIDTH)
        print(f"({x},{y}) : {conn[0]}\t-->\t{conn[1]} : w={conn[2]}")


    cell_conn_0 = p.FromListConnector(conn_list_0, safe=True)
    cell_conn_1 = p.FromListConnector(conn_list_1, safe=True)
    cell_conn_2 = p.FromListConnector(conn_list_2, safe=True)
    cell_conn_3 = p.FromListConnector(conn_list_3, safe=True)



    con_move = []

    con_move.append(p.Projection(capture, motor_neurons[0], cell_conn_0, receptor_type='excitatory'))
    con_move.append(p.Projection(capture, motor_neurons[1], cell_conn_1, receptor_type='excitatory'))
    con_move.append(p.Projection(capture, motor_neurons[2], cell_conn_2, receptor_type='excitatory'))
    con_move.append(p.Projection(capture, motor_neurons[3], cell_conn_3, receptor_type='excitatory'))

    # w_0 = con_move[0].get("weight", format="array")
    w_1 = con_move[1].get("weight", format="array")
    w_2 = con_move[2].get("weight", format="array")
    w_3 = con_move[3].get("weight", format="array")

    # Spike reception (from SpiNNaker to CPU)

    port_generator = get_port(14000)
    


        
    live_spikes_receiver_1 = p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=[m_labels[1]], local_port=next(port_generator))
    live_spikes_receiver_2 = p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=[m_labels[2]], local_port=next(port_generator))
    live_spikes_receiver_3 = p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=[m_labels[3]], local_port=next(port_generator))
    live_spikes_receiver_0 = p.external_devices.SpynnakerLiveSpikesConnection(receive_labels=[m_labels[0]], local_port=next(port_generator))


    _ = p.external_devices.activate_live_output_for(motor_neurons[1], database_notify_port_num=live_spikes_receiver_1.local_port)
    live_spikes_receiver_1.add_receive_callback(m_labels[1], receive_spikes_from_sim)
    _ = p.external_devices.activate_live_output_for(motor_neurons[2], database_notify_port_num=live_spikes_receiver_2.local_port)
    live_spikes_receiver_2.add_receive_callback(m_labels[2], receive_spikes_from_sim)
    _ = p.external_devices.activate_live_output_for(motor_neurons[3], database_notify_port_num=live_spikes_receiver_3.local_port)
    live_spikes_receiver_3.add_receive_callback(m_labels[3], receive_spikes_from_sim)
    _ = p.external_devices.activate_live_output_for(motor_neurons[0], database_notify_port_num=live_spikes_receiver_0.local_port)
    live_spikes_receiver_0.add_receive_callback(m_labels[0], receive_spikes_from_sim)

    # pdb.set_trace()

    # Run the simulation for long enough for packets to be sent
    p.run(RUN_TIME)


    # # Get out the spikes

    in_spikes = capture.get_data("spikes")
    # out_spikes = motor_neurons.get_data("spikes")

    for h in range(HEIGHT):
        for w in range(WIDTH):
            l = len(np.asarray(in_spikes.segments[0].spiketrains[h*WIDTH+w]))
            print(f"({w},{h})-->{l}")


    pdb.set_trace()



    # Tell the software we are done with the board
    p.end()

    # Let other processes know that spinnaker simulation has come to an ned
    end_of_sim.value = 1


def produce_data(l, w, r, vx, vy, duration):

    dt = 1 # ms
    cx = int(2*l/4)
    cy = int(2*w/4)

    data_file_title = "ball_{:d}x{:d}_r{:d}_{:d}s.npy".format(l, w, r, duration)


    mat = np.zeros((w, l, duration*1000))
    coor = np.zeros((2,duration*1000))

    
    
    bball = BouncingBall(dt, w, l, r, cx, cy, vx, vy)

    fps = 100
    LED_f = 100
    ball_update = 0
    LED_update = 0
    LED_on = 1
    t = 0
    LED_blink = True
    while t < duration*1000:   
        time.sleep(0.001)

        
        if LED_update >= 1000/(LED_f*2):
            LED_update = 0
            LED_blink = True
        
        if LED_blink:       
            LED_blink = False
            LED_on = 1
        else:         
            LED_on = 0
        mat[:,:,t] = update_pixel_frame(bball, LED_on)

        if ball_update >= 1000/fps:
            ball_update = 0      
            bball.update_c()
            # print("\r{:.1f} %".format(np.round(t/(duration*1000),3)*100), end="")

            # im_final = cv2.resize(mat[:,:,t]*255,(640,480), interpolation = cv2.INTER_NEAREST)
            # cv2.imshow("Pixel Space", im_final)
            # # cv2.imshow("Pixel Space", mat[:,:,t]*255)
            # cv2.waitKey(1) 
        
        coor[0,t] = bball.cx
        coor[1,t] = bball.cy

        ball_update += 1
        LED_update += 1
        t += 1
    
    
    
    # np.save(data_file_title, mat)
    
    return mat, coor

def set_inputs(end_of_sim, input_q):

    while True:

        # Check if the spinnaker simulation has ended
        if end_of_sim.value == 1:
            time.sleep(1)
            print("No more inputs to be sent")
            break
    
        # input_q.put([0, 9])
        # input_q.put([1, 0])
        # input_q.put([2, 0])
        # input_q.put([3, 0])

        time.sleep(0.005)



def get_outputs(end_of_sim, output_q):

    while True:

        # Check if the spinnaker simulation has ended
        if end_of_sim.value == 1:
            time.sleep(1)
            print("No more outputs to be received")
            break

        # while not output_q.empty():
        #     spike = output_q.get(False)

        time.sleep(0.005)

if __name__ == '__main__':

    manager = multiprocessing.Manager()

    end_of_sim = manager.Value('i', 0)

    input_q = multiprocessing.Queue()
    output_q = multiprocessing.Queue()


    p_i_data = multiprocessing.Process(target=set_inputs, args=(end_of_sim, input_q,))
    p_o_data = multiprocessing.Process(target=get_outputs, args=(end_of_sim, output_q,))
    # p_spinn = multiprocessing.Process(target=run_spinnaker_sim, args=(end_of_sim, input_q, output_q,))
    
    run_spinnaker_sim(end_of_sim, input_q, output_q)

    p_i_data.start()
    p_o_data.start()
    # p_spinn.start()


    p_i_data.join()
    p_o_data.join()
    # p_spinn.join()