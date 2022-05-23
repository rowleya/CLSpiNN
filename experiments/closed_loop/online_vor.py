
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
import collections

# Input imports
import pygame
sys.path.insert(1, '../../miscelaneous')
from stimulator import update_pixel_frame, BouncingBall

# Oscilloscope Imports
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use("TkAgg")
matplotlib.rcParams['toolbar'] = 'None' 


global end_of_sim, input_q, output_q, spike_q


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


M_LABELS = ["go_right", "go_left", "go_up", "go_down"]
NB_PTS = 100

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

    global end_of_sim, input_q, output_q
    
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
'''
This function launches the input generator and packs generated events to send them to SpiNNaker
'''
def launch_input_handler():

    global X_SHIFT, Y_SHIFT, P_SHIFT, SLEEP_TIME, N_PACKETS, DEVICE_PARAMETERS
    global end_of_sim, input_q

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
    r = min(8, int(WIDTH*7/637+610/637))
    print(r)
    cx = int(l*1/4)
    cy = int(w*2/4)
    vx = -WIDTH/100
    vy = HEIGHT/400
    duration = 60


    
    bball = BouncingBall(dt, w, l, r, cx, cy, vx, vy)

    fps = 50
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
        obj_coor_q.put(coor)
        input_q.put(events)

        ball_update += 1
        LED_update += 1
        t += 1


        time.sleep(0.0001)



def get_outputs():

    global end_of_sim, output_q, spike_q
    
    dt = 0.100
    
    start = time.time()
    current_t = time.time()
    next_check = current_t + dt

    spike_times = []
    spike_count = [0,0,0,0]
    nb_spikes_max = 100
    for i in range(4):  # 4 motor neurons          
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
            for i in range(4):  # 4 motor neurons
                train = np.array(spike_times[i])
                short_train = train[train>(current_t-dt-start)*1000]
                # print(f"For MN[{i}]: {len(train)} >= {len(short_train)} (Train vs Short Train)")
                spike_count[i] = len(short_train)
            spike_q.put(spike_count, False)

        time.sleep(0.005)

def rt_plot(i, fig, axs, t, x, y, mn, obj_xy, spike_count):

    global spike_q, obj_coor_q, end_of_sim


    while not obj_coor_q.empty():
        obj_xy = obj_coor_q.get(False)
        # print(spike_count)

    while not spike_q.empty():
        spike_count = spike_q.get(False)

    # Add x and y to lists
    txt_x = txt_y = ""
    t.append(datetime.datetime.now().strftime('%H:%M:%S.%f'))
    t = t[-NB_PTS:]

    x.append(obj_xy[0])
    y.append(obj_xy[1])
    x = x[-NB_PTS:]
    y = y[-NB_PTS:]
    if x[-1] != -NB_PTS:
        txt_x = "x = {:.3f} ".format(x[-1]) 
    if y[-1] != -NB_PTS:
        txt_y = "y = {:.3f} ".format(y[-1])

    max_y = int(1.2*max(WIDTH, HEIGHT))


    axs[0].clear()
    axs[0].plot(t, x)
    axs[0].plot(t, y)
    axs[0].text(t[0], 0.75*max_y, txt_x, fontsize='xx-large')
    axs[0].text(t[0], 0.15*max_y, txt_y, fontsize='xx-large')
    axs[0].xaxis.set_visible(False)
    axs[0].set_ylabel('Pixels')
    axs[0].set_ylim([0,max_y])


    max_y = 200

    for j in range(4):
        mn[j].append(spike_count[j])
        mn[j] = mn[j][-NB_PTS:]

        axs[j+1].clear()
        axs[j+1].plot(t, mn[j], color='g')
        axs[j+1].xaxis.set_visible(False)
        axs[j+1].set_ylabel(M_LABELS[j])
        axs[j+1].set_ylim([0,max_y])



    if end_of_sim.value == 1:
            time.sleep(1)
            print("No more data to be shown.")
            plt.savefig('EndOfSim.png')
            plt.close(fig)


def oscilloscope():


    print("Starting Oscilloscope")

    # Create figure for plotting
    fig, axs = plt.subplots(5, figsize=(8, 8))
    fig.canvas.manager.set_window_title('World Space')


    t = []
    x = []
    y = []

    mn = []
    for i in range(4):
        mn.append([])

    i = 0
    obj_xy = -NB_PTS*np.ones((2,))
    spike_count = -NB_PTS*np.ones((4,))

    # Set up plot to call rt_xyz() function periodically
    ani = animation.FuncAnimation(fig, rt_plot, fargs=(fig, axs, t, x, y, mn, obj_xy, spike_count), interval=1)
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

        pygame.quit()

def update():

    global shared_pix_mat

    pix_mat_mp, pix_mat_np = shared_pix_mat
    
    image = np.zeros((pix_mat_np.shape[0],pix_mat_np.shape[1],3))
    image[:,:,1] = pix_mat_np*255

    return image.astype('uint8')

def show_screen():

    global control_q, end_of_sim


    viewer = Viewer(update, control_q, (346, int(USER_HEIGTH*346/USER_WIDTH)))
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




    p_i_data = multiprocessing.Process(target=launch_input_handler, args=())
    p_o_data = multiprocessing.Process(target=get_outputs, args=())
    p_visual = multiprocessing.Process(target=oscilloscope, args=())
    p_screen = multiprocessing.Process(target=show_screen, args=())
    p_spiNN = multiprocessing.Process(target=run_spinnaker_sim, args=())
    

    p_i_data.start()
    p_o_data.start()
    p_visual.start()
    p_screen.start()
    p_spiNN.start()

    
    while True:

        if end_of_sim.value == 1:
            time.sleep(1)
            print("No more commands to be executed.")
            break


    p_spiNN.join()
    p_screen.join()
    p_visual.join()
    p_o_data.join()
    p_i_data.join()