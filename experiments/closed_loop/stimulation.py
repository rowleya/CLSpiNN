
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


# Input imports
import pygame
sys.path.insert(1, '../../miscelaneous')
from stimulator import update_pixel_frame, BouncingBall


# An event frame has 32 bits <t[31]><x [30:16]><p [15]><y [14:0]>
P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16

class Stimulator:
    def __init__(self, d_params, end_of_sim):
        self.display = []
        self.ip_addr = d_params[0]
        self.port = int(d_params[1])
        self.h = d_params[2]
        self.w = d_params[3]
        self.mode = True # Automatic
        self.display_size = (346, int(self.h*346/self.w))
        self.input_q = multiprocessing.Queue()
        self.control_q = multiprocessing.Queue()
        self.end_of_sim = end_of_sim
        self.pix_mat_mp = multiprocessing.Array('I', int(np.prod((self.w, self.h))), lock=multiprocessing.Lock())
        self.pix_mat_np = np.frombuffer(self.pix_mat_mp.get_obj(), dtype='I').reshape((self.w, self.h))
        self.shared_pix_mat = (self.pix_mat_mp, self.pix_mat_np)
        self.p_screen = multiprocessing.Process(target=self.show_screen, args=())
        self.p_i_data = multiprocessing.Process(target=self.set_inputs, args=())
        self.p_stream = multiprocessing.Process(target=self.launch_input_handler, args=())

    
    def __enter__(self):

        pygame.init()     
        # pygame.display.set_caption("Lala")
        self.p_screen.start()
        self.p_i_data.start()
        self.p_stream.start()
    
    def __exit__(self, e, b, t): 
        pygame.quit()
        self.p_screen.join()
        self.p_i_data.join()
        self.p_stream.join()
    

    def set_inputs(self):

        dt = 1 #ms
        l = self.w
        w = self.h
        r = min(8, int(self.w*7/637+610/637))
        print(r)
        cx = int(l*1/4)
        cy = int(w*2/4)
        vx = -self.w/100
        vy = self.h/400
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
            if self.end_of_sim.value == 1:
                time.sleep(1)
                print("No more inputs to be sent")
                break
        
            if LED_update >= 1000/LED_f:
                LED_update = 0
                LED_on = 1 - LED_on

            mat, events = update_pixel_frame(bball, LED_on)
            if LED_on:
                pix_mat_mp, pix_mat_np = self.shared_pix_mat
                pix_mat_np[:,:] = np.transpose(mat)

            if ball_update >= 1000/fps:
                ball_update = 0      
                dx = 0
                dy = 0
                while not self.control_q.empty():
                    command = self.control_q.get(False)
                    if command == 'up':
                        dy = -1   
                    if command == 'down':
                        dy = 1   
                    if command == 'left':
                        dx = -1
                    if command == 'right':
                        dx = 1
                bball.update_c(self.mode, dx, dy)


                # im_final = cv2.resize(mat*255,(640,480), interpolation = cv2.INTER_NEAREST)
                # cv2.imshow("Pixel Space", mat*255)
                # cv2.waitKey(1) 
            

            coor = [bball.cx, bball.cy]
            self.input_q.put(events)

            ball_update += 1
            LED_update += 1
            t += 1


            time.sleep(0.0001)

    def launch_input_handler(self):

        NO_TIMESTAMP = 0x80000000
        polarity = 1

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


        while True:

            # Check if the spinnaker simulation has ended
            if self.end_of_sim.value == 1:
                time.sleep(1)
                print("No more events to be created")
                break
            
            events = []
            while not self.input_q.empty():
                events = self.input_q.get(False)


            data = b""
            for e in events:        
                x = e[0]
                y = e[1]
            
                packed = (
                    NO_TIMESTAMP + (polarity << P_SHIFT) +
                    (y << Y_SHIFT) + (x << X_SHIFT))

                data += pack("<I", packed)
            sock.sendto(data, (self.ip_addr, self.port))


    def update_screen(self):

        pix_mat_mp, pix_mat_np = self.shared_pix_mat
        
        image = np.zeros((pix_mat_np.shape[0],pix_mat_np.shape[1],3))
        image[:,:,1] = pix_mat_np*255

        return image.astype('uint8')

    def show_screen(self):

        self.display = pygame.display.set_mode(self.display_size)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        print(u"\u2191")
                        self.control_q.put('up')
                    if event.key == pygame.K_s:
                        print(u"\u2193")
                        self.control_q.put('down')
                    if event.key == pygame.K_a:
                        print(u"\u2190")
                        self.control_q.put('left')
                    if event.key == pygame.K_d:
                        print(u"\u2192")
                        self.control_q.put('right')

            Z = self.update_screen()
            surf = pygame.surfarray.make_surface(Z)
            surf = pygame.transform.scale(surf, self.display_size)

            self.display.blit(surf, (0, 0))

            pygame.display.update()  

