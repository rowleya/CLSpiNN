import matplotlib.pyplot as plt
import os
import time
import numpy as np
import math
# import cv2



def get():
        inkey = _Getch()
        while(1):
                k=inkey()
                if k!='':break
        if k=='\x1b[A':
                print("up")
        elif k=='\x1b[B':
                print("down")
        elif k=='\x1b[C':
                print("right")
        elif k=='\x1b[D':
                print("left")
        else:
                print("not an arrow key!")

class BouncingBall:

  def __init__(self, dt, w, l, r, cx, cy, vx, vy):
    self.dt = dt
    self.r = r
    self.w = w
    self.l = l
    self.cx = cx
    self.cy = cy
    self.vx = vx
    self.vy = vy

  def update_c(self):

    marg = 1

    next_cx = self.cx+self.vx*self.dt
    next_cy = self.cy+self.vy*self.dt
    # print("nxt_center: ({:.3f}, {:.3f})".format(next_cx, next_cy))
    if self.l-self.r-marg < next_cx:
        # print("Right edge")
        next_cx = (self.l-self.r-marg)-(next_cx-(self.l-self.r-marg))
        self.vx = -self.vx
    if marg+self.r > next_cx:
        # print("Left edge")
        next_cx = marg+self.r-(next_cx-(marg+self.r))
        self.vx = -self.vx
    self.cx = next_cx
    
    if self.w-self.r-marg < next_cy:
        # print("Bottom edge")
        next_cy = (self.w-self.r-marg)-(next_cy-(self.w-self.r-marg))
        self.vy = -self.vy
        
    if marg+self.r > next_cy:
        # print("Top edge")
        next_cy = marg+self.r-(next_cy-(marg+self.r))
        self.vy = -self.vy
    self.cy = next_cy


def ring(r):
    cir = np.zeros((2*r+1, 2*r+1))
    idx = np.array(range(-r, r+1, 1))
    for x in idx:
        for y in idx:
            d = math.sqrt(x*x+y*y)
            if d <= r and d > r-2:
                cir[r+x, r+y] = 1         
                
    return cir 

def circle(r):
    cir = np.zeros((2*r+1, 2*r+1))
    idx = np.array(range(-r, r+1, 1))
    for x in idx:
        for y in idx:
            d = math.sqrt(x*x+y*y)
            if d <= r:
                cir[r+x, r+y] = 1         
                
    return cir

def update_pixel_frame(bball, LED_on):
    
    mat = np.zeros((bball.w,bball.l))
    if LED_on:
        cir = ring(bball.r)
        cx = int(bball.cx)
        cy = int(bball.cy)
        mat[cy-bball.r:cy+bball.r+1, cx-bball.r:cx+bball.r+1] = cir  
    
    return mat

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

