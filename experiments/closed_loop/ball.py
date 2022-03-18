import numpy as np
import time
import math
import os
import cv2
import matplotlib.pyplot as plt
import sys,tty,termios
import pdb

class _Getch:
    def __call__(self):
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(3)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

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

    next_cx = self.cx+self.vx*dt
    next_cy = self.cy+self.vy*dt
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
        cir = circle(bball.r)
        cx = int(bball.cx)
        cy = int(bball.cy)
        mat[cy-bball.r:cy+bball.r+1, cx-bball.r:cx+bball.r+1] = cir  
    
    return mat

if __name__ == "__main__":

    dt = 1 #ms
    l = 64
    w = int(l*3/4)
    r = 5
    cx = int(l/2)
    cy = int(w/2)
    vx = 0.1
    vy = 0.1
    duration = 60


    try:
        duration = int(sys.argv[1])

    except:
        print("Try python3 ball.py <duration>")
        quit()

    data_file_title = "ball_{:d}x{:d}_r{:d}_{:d}s.npy".format(l, w, r, duration)


    mat = np.zeros((w, l, duration*1000))

    
    
    bball = BouncingBall(dt, w, l, r, cx, cy, vx, vy)

    fps = 50
    LED_f = 200
    ball_update = 0
    LED_update = 0
    LED_on = 1
    t = 0
    while t < duration*1000:   
        time.sleep(0.001)

        
        if LED_update >= 1000/LED_f:
            LED_update = 0
            LED_on = 1 - LED_on

        mat[:,:,t] = update_pixel_frame(bball, LED_on)

        if ball_update >= 1000/fps:
            ball_update = 0      
            bball.update_c()
            print("{:.1f} %".format(np.round(t/(duration*1000),3)*100))
            cv2.imshow("Pixel Space", mat[:,:,t]*255)
            cv2.waitKey(1) 

        ball_update += 1
        LED_update += 1
        t += 1
    
    
    
    np.save(data_file_title, mat)

    
    
        
        