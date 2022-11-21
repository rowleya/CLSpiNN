
import multiprocessing 
import argparse
import sys, os, time
import pdb



from visualization import *
from stimulation import *
from computation import *


def parse_args():

    
    parser = argparse.ArgumentParser(description='SpiNNaker-SPIF Simulation with Artificial Data')

    parser.add_argument('-r','--runtime', type=int, help="Run Time, in seconds", default=10)
    parser.add_argument('-i', '--ip', type= str, help="SPIF's IP address", default="172.16.223.98")
    parser.add_argument('-p', '--port', type=int, help="SPIF's port", default=3333)
    parser.add_argument('-m', '--mode', type=str, help="Stimulus mode: auto | manual", default='auto')
    parser.add_argument('-w', '--width', type=int, help="Image size (in px)", default=24)
    
    return parser.parse_args()
   

if __name__ == '__main__':

    args = parse_args()
        
    manager = multiprocessing.Manager()
    end_of_sim = manager.Value('i', 0)
    output_q = multiprocessing.Queue() # events


    spin = Computer(args, output_q)
    stim = Stimulator(args, end_of_sim)
    osci = Oscilloscope(spin.labels, output_q, end_of_sim)


    with stim:
            with osci:
            
                sleep(20)
                end_of_sim.value = 1 # Let other processes know that simulation stopped

    
    