
import multiprocessing 
import argparse
import sys, os, time
import pdb



from visualization import *
from stimulation import *
from computation import *


def parse_args():

    
    parser = argparse.ArgumentParser(description='SpiNNaker-SPIF Simulation with Artificial Data')

    parser.add_argument('--runtime', type= str, help="Run Time, in seconds", default=10)
    parser.add_argument('--ip', type= str, help="SPIF's IP address", default="172.16.223.98")
    parser.add_argument('--port', type= str, help="SPIF's port", default=3333)
    parser.add_argument('--pipe', type= str, help="SPIF's pipe", default=parser.parse_args().port-3333)
    parser.add_argument('--width', type= str, help="Image size (in px)", default=24)
    parser.add_argument('--height', type= str, help="Image size (in px)", default=int(parser.parse_args().width*3/4))

    return parser.parse_args()
   

if __name__ == '__main__':

    args = parse_args()
        
    manager = multiprocessing.Manager()
    end_of_sim = manager.Value('i', 0)
    output_q = multiprocessing.Queue() # events


    spin = Computer(args, output_q)
    stim = Stimulator(args, end_of_sim)
    osci = Oscilloscope(spin.labels, output_q, end_of_sim)


    with spin:
        with stim:
            with osci:
            
                spin.run_sim()
                end_of_sim.value = 1 # Let other processes know that simulation stopped
                spin.wrap_up()
            