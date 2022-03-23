import collections
import time
import pdb
import numpy as np

d = collections.deque(maxlen=10)

for i in range(20):
    d.append(i)
    print(d)
    time.sleep(0.05)

pdb.set_trace()