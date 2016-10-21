#!/usr/bin/env python
import sys
from os import system
import numpy as np
import time

# Start timer
start = time.time()

# HYPERPARAMETERS TO EXPLORE
# Simulation hyperparameters
patchSizes = [4**i for i in range(1,5)]
iterations = 200
numImages  = 300
snr        = 1000.

# Architecture hyperparameters
keys              = [2,3]
numPhotoreceptors = [4**i for i in range(1,5)]
numSubunits       = [4**i for i in range(1,5)]
gains             = [0.25, 1.]




progress = 0
totalIterations = len(patchSizes)*len(keys)*len(numPhotoreceptors)*len(numSubunits)*len(gains)
print("%i executions required." %totalIterations)
for p in patchSizes:
    for k in keys:
        for nPhoto in numPhotoreceptors:
            for nSub in numSubunits:
                for g in gains:
                    sys.stdout.write('\r')
                    cmd = "python lnl_model.py -n %i -p %i -i %i -k %i -o %i -s %i -g %f -e %f" %(numImages, p, iterations, k, nPhoto, nSub, g, snr)
                    system(cmd)
                    time.sleep(1)
                    progress += 1
                    msg = "MASTER PROGRESS IS %f%% COMPLETE AFTER %d MINUTES." %(100.*float(progress)/totalIterations,(time.time()-start)/60.)
                    sys.stdout.write(msg)
                    sys.stdout.flush()
print('')

