# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 10:25:28 2015

@author: pietro
"""

import numpy as np
import matplotlib . pyplot as plt
from hdnet . spikes import Spikes
from hdnet . spikes_model import SpikeModel
import elephant.spike_train_generation as stoc
import elephant.conversion as conv
import quantities as pq
import neo
import models_for_worm_calibration as models
import os
## Stationary SIP
#sts = stoc.single_interaction_process(
#    rate=10*pq.Hz, rate_c=2*pq.Hz, t_stop=5*pq.s, n=5)
#sts += stoc._n_poisson(10*pq.Hz, 5*pq.s, n=95)

# Non_stationary SIP
#rate_nonstat = neo.AnalogSignal(
#    [10]*4000+[50]*1000+[10]*4000, units=pq.Hz, sampling_period=1*pq.ms)
#sts = stoc.sip_nonstat(5, 95, rate_b=rate_nonstat, rate_c=4*pq.Hz)

# Worms calibration models
#data_type = 3
#sse_parameters = (5,5,10)
#sts = models.generate_sts(data_type=data_type, sse_params=sse_parameters)[0]

data_path = './artificial_data_simul/data_stp/stp_data%i.npy' % data_idx
if not os.path.exists(data_path):
    raise ValueError('Data path not existing')
#datafile = neo.NeoHdf5IO(data_path)
comm = MPI.COMM_WORLD   # create MPI communicator
rank = comm.Get_rank()  # get rank of current MPI task
size = comm.Get_size()  # get tot number of MPI tasks
print size
datafile = np.load(data_path).item()

sts_bin = conv.BinnedSpikeTrain(
    sts, binsize=1*pq.ms).to_bool_array().astype('int')
spikes = Spikes(spikes=sts_bin)

spikes_raster = spikes.rasterize()
plt.figure()
for i in range(spikes_raster.shape[0]):
    plt.plot(
        np.where(
        spikes_raster[i,:]>0)[0], [i]*len(np.where(spikes_raster[i,:]>0)[0]),'b.')
plt.ylim([0,len(sts)])

#plt.pcolor(spikes.rasterize(), cmap='gray')

plt.title('Raw spikes')






spikes_model = SpikeModel(spikes=spikes)
spikes_model.fit()  # note: this fits a single network to all trials
spikes_model.chomp()
converged_spikes = spikes_model.hopfield_spikes


#plt.figure()
#plt.title('Converge dynamics on Raw data')
#plt.pcolor(converged_spikes.rasterize(), cmap='gray')

converged_raster = converged_spikes.rasterize()
plt.figure()
for i in range(converged_raster.shape[0]):
    plt.plot(
        np.where(converged_raster[i,:]>0)[0], [i]*len(np.where(converged_raster[i,:]>0)[0]),'b.')
plt.title('Converge dynamics on Raw data')
plt.ylim([0,len(sts)])
