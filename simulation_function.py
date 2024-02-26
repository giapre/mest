#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tvb.simulator.lab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as plc
import time as tm
import mest_function
from utils import *
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Data
data_path = '/Users/gpret/Desktop/MEDICINA/TVB_Distribution/demo_scripts/myScripts/For_OHBM/For_OHBM/tutorial_data'


# In[3]:


def general_setup(fdata_path):
    data_path = fdata_path
    con = connectivity.Connectivity.from_file(f'{data_path}/connectivity.vep.zip')
    con.tract_lengths = np.zeros((con.tract_lengths.shape))             # no time-delays
    con.weights[np.diag_indices(con.weights.shape[0])] = 0 #mette a zero i weights degli indici diagonali, che sono quelli di una
    #area con se stessa
    # con.weights = np.log(con.weights+1)
    con.weights /= con.weights.max() # a /= b equivale a fare a = a/b
    con.configure()
    nb_regions = con.number_of_regions
    roi = con.region_labels
    
    # Coupling
    coupl = coupling.Difference(a=np.array([-0.2]))

    # Integrators
    hiss = noise.Additive(nsig = np.array([0., 0., 0., 0.0003, 0.0003, 0.]))
    heunint = integrators.HeunStochastic(dt=0.05, noise=hiss) 

    # Monitors
    mons = [monitors.TemporalAverage(period=3)]
    
    return con, coupl, heunint, mons


# In[ ]:


def simulate(fEZ, fPZ, fx0, fI, fsetup):
    
    # Here we set up the EZ node such that it is close to the critical working point, i.e. the seizure threshold
    x0ez= fx0
    x0pz= x0ez
    x0num=-2.5

    con = fsetup[0]
    roi = con.region_labels
    nb_regions = con.number_of_regions
    EZ = fEZ
    PZ = fPZ
    idx_EZ = np.where(roi == EZ)
    idx_PZ = [np.where(roi == pz) for pz in PZ]

    epileptors = models.Epileptor(r=np.array([0.00035]))
    epileptors.x0 = x0num*np.ones(nb_regions)
    epileptors.x0[idx_EZ] = x0ez
    for id in idx_PZ:
        epileptors.x0[id] = x0pz
        x0pz -= 0.01

    # Initial conditions
    init_cond = np.array([-1.98742113e+00 , -1.87492138e+01, 4.0529597e+00, -1.05214059e+00, -4.95543740e-20, -1.98742113e-01])
    print(init_cond)
    init_cond_reshaped = np.repeat(init_cond, nb_regions).reshape((1, len(init_cond), nb_regions, 1))

    # Spatial stimulation pattern (via weights accross the network nodes)
    
    dt = 0.05
    onset = 2000 # ms
    stim_length = onset + 200 # stimulation length (including onset) ms
    simulation_length = 4000 # ms
    freq = 50/1000 # frequency converted to 1/ms
    T = 1/freq # pulse repetition period [ms]
    tau = 10 # pulse width [ms]
    I = fI # intensity [mA]

    # Temporal stimulation pattern
    class vector1D(equations.DiscreteEquation):
        equation = equations.Final(default="emp")
    eqn_t = vector1D()
    parameters = {'T': T, 'tau': tau, 'amp': I, 'onset': onset}
    pulse1, _ = equations.PulseTrain(parameters=parameters).get_series_data(max_range=stim_length, step=dt)
    pulse1_ts = [p[1] for p in pulse1]
    parameters = {'T': T, 'tau': tau, 'amp': I, 'onset': onset + tau}
    pulse2, _ = equations.PulseTrain(parameters=parameters).get_series_data(max_range=stim_length, step=dt)
    pulse2_ts = [p[1] for p in pulse2]
    pulse_ts = -np.asarray(pulse1_ts) #- np.asarray(pulse2_ts)
    stimulus_ts = np.hstack((pulse_ts[:-1], np.zeros(int(np.ceil((simulation_length - stim_length) / dt)))))
    eqn_t.parameters['emp'] = np.copy(stimulus_ts)

    print("Stimuli applied from the SEEG electrode is", I)
    
    stim_weight = 3
    stim_weights = np.zeros((nb_regions))
    stim_weights[idx_EZ] = np.array([stim_weight])
    stimulus = patterns.StimuliRegion(temporal=eqn_t,
                                      connectivity=con,
                                      weight=stim_weights)
    stimulus.configure_space()
    stimulus.configure_time(np.arange(0., np.size(stimulus_ts), 1))

    # Simulator
    sim = simulator.Simulator(model=epileptors,
                              stimulus=stimulus, 
                              initial_conditions=init_cond_reshaped,
                              connectivity=fsetup[0],
                              coupling=fsetup[1],
                              integrator=fsetup[2],
                              monitors=fsetup[3])

    sim.configure()

    # Run
    print("Starting simulation...")
    tic = tm.time()
    ttavg = sim.run(simulation_length=simulation_length)
    print("Finished simulation.")
    print('execute for ' + str(tm.time()-tic))
    
    # Take the results
    tts = ttavg[0][0]
    tavg = ttavg[0][1]
    srcSig = tavg[:,0,:,0]
    start_idx = 0
    end_idx = tavg.shape[0]

    srcSig_normal=srcSig/np.ptp(srcSig)
    
    return(srcSig_normal, roi, epileptors)

