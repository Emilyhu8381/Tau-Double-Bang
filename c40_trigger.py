# this script selects trigger events based on the amount of photon deposition in the first 40% of time 

import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

from scipy.stats import gamma
from scipy.stats import norm 
from scipy.integrate import quad
from scipy.optimize import curve_fit 


class Event(): 
    def __init__(self, awk, event_id):
        self.a = awk
        self.event_id = event_id

        # hit times and photon sources at each DOM
        self.d = {}
        for x, y, z, w in zip(
            self.a[event_id]["photons"]["string_id"], 
            self.a[event_id]["photons"]["sensor_id"],
            self.a[event_id]["photons"]["t"],
            self.a[event_id]["photons"]["id_idx"]
        ):
            om_id = (x, y)
            if om_id not in self.d:
                self.d[om_id] = []
            self.d[om_id].append((z, w))
    



class DOM():

    def __init__(self, awk, event_id, dom_id, d): 
        self.a = awk 
        self.event_id = event_id 
        self.dom_id = dom_id 
        self.d = d
        self.times = [x[0] for x in d[dom_id]] 
        # dom labelling (two conditions: interact_count != 0 and < tot count)
        self.ididx = [x[1] for x in d[dom_id]]
        self.last_idx = max(awk[event_id]["photons"]["id_idx"])
        self.ididx = [x[1] for x in d[dom_id]]
        self.interact_count = 0
        for _ in self.ididx:
            if _ == 1 or _ == self.last_idx:
                self.interact_count += 1
            if self.interact_count > 0 and self.interact_count < len(self.ididx):
                self.domlabel = 1
            else:
                self.domlabel = 0
        
  
        self.interact_ratio = self.interact_count/len(self.ididx)
        self.decay_ratio = 1 - self.interact_ratio
        score = abs(self.interact_ratio - self.decay_ratio)
        self.score = 1 - score 


    def waveform(self): # approximate each hit with a gamma distribution to produce pseudowaveform
        #xs = np.linspace(np.min(self.times), np.max(self.times), 500) 
        self.xs = np.arange(min(self.times), max(self.times))
        
        # Use numpy broadcasting to compute gamma values for each t0 in self.times
        gamma_values = gamma.pdf(self.xs[:, None] - np.asarray(self.times), 9, 1)

        # Sum the gamma values along the appropriate axis
        self.ys = np.sum(gamma_values, axis=1) 
        

    def cumulative_charge(self): 
            #self.xs = np.linspace(np.min(self.times), np.max(self.times), 500) 
            self.xs = np.arange(min(self.times), max(self.times))
            gamma_values = gamma.pdf(self.xs[:, None] - np.asarray(self.times), 9, 1)
            self.ys = np.sum(gamma_values, axis=1)

            cum_charge = np.cumsum(self.ys)
            t0 = min(self.times)
            t500 = t0 + 500
            self.c500 = np.interp(t500, self.xs, cum_charge) 
            

a = ak.from_parquet(f"777_photons.parquet")
#a = ak.from_parquet(f"/n/home06/qhu/tau_sim/output/777_photons.parquet")


event_id = [] # all NuTauCC events
lep_event_id = [] # NuTauCC events in which tau minus has leptonic decay
had_event_id = [] 
for i, j in enumerate(a):
    event_id.append(i)
    type = j["mc_truth"]["final_state_type"]
    if 11 in type or 13 in type: 
        lep_event_id.append(i)
    else:
        had_event_id.append(i)

lep_event_count = len(lep_event_id)

dom_count = 0
c40 = []    
dom_c40 = {}
dom_trigger = []
waveform_trigger = {}



for i in lep_event_id:

    d = {} # d is a local variable that updates for every event 
    for x, y, z, w in zip(
        a[i]["photons"]["string_id"],
        a[i]["photons"]["sensor_id"],
        a[i]["photons"]["t"],
        a[i]["photons"]["id_idx"]
    ):
        om_id = (x, y)
        if om_id not in d:
            d[om_id] = []
        d[om_id].append((z,w))
    print(f"the dom dictionary for event{i} was generated")

    domids= d.keys() # local variable for each event
    dom_len = len(domids)
    dom_count += dom_len # for further calculation of percentage of triggered events


    for j in domids:
        if len(d[j]) < 1500: # only put an upper bound to prevent saturation 
            dom = DOM(awk = a, event_id = i, dom_id = j, d = d) 
            dom_identity = (i,j)
            print(f"the current dom is {dom_identity} ")

            tmin = min(dom.times)
            tmax = max(dom.times)
            t_cut = tmin + 0.4 * (tmax - tmin)
            photon_count = len(d[j]) 
            l = [t[0] for t in d[j] if t[0] <= t_cut]
            l = len(l)
            c40.append(l) 
            print(f"the c40 value for dom {dom_identity} is {l}" )
            dom_c40[dom_identity] = c40  

            # only if pass through the trigger criteria 
            #dom.waveform()
            #waveform_trigger[dom_identity] = ((dom.xs, dom.ys), dom.domlabel) 
    


# before the cut, first define a function that looks for p40 trigger value 
# (cannot directly apply numpy quantile in this case because this only applies for unsaturated light-level DOMs, which are already only ~20% of all the DOMs)

# we hope to have finished generating all the relevant dom info at this stage 
#c40_trigger = np.percentile(c40, 60)  # need to know light-rate info to determine percentile 
#print(f"c40 trigger value based on all light-level doms is: {c40_trigger}")


# the general summary for the events
print(f"total no. of light-level doms is: {dom_count}") 
light_rate = dom_count / (lep_event_count * 5160) 
print(f"the light rate is {light_rate}")
#print(f"total no. of trigger-level doms is: {len(dom_trigger)}")  
#trigger_rate = len(dom_trigger) / dom_count
#print(f"calculated trigger rate is: {trigger_rate}") 
#print(f"the event and dom id for each triggered dom is: {dom_trigger}") # dom identifications 






