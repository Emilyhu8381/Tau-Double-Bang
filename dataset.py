# this script is to select a set of training samples based on the amount of charge within 500ns of the first pulse 

import numpy as np
import awkward as ak
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import math
import time

from scipy.stats import gamma
from scipy.stats import norm 
from scipy.integrate import quad
from scipy.optimize import curve_fit 


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
            

a = ak.from_parquet(f"/n/home06/qhu/tau_sim/output/782_photons.parquet")
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


dom_count = 0
c500 = []
dom_c500 = {}
dom_charge = [] # charge-level doms are those we can call the cumulative_charge method
waveform_charge = {}
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
        if len(d[j]) > 20 and len(d[j]) < 1500: # contrain photon_len before creating instances 
            dom = DOM(awk = a, event_id = i, dom_id = j, d = d)  
            dom_identity = (i,j)
            dom_charge.append(dom_identity)
            # a dictionary of waveforms for all of the charge-level doms
            dom.waveform()
            waveform_charge[dom_identity] = ((dom.xs, dom.ys), dom.domlabel) #to-be-filtered training samples
            dom.cumulative_charge()
            print(f"the c500 value for dom({dom_identity}) is {dom.c500}")
            c500.append(dom.c500)  
            dom_c500[dom_identity] = dom.c500      

# we hope to have finished generating all the relevant dom info at this stage 
print(f"c500 values for charge-level doms: {dom_c500}")
#print(f"the waveforms for each charge-level dom are: {waveform_charge}")
c500_trigger = np.percentile(c500, 20)  
print(f"c500 trigger value based on all charge-level doms are: {c500_trigger}")


for i,j in zip(dom_c500.keys(), dom_c500.values()):
    if j >= c500_trigger:
        waveform_trigger[i] = waveform_charge[i]
        dom_trigger.append(i)


# to count the proportion of each among the triggered doms/waveforms
signal_count = 0
background_count = 0

for info in waveform_trigger.values(): # a dictionary with a being the key, and 
    if info[-1] == 1:
        signal_count += 1 
    else:
        background_count += 1


# the general summary for the events
print(f"total no. of light-level doms is: {dom_count}") 
print(f"total no. of charge-level doms is: {len(waveform_charge)}") 
print(f"total no. of trigger-level doms is: {len(dom_trigger)}")  
trigger_rate = len(dom_trigger) / dom_count
print(f"calculated trigger rate is: {trigger_rate}") 

signal_ratio = signal_count/len(dom_trigger)
background_ratio = background_count/len(dom_trigger)
print(f"the number of signal doms in the triggered doms is: {signal_count}")
print(f"the number of background doms in the triggered doms is: {background_count}")
print(f"the ratio of signals in the triggered doms is: {signal_ratio}")
print(f"the ratio of backgrounds in the triggered doms is: {background_ratio}")

print(f"the event and dom id for each triggered dom is: {dom_trigger}") # dom identifications 
print(f"the waveforms for each trigger-level doms are: {waveform_trigger}")


# save trigger events info into a parquet file 
trigger_events = pa.table(
    {
        "event_id": [i[0] for i in waveform_trigger.keys()],
        "dom_id": [i[1] for i in waveform_trigger.keys()],
        "times": [i[0][0] for i in waveform_trigger.values()],
        "amplitudes": [i[0][1] for i in waveform_trigger.values()],
        "label": [i[1] for i in waveform_trigger.values()]
    }
)

pa.parquet.write_table(trigger_events, "782_trigger_events.parquet")


