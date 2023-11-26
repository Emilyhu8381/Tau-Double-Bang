# this script is to find an appropriate c500 trigger value

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
            

a = ak.from_parquet(f"111_photons.parquet")


def trigger_selection(a):

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
        if i < 109:
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
                dom = DOM(awk = a, event_id = i, dom_id = j, d = d) # generate light-level dom instances for event, inherit the event dictinory 
                if len(dom.times) > 70 and len(dom.times) < 960,8541: 
                    dom_identity = (i,j)
                    dom.cumulative_charge()
                    print(f"the c500 value for dom({dom_identity}) is {dom.c500}")
                    c500.append(dom.c500)  
                    dom_c500[dom_identity] = dom.c500 



    # we hope to have finished generating all the relevant dom info at this stage 
    c500_trigger = np.percentile(c500, 60)  
    print(f"c500 trigger value based on all light-level doms is: {c500_trigger}")


    # the general summary for the events
    print(f"total no. of light-level doms is: {dom_count}") 
    print(f"total no. of trigger-level doms is: {len(dom_trigger)}")  
    trigger_rate = len(dom_trigger) / dom_count
    print(f"calculated trigger rate is: {trigger_rate}") 
    print(f"the event and dom id for each triggered dom is: {dom_trigger}") # dom identifications 


    return c500, dom_c500

