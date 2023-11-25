# return a plot showing original curve with fitted curve, return amplitudes and time_diff
# for these events, specify seed and event_id 
# plot the same accumulative gamma distribution but with separating colors 

import numpy as np
import awkward as ak
import pandas as pd
import matplotlib.pyplot as plt
import math

from scipy.stats import gamma
from scipy.stats import norm 
from scipy.integrate import quad
from scipy.optimize import curve_fit 


# shortlist events with significant detector hits 
a = ak.from_parquet("111_photons.parquet")
event_id = []
for i, j in enumerate(a):
    size = len(j["photons"]["t"])
    if size >= 10000:
        event_id.append(i)

len(event_id)



class Simulation():
    def __init__(self, seed_id):
        self.seed_id = seed_id
        self.event = ak.from_parquet(f"{seed_id}_photons.parquet") 
        self.event_id = [] # all NuTauCC events
        self.lep_event_id = [] # NuTauCC events in which tau minus has leptonic decay
        self.had_event_id = [] # NuTauCC events in which tau minus has hadronic deacy


        for i, j in enumerate(self.event):
            self.event_id.append(i)
            type = j["mc_truth"]["final_state_type"]
            if 11 in type or 13 in type: 
                self.lep_event_id.append(i)
            else:
                self.had_event_id.append(i)



class Event(): # add a function that returns fit_params[dom_id]=(amp_ratio, time_diff)
    # a = ak.from_parquet(f"111_photons.parquet") 
    
    def __init__(self, awk, event_id):
        self.a = awk
        self.event_id = event_id
        self.energy = self.a[event_id]["mc_truth"]["initial_state_energy"]
        self.len = len(self.a[event_id]["photons"]["t"]) 
        self.decay_vertex = (
            self.a[event_id]["mc_truth"]["final_state_x"][1],
            self.a[event_id]["mc_truth"]["final_state_y"][1],
            self.a[event_id]["mc_truth"]["final_state_z"][1])
        self.final_state_type = self.a[event_id]["mc_truth"]["final_state_type"]
        self.decay_type = None 

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
            
     

    def decay_channel(self): # tau minus decay mode 
        if 11 in self.final_state_type or 13 in self.final_state_type: 
            self.decay_type = "leptonic decay"
        else:
            self.decay_type = "hadronic decay"
        return self.decay_type
    
    
    def tau_decay_len(self):
        import awkward as ak
        import math
        import numpy as np
        
        self.tau_x = self.a[self.event_id]["mc_truth"]["final_state_x"][0]
        self.tau_y = self.a[self.event_id]["mc_truth"]["final_state_y"][0]
        self.tau_z = self.a[self.event_id]["mc_truth"]["final_state_z"][0]

        self.decay_x = self.a[self.event_id]["mc_truth"]["final_state_x"][1]
        self.decay_y = self.a[self.event_id]["mc_truth"]["final_state_y"][1]
        self.decay_z = self.a[self.event_id]["mc_truth"]["final_state_z"][1]

        self.decay_len = math.sqrt((self.tau_x - self.decay_x)**2 + (self.tau_y - self.decay_y)**2 + (self.tau_z - self.decay_z)**2) # Value Error 
        return self.decay_len
 
    
    def detector_pos(self): # detector[dom_id] = (pos_x, pos_y, pos_z)
        self.detector = {}

        with open('icecube.geo') as geo_in:
            read_lines = geo_in.readlines()
            modules_i = read_lines.index("### Modules ###\n")   
            for line in read_lines[modules_i+1:]:
                pos = []
                id = []
                line = line.strip("\n").split("\t")
                pos = np.array([float(line[0]), float(line[1]), float(line[2])])
                id = (int(line[3]), int(line[4]))
                self.detector[id] = pos 
        return self.detector 
    

    def vertex_times(self):   
        self.dom_interact_t = {}
        self.dom_decay_t = {}

        for i, j in list(self.d.items()):
            self.dom_interact_t[i] = []
            self.dom_decay_t[i] = []

            for x, y in j:      
                if y == 1 or y == 5:  
                    self.dom_interact_t[i].append(x)
                else:
                    self.dom_decay_t[i].append(x) 




class DOM(): # add a method that returns dom_ids of the neighbouring 8 activated doms
    a = ak.from_parquet(f"111_photons.parquet") 
    
    def __init__(self, event_id, dom_id): 
        self.event_id = event_id 
        self.dom_id = dom_id 
        self.event = self.a[event_id]
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

        self.times = [x[0] for x in self.d[dom_id]] 

        self.times_interact = []
        self.times_decay = []
        self.last_idx = max(self.event["photons"]["id_idx"])
        for i, j in self.d[dom_id]:
            if j == 1 or j == self.last_idx:
                self.times_interact.append(i)
            else:
                self.times_decay.append(i) 


    def waveform(self): # approximate each hit with a gamma distribution to produce pseudowaveform
        xs = np.linspace(np.min(self.times), np.max(self.times), 500) 
        #xs = np.arange(min(self.times), max(self.times), 2)
        
        # Use numpy broadcasting to compute gamma values for each t0 in self.times
        gamma_values = gamma.pdf(xs[:, None] - np.asarray(self.times), 150, 0.5)

        # Sum the gamma values along the appropriate axis
        ys = np.sum(gamma_values, axis=1)
        
        max_idx = np.argmax(ys)
        x_at_max_y = xs[max_idx]
        
        self.new_xs = np.linspace(x_at_max_y, np.max(xs), 500)
        # Again, compute the gamma values using numpy broadcasting for new_xs
        new_gamma_values = gamma.pdf(self.new_xs[:, None] - np.asarray(self.times), 150, 0.5)
        self.new_ys = np.sum(new_gamma_values, axis=1)
        
        area = np.trapz(self.new_ys, self.new_xs)
        self.new_ys /= area
        
        
        # gamma_values subplot for interaction vertex 
        self.xs_interact = np.linspace(x_at_max_y, np.max(self.times_interact), 500) 

        # Use numpy broadcasting to compute gamma values for each t0 in self.times_interact
        gamma_values_interact = gamma.pdf(self.xs_interact[:, None] - np.asarray(self.times_interact), 150, 0.5)

        # Sum the gamma values along the appropriate axis 
        self.ys_interact = np.sum(gamma_values_interact, axis=1) 
        
        # be careful with the area, it should not be normalized to one, it should be expressed as a proportion 
        area_interact = np.trapz(self.xs_interact, self.ys_interact)
        self.ys_interact /= area_interact
        ratio_interact = area_interact/area
        self.ys_interact *= ratio_interact


        # gamma_values subplot for decay vertex
        if np.min(self.times_decay) <= x_at_max_y:     
            self.xs_decay = np.linspace(x_at_max_y, np.max(self.times_decay), 500) 
        else:
            self.xs_decay = np.linspace(np.min(self.times_decay), np.max(self.times_decay), 500) 
            
        # Use numpy broadcasting to compute gamma values for each t0 in self.times
        gamma_values_decay = gamma.pdf(self.xs_decay[:, None] - np.asarray(self.times_decay), 150, 0.5)

        # Sum the gamma values along the appropriate axis
        self.ys_decay = np.sum(gamma_values_decay, axis=1)

        area_decay = np.trapz(self.xs_decay, self.ys_decay)
        self.ys_decay /= area_decay
        ratio_decay = area_decay/area 
        self.ys_decay *= ratio_decay 
        
        return self.new_xs, self.new_ys
    


    # this results in formatting error (might need to move it outisde the class)
    def exp_gau_fit(self): # this returns a dictionary param[dom_id] = (amp_ratio, time_diff) 
        def exp_gau(x_data, a1, lambda_param, mu, sigma):

            def exp_dist(x_data, lambda_param):
                return lambda_param * np.exp(-1 * (x_data - min(x_data)) * lambda_param)

            g1 = exp_dist(x_data, lambda_param)
            g2 = norm.pdf(x_data, loc = mu, scale = sigma)

            return a1 * g1 + (1 - a1) * g2

        init_params = [0.7, 1e-2, (min(self.new_xs) + 100), 5] 
        popt, _ = curve_fit(exp_gau, self.new_xs, self.new_ys, p0 = init_params, bounds=([0, 0, min(self.new_xs), 0], [1, 1, min(self.new_xs) + 200, 20]))

        param_list = popt 
        self.amp_ratio = popt[0] / (1 - popt[0]) 
        self.time_diff = popt[-2] - min(self.new_xs)

        return self.amp_ratio, self.time_diff 
    
    # check the amplitude of the waveform at the same time step for the neighbouring 8 doms 
   


def exp_gau_fit(
        event, 
        dom_id, 
        seed, 
        event_id,
        times, 
        new_xs,
        new_ys,
        figname = None,
        save = True
        ):
    
    if figname is None:
        figname = f'DC_curvefit_{seed}_{event_id}_{dom_id}.pdf'

    #gamma_ref = lambda t: gamma.pdf(t, 150, 0.5)
    #f0 = lambda t: sum([gamma_ref(t-t0) for t0 in times])

    def exp_gau(x_data, a1, lambda_param, mu, sigma):

        def exp_dist(x_data, lambda_param):
            return lambda_param * np.exp(-1 * (x_data - min(x_data)) * lambda_param)

        g1 = exp_dist(x_data, lambda_param)
        g2 = norm.pdf(x_data, loc = mu, scale = sigma)

        return a1 * g1 + (1 - a1) * g2


    init_params = [0.7, 1e-2, (min(new_xs) + 100), 5]
    popt, _ = curve_fit(exp_gau, new_xs, new_ys, p0 = init_params, bounds=([0, 0, min(new_xs), 0], [1, 1, min(new_xs) + 200, 20]))

    param_list = popt 
    amplitude_ratio = popt[0] / (1 - popt[0]) 
    time_diff = popt[-2] - min(new_xs)

    plt.plot(new_xs, new_ys)
    plt.plot(new_xs, exp_gau(new_xs, *popt), alpha = 0.5)
    plt.title(f"DOM[{dom_id}]")
    plt.xlabel("time[ns]")
    plt.ylabel("ADC_count (normalized)")
   
    return amplitude_ratio, time_diff 

    # only exp fitting function as the first step
    #init_lambda = 1e-2
    #popt, _ = curve_fit(exp_dist, new_xs, new_ys, p0 = init_lambda)
    #plt.plot(new_xs, new_ys)
    #plt.plot(new_xs, exp_dist(new_xs, popt))


    #f1 = lambda t: sum([gamma_ref(t-t0) for t0 in interaction_time])
    #f2= lambda t: sum([gamma_ref(t-t0) for t0 in decay_time])
    #x1 = np.linspace(min(interaction_time), max(interaction_time), 1000)
    #x2 = np.linspace(min(decay_time), max(decay_time), 1000) 

    #plt.plot(new_xs, new_ys, alpha = 0.9) # blue
    #plt.plot(x1, f1(x1), alpha = 0.3) # orange
    #plt.plot(x2, f2(x2), alpha = 0.3) # green 

   