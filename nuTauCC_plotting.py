# standardized plotting script for nuTauCC events
# hit-level, dom-level dictionaries 
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from find_cog import find_cog


def plot_tau_event(
        event, 
        seed,
        event_id,
        decay_idx,
        had_idx,
        figname = None,
        save = True,
        channel = 'photons',
        cmap = 'jet_r',
        elev_angle = 15,
        azi_angle = 20       
        ):
    
    if figname is None:
        figname = f'nuTauCC_event_display_{seed}_{event_id}.pdf'

    fig = plt.figure(figsize = (48, 30))
    gs = plt.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.07])

    # relevant dom-level info 
    d = {}
    for x, y, z, w in zip(
        event["photons"]["string_id"], 
        event["photons"]["sensor_id"],
        event["photons"]["t"],
        event["photons"]["id_idx"]
    ):
        om_id = (x, y)
        if om_id not in d:
            d[om_id] = []
        d[om_id].append((z, w))

    tau_d = {}
    decay_d = {}
    had_d = {}
    for x, y, z, w in zip(
        event["photons"]["string_id"], 
        event["photons"]["sensor_id"],
        event["photons"]["t"],
        event["photons"]["id_idx"]
    ):
        
        om_id = (x, y)
        
        if w == 1:
            if om_id not in tau_d:
                tau_d[om_id] = []
            tau_d[om_id].append((z, w))


        elif w == decay_idx:
            if om_id not in decay_d:
                decay_d[om_id] = []
            decay_d[om_id].append((z, w))

        elif w == had_idx:
            if om_id not in had_d:
                had_d[om_id] = []
            had_d[om_id].append((z, w))


    # detector info: dom_id - dom_position correspondence
    pos = []
    detector = {}

    with open('icecube.geo') as geo_in:
        read_lines = geo_in.readlines()
        modules_i = read_lines.index("### Modules ###\n")   
        for line in read_lines[modules_i+1:]:
            pos = []
            id = []
            line = line.strip("\n").split("\t")
            pos = np.array([float(line[0]), float(line[1]), float(line[2])])
            id = (int(line[3]), int(line[4]))
            detector[id] = pos


    # re-generate vertex specific dom_level information 
    dom_level = {}
    for x, y in zip(d.keys(), d.values()):
        for k, v in zip(list(detector.keys()), list(detector.values())):
            if x == k:           
                pos = v

        lcharge = len(y)

        hit_time = []
        for i in y:
            hit_time.append(y[0][0])
        color = np.mean(hit_time)

        dom_level[x] = (pos, lcharge, color)

    
    tau_dom_level = {}
    for x, y in zip(tau_d.keys(), tau_d.values()):
        for k, v in zip(list(detector.keys()), list(detector.values())):
            if x == k:           
                pos = v

        lcharge = len(y)

        hit_time = []
        for i in y:
            hit_time.append(y[0][0])
        color = np.mean(hit_time)

        tau_dom_level[x] = (pos, lcharge, color)


    decay_dom_level = {}
    for x, y in zip(decay_d.keys(), decay_d.values()):

        for k, v in zip(list(detector.keys()), list(detector.values())):
            if x == k:           
                pos = v

        lcharge = len(y)

        hit_time = []
        for i in y:
                hit_time.append(y[0][0])
        color = np.mean(hit_time)

        decay_dom_level[x] = (pos, lcharge, color)


    had_dom_level = {}
    for x, y in zip(had_d.keys(), had_d.values()):

        for k, v in zip(list(detector.keys()), list(detector.values())):
            if x == k:           
                pos = v

        lcharge = len(y)

        hit_time = []
        for i in y:
                hit_time.append(y[0][0])
        color = np.mean(hit_time)

        had_dom_level[x] = (pos, lcharge, color)


    # general event visualization
    ax1 = fig.add_subplot(gs[0, :-1], projection='3d')

    dom_x = [x[0][0] for x in list(dom_level.values())]
    dom_y = [y[0][1] for y in list(dom_level.values())]
    dom_z = [z[0][2] for z in list(dom_level.values())]
    dom_lcharge = [30 + np.log(c[1])*90 for c in list(dom_level.values())]
    dom_color = [c[2] for c in list(dom_level.values())] 

    times = event["photons"]["t"]
    norm = plt.Normalize(vmin=min(times), vmax=max(times))

    scat_d = ax1.scatter(
        dom_x,
        dom_y,
        dom_z,
        c = dom_color,
        cmap = cmap,
        norm = norm,
        alpha = 0.4,
        s = dom_lcharge,
    )

    nutau_x = event["mc_truth"]["initial_state_x"]
    nutau_y = event["mc_truth"]["initial_state_y"]
    nutau_z = event["mc_truth"]["initial_state_z"]

    scat_interaction = ax1.scatter(
        nutau_x,
        nutau_y,
        nutau_z, 
        color = "#1f77b4", 
        alpha = 1,
        marker = 'x',
        s= 80
    ) 

    # plot a track that shows the direction of the incoming Tau neutrino 
    x, y, z = nutau_x, nutau_y, nutau_z
    phi = event["mc_truth"]["initial_state_azimuth"]  
    theta = event["mc_truth"]["initial_state_zenith"] 
    r = 400


    def spherical_to_cartesian(r, theta, phi):
        dx = r * np.sin(theta) * np.cos(phi)
        dy = r * np.sin(theta) * np.sin(phi)
        dz = r * np.cos(theta)

        return dx, dy, dz

    dx, dy, dz = spherical_to_cartesian(r = r, theta = theta, phi = phi)

    # could change 0.8 to a less ad hoc specification
    line_x_in = [x - 0.8 * dx, x]
    line_y_in = [y - 0.8 * dy, y]
    line_z_in = [z - 0.8 * dz, z]
    ax1.plot(line_x_in, line_y_in, line_z_in, color = "#1f77b4", alpha = 0.9)

    line_x_out = [x,  x + 0.8 * dx]
    line_y_out = [y,  y + 0.8 * dy]
    line_z_out = [z,  z + 0.8 * dz]
    ax1.plot(line_x_out, line_y_out, line_z_out, linestyle = ":", color = "#1f77b4", alpha = 0.9)


    ax1.set_xlim(min(event["photons"]["sensor_pos_x"]), max(event["photons"]["sensor_pos_x"]))
    ax1.set_ylim(min(event["photons"]["sensor_pos_y"]), max(event["photons"]["sensor_pos_y"]))
    ax1.set_zlim(min(event["photons"]["sensor_pos_z"]), max(event["photons"]["sensor_pos_z"]))
    ax1.view_init(elev = elev_angle, azim = azi_angle)

   
    nutau_energy = event["mc_truth"]["initial_state_energy"]
    nutau_zenith = event["mc_truth"]["initial_state_zenith"]
    tot_hit = len(times)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1_line1 = "Event visualization (DOM-level)"
    ax1_line2 = r"$ E_{ {\nu}_{\tau}} $" + f"={nutau_energy} GeV; " + r"$ \theta^{zen}_{\nu_\tau}$" +  f"={nutau_zenith}"
    ax1_line3 = r"$ N_{hit}^{tot}$" +  f"= {tot_hit}" 
    ax1.set_title(f"{ax1_line1}\n{ax1_line2}\n{ax1_line3}", fontsize=30)


    # Tau_Minus interaction vertex subplot 
    ax2 = fig.add_subplot(gs[1, 0], projection='3d')

    tau_x = [x[0][0] for x in list(tau_dom_level.values())]
    tau_y = [y[0][1] for y in list(tau_dom_level.values())]
    tau_z = [z[0][2] for z in list(tau_dom_level.values())]
    tau_lcharge = [30 + np.log(c[1])*90 for c in list(tau_dom_level.values())] 
    tau_color = [c[2] for c in list(tau_dom_level.values())]


    scat_tau = ax2.scatter(
        tau_x,
        tau_y,
        tau_z,
        c = tau_color,
        cmap = cmap, 
        norm = norm,
        alpha = 0.4,
        s = tau_lcharge,
    )

    tau_minus_x = event["mc_truth"]["final_state_x"][0]
    tau_minus_y = event["mc_truth"]["final_state_y"][0]
    tau_minus_z = event["mc_truth"]["final_state_z"][0]
    tau_minus_phi = event["mc_truth"]["final_state_azimuth"][0]
    tau_minus_theta = event["mc_truth"]["final_state_zenith"][0]


    scat_tau_vertex = ax2.scatter(
        tau_minus_x,
        tau_minus_y,
        tau_minus_z, 
        color = "#1f77b4",
        alpha=1,
        marker = 'x',
        s= 80
    ) 


    pion_minus_x = event["mc_truth"]["final_state_x"][1]
    pion_minus_y = event["mc_truth"]["final_state_y"][1]
    pion_minus_z = event["mc_truth"]["final_state_z"][1]

    scat_pion_minus_vertex = ax2.scatter(
        pion_minus_x,
        pion_minus_y,
        pion_minus_z, 
        color = "#ff7f0e",
        alpha=1,
        marker = 'x',
        s= 80
    ) 


    # plot a track that shows the direction of the incoming Tau neutrino 
    x, y, z = tau_minus_x, tau_minus_y, tau_minus_z
    phi = tau_minus_phi  
    theta = tau_minus_theta 
    r = 200

    dx, dy, dz = spherical_to_cartesian(r, theta, phi)

    # Plot the propagating line
    line_x_in = [x - 0.8 * dx, x]
    line_y_in = [y - 0.8 * dy, y]
    line_z_in = [z - 0.8 * dz, z]
    ax2.plot(line_x_in, line_y_in, line_z_in, linestyle = ":", color = "#1f77b4", alpha = 0.9)

    line_x_out = [x,  x + 0.8 * dx]
    line_y_out = [y,  y + 0.8 * dy]
    line_z_out = [z,  z + 0.8 * dz]
    ax2.plot(line_x_out, line_y_out, line_z_out, color = "#1f77b4", alpha = 0.9)


    ax2.set_xlim(min(event["photons"]["sensor_pos_x"]), max(event["photons"]["sensor_pos_x"]))
    ax2.set_ylim(min(event["photons"]["sensor_pos_y"]), max(event["photons"]["sensor_pos_y"]))
    ax2.set_zlim(min(event["photons"]["sensor_pos_z"]), max(event["photons"]["sensor_pos_z"]))
    ax2.view_init(elev = elev_angle, azim = azi_angle)


    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_zlabel('Z [m]')
    ax2.set_title("Tau_Minus Vertex Photons (DOM-level)",  fontsize = 30)
    #ax.text2D(0.5, 0.95, "Tau_Minus Vertex Photons (DOM-level)", transform=ax.transAxes, ha="center", fontsize=16)
    #ax.text2D(0.5, 0.91, r"$ E_{\tau^{-} } = 83, 776 GeV $", transform=ax.transAxes, ha="center", fontsize=12)

    tau_energy = event["mc_truth"]["final_state_energy"][0]
    tau_hit = np.sum([c[1] for c in list(tau_dom_level.values())])
    ax2_line1 = "Tau_Minus Vertex Photons (DOM-level)"
    ax2_line2 = r"$ E_{{\tau}^{-}}$" + f" ={tau_energy} GeV;" + r"$ N_{hit}^{{\tau}^{-}}$" + f"= {tau_hit}"
    # line 3: hit-level information (no. of Cherenkov photons emitted from tau minus vertex that actually hitted the doms)
    ax2.set_title(f"{ax2_line1}\n{ax2_line2}", fontsize=30)



    # Decay Vertex (whatever the tau minus decay products are) subplot
    ax3 = fig.add_subplot(gs[1, 1], projection='3d')

    decay_x = [x[0][0] for x in list(decay_dom_level.values())]
    decay_y = [y[0][1] for y in list(decay_dom_level.values())]
    decay_z = [z[0][2] for z in list(decay_dom_level.values())]
    decay_lcharge = [30 + np.log(c[1])*90 for c in list(decay_dom_level.values())] 
    decay_color = [c[2] for c in list(decay_dom_level.values())] 

    scat_decay = ax3.scatter(
        decay_x,
        decay_y,
        decay_z,
        c = decay_color,
        cmap = cmap,
        norm = norm, 
        alpha = 0.4,
        s = decay_lcharge,
    )

    ax3.set_xlim(min(event["photons"]["sensor_pos_x"]), max(event["photons"]["sensor_pos_x"]))
    ax3.set_ylim(min(event["photons"]["sensor_pos_y"]), max(event["photons"]["sensor_pos_y"]))
    ax3.set_zlim(min(event["photons"]["sensor_pos_z"]), max(event["photons"]["sensor_pos_z"]))
    ax3.view_init(elev = elev_angle, azim = azi_angle)


    ax3.set_xlabel('X [m]')
    ax3.set_ylabel('Y [m]')
    ax3.set_zlabel('Z [m]')
    decay_energy = event["mc_truth"]["final_state_energy"][1]
    decay_hit = np.sum([c[1] for c in list(decay_dom_level.values())])
    ax3_line1 = "Decay Vertex Photons (DOM-level)"
    ax3_line2 = r"$ E_{\pi^{-}}$" + f" = {decay_energy} GeV;" + r"$ N_{hit}^{\pi^{-}}$" + f"= {decay_hit}"
    ax3.set_title(f"{ax3_line1}\n{ax3_line2}", fontsize=30)


    # Hadronic Vertex hit level 
    ax4 = fig.add_subplot(gs[1, 2], projection='3d')

    had_x = [x[0][0] for x in list(had_dom_level.values())]
    had_y = [y[0][1] for y in list(had_dom_level.values())]
    had_z = [z[0][2] for z in list(had_dom_level.values())]
    had_lcharge = [30 + np.log(c[1])*90 for c in list(had_dom_level.values())] 
    had_color = [c[2] for c in list(had_dom_level.values())]


    scat_had = ax4.scatter(
        had_x,
        had_y,
        had_z,
        c = had_color,
        cmap = cmap,
        norm = norm,
        alpha = 0.4,
        s = had_lcharge,
    )
    

    ax4.set_xlim(min(event["photons"]["sensor_pos_x"]), max(event["photons"]["sensor_pos_x"]))
    ax4.set_ylim(min(event["photons"]["sensor_pos_y"]), max(event["photons"]["sensor_pos_y"]))
    ax4.set_zlim(min(event["photons"]["sensor_pos_z"]), max(event["photons"]["sensor_pos_z"]))
    ax4.view_init(elev = elev_angle, azim = azi_angle)


    ax4.set_xlabel('X [m]')
    ax4.set_ylabel('Y [m]')
    ax4.set_zlabel('Z [m]')
    had_energy = event["mc_truth"]["final_state_energy"][-1]
    had_hit = np.sum([c[1] for c in list(had_dom_level.values())])
    ax4_line1 = "Hadronic Vertex Photons (DOM-level)"
    ax4_line2 = r"$ E_{had}$" + f" = {had_energy}GeV;" + r"$ N_{hit}^{had}$" + f"= {had_hit}"
    ax4.set_title(f"{ax4_line1}\n{ax4_line2}", fontsize=30)


    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(scat_d, cax = cbar_ax)
    ticks = np.linspace(cbar.vmin, cbar.vmax, 7)
    labels = [f"{int(t)}[ns]" for t in ticks]
    cbar.set_ticks(ticks) 
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=20)


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(f"Event_visualization_{seed}_{event_id}.png", dpi=300) 


