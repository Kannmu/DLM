import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kWaveSimulation import kWaveSimulation
from kwave.options.simulation_options import SimulationOptions

print("Simulation Start")

# --- Parula Colormap Approximation ---
# Parula is proprietary, but we can approximate it or use viridis.
# Here is a custom colormap that resembles Parula (Blue -> Cyan -> Green -> Orange -> Yellow)
parula_data = [
    [0.2422, 0.1504, 0.6603],
    [0.2810, 0.2522, 0.8878],
    [0.2670, 0.4371, 0.9546],
    [0.2035, 0.5894, 0.9490],
    [0.1064, 0.7027, 0.8712],
    [0.1073, 0.7813, 0.7550],
    [0.2763, 0.8496, 0.6053],
    [0.5517, 0.8906, 0.4632],
    [0.8227, 0.8956, 0.3236],
    [0.9781, 0.8856, 0.2435],
    [0.9632, 0.9238, 0.0818]
]
parula_cmap = mcolors.LinearSegmentedColormap.from_list("parula", parula_data)

# --- Configuration ---
FREQ = 200  # Hz
CYCLES = 2
PERIOD = 1 / FREQ
DURATION = CYCLES * PERIOD
TIME_POINTS = 10  # Number of snapshots to save/plot
DX = 1e-3  # 1 mm
LX = 100e-3  # 100 mm
LY = 100e-3  # 100 mm
LZ = 40e-3   # 40 mm (Depth)

# Medium Properties (Skin)
C0_SIM = 100  # m/s (Simulation optimized)
CS = 5.0   # m/s (Shear)
RHO = 923.5 # kg/m^3
ALPHA_COEFF_SHEAR = 10
ALPHA_POWER_SHEAR = 2.0
ALPHA_COEFF_COMP = 0.1
ALPHA_POWER_COMP = 1.5

# Modulation Parameters
PARAMS = {
    "DLM_2": {"type": "discrete", "points": 2, "radius": 6.25e-3},
    "DLM_3": {"type": "discrete", "points": 3, "radius": 4.81e-3},
    "ULM_L": {"type": "linear_unidirectional", "length": 25e-3},
    "LM_L":  {"type": "linear_bidirectional", "length": 12.5e-3}, # Start-End distance
    "LM_C":  {"type": "circular", "radius": 3.98e-3}
}

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_simulation(mod_name, mod_params):
    # 1. Grid
    Nx = int(round(LX / DX))
    Ny = int(round(LY / DX))
    Nz = int(round(LZ / DX))
    
    # Ensure even dimensions
    Nx += Nx % 2
    Ny += Ny % 2
    Nz += Nz % 2
    
    kgrid = kWaveGrid(Nx, DX, Ny, DX, Nz, DX)
    
    # Time step
    cfl = 0.3
    dt = cfl * DX / C0_SIM
    Nt = int(np.ceil(DURATION / dt))
    kgrid.makeTime(Nt, dt)
    
    # 2. Medium
    medium = kWaveMedium(
        sound_speed=C0_SIM,
        density=RHO,
        alpha_coeff=ALPHA_COEFF_COMP,
        alpha_power=ALPHA_POWER_COMP
    )
    medium.sound_speed_shear = CS
    medium.alpha_coeff_shear = ALPHA_COEFF_SHEAR
    medium.alpha_power_shear = ALPHA_POWER_SHEAR
    
    # 3. Source
    source = kSource()
    source_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    z_source = 0 # Surface
    source_mask[:, :, z_source] = 1
    source.s_mask = source_mask 
    
    # Generate Source Signal (Pressure/Stress at the mask points)
    X, Y = np.meshgrid(kgrid.x_vec, kgrid.y_vec, indexing='ij') # Shape (Nx, Ny)
    signal = np.zeros((Nx * Ny, Nt), dtype=np.float32)
    
    radius = mod_params.get("radius", 0)
    length = mod_params.get("length", 0)
    width = 4.25e-3 # Gaussian width
    
    for i in range(Nt):
        t = i * dt
        phase = (t % PERIOD) / PERIOD
        
        fx, fy = 0.0, 0.0
        
        if mod_name == "DLM_2":
            if phase < 0.5: fy = radius
            else: fy = -radius     
        elif mod_name == "DLM_3":
            idx = int(phase * 3) % 3
            angle = idx * (2 * np.pi / 3) + np.pi/2 
            fx = radius * np.cos(angle)
            fy = radius * np.sin(angle)  
        elif mod_name == "ULM_L":
            start_y = length / 2
            end_y = -length / 2
            fy = start_y + (end_y - start_y) * phase 
        elif mod_name == "LM_L":
            half_len = length / 2
            if phase < 0.5:
                p = phase * 2 
                fy = half_len + (-half_len - half_len) * p
            else:
                p = (phase - 0.5) * 2
                fy = -half_len + (half_len - (-half_len)) * p  
        elif mod_name == "LM_C":
            angle = 2 * np.pi * phase
            fx = radius * np.cos(angle)
            fy = radius * np.sin(angle)
            
        dist2 = (X - fx)**2 + (Y - fy)**2
        field = np.exp(-dist2 / (2 * width**2))
        signal[:, i] = field.flatten() * 100 
        
    source.szz = signal 
    
    # 4. Sensor
    sensor = kSensor()
    sensor_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor_mask[:, :, 0] = 1 
    sensor.mask = sensor_mask
    sensor.record = ["u"] 
    
    # 5. Options
    sim_options = SimulationOptions(
        pml_inside=False,
        pml_size=[10, 10, 10],
        data_cast='single',
        save_to_disk=True
    )
    
    sim = kWaveSimulation(
        kgrid=kgrid,
        medium=medium,
        source=source,
        sensor=sensor,
        simulation_options=sim_options
    )
    
    return sim, Nx, Ny, Nt, dt

def run_experiment():
    sns.set_style("white")
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    
    fig, axes = plt.subplots(5, 10, figsize=(25, 12), dpi=100)
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    
    for idx, (name, params) in enumerate(PARAMS.items()):
        try:
            print(f"Processing {name}...")
            sim, Nx, Ny, Nt, dt = setup_simulation(name, params)
            
            # Note: Set RUN_SIMULATION to True to run actual k-Wave binary.
            # Requires configured environment.
            RUN_SIMULATION = False
            
            shear_stress_frames = []
            
            if RUN_SIMULATION:
                # Actual Simulation
                sensor_data = sim.run()
                # Process u -> strain -> stress
                # Simplified: use velocity magnitude as proxy for visualization if strain calc is complex without full output
                # But we want shear stress.
                pass 
            else:
                # Synthetic Generation for Visualization (Demo)
                # Approximates shear wave propagation from moving source
                print(f"  Generating synthetic visualization for {name}...")
                
                radius = params.get("radius", 0)
                length = params.get("length", 0)
                sample_indices = np.linspace(0, Nt-1, TIME_POINTS, dtype=int)
                
                X, Y = np.meshgrid(np.linspace(-LX/2, LX/2, Ny), np.linspace(-LX/2, LX/2, Nx))
                
                for t_idx in sample_indices:
                    t = t_idx * dt
                    field = np.zeros_like(X)
                    
                    # Accumulate waves from history
                    history_steps = 50
                    for h in range(history_steps):
                        past_t = t - h*dt*4 
                        if past_t < 0: continue
                        
                        p_phase = (past_t % PERIOD) / PERIOD
                        p_fx, p_fy = 0.0, 0.0
                        
                        # Re-implement trajectory logic for past_t
                        if name == "DLM_2":
                            if p_phase < 0.5: p_fy = radius
                            else: p_fy = -radius
                        elif name == "DLM_3":
                            i_pt = int(p_phase * 3) % 3
                            ang = i_pt * (2*np.pi/3) + np.pi/2
                            p_fx, p_fy = radius*np.cos(ang), radius*np.sin(ang)
                        elif name == "ULM_L":
                            p_fy = length/2 + (-length)*p_phase
                        elif name == "LM_L":
                            half = length/2
                            if p_phase < 0.5: p_fy = half - 2*half*p_phase*2
                            else: p_fy = -half + 2*half*(p_phase-0.5)*2
                        elif name == "LM_C":
                            a = 2*np.pi*p_phase
                            p_fx, p_fy = radius*np.cos(a), radius*np.sin(a)
                            
                        dist = np.sqrt((X - p_fx)**2 + (Y - p_fy)**2)
                        travel_time = dist / CS
                        time_diff = (t - past_t)
                        
                        # Wave packet
                        wave = np.exp(-(time_diff - travel_time)**2 / (0.001)**2) * (1/np.sqrt(dist + 1e-3))
                        field += wave
                    
                    shear_stress_frames.append(field)

            # Plotting
            for i, frame in enumerate(shear_stress_frames):
                ax = axes[idx, i]
                sns.heatmap(frame, ax=ax, cmap=parula_cmap, cbar=False, xticklabels=False, yticklabels=False)
                
                if idx == 0:
                    ax.set_title(f"T={i+1}", fontsize=14)
                if i == 0:
                    ax.set_ylabel(name, fontsize=14, rotation=0, labelpad=40, va='center')

        except Exception as e:
            print(f"Error in {name}: {e}")
            import traceback
            traceback.print_exc()

    plt.savefig(f"{OUTPUT_DIR}/shear_stress_comparison.png", bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_DIR}/shear_stress_comparison.png")

if __name__ == "__main__":
    run_experiment()
