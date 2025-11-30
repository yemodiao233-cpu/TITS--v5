import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from solvers.OORAA_Solver import OORAA_Solver

# Test configuration
num_devices = 5
num_servers = 3

# Create test environment configuration
env_config = {
    'num_mec_servers': num_servers,
    'num_devices': num_devices,
    'N_max': 4,  # Maximum devices per MEC
    'tau': 1e-3,  # Time slot length
    'bandwidth': 1e6,  # Total bandwidth (Hz)
    'noise_variance': 10**(-174/10)*1e-3,  # Noise power
    'interference_power': 1e-13,  # Interference power
    'path_loss_coeff': 10**(-40/10),  # Path loss coefficient
    'reference_distance': 1.0,  # Reference distance
    'path_loss_exponent': 4.0,  # Path loss exponent
    'capacitance_coeff': 1e-28,  # Capacitance coefficient
    'comp_intensity': 737.5,  # Computation intensity (cycles/bit)
    'max_cpu_freq': 2.15e9,  # Maximum CPU frequency
    'max_transmit_power': 1.0,  # Maximum transmit power
    'task_min_size': 1e3,  # Minimum task size
    'task_max_size': 2e3   # Maximum task size
}

# Create solver configuration
cfg = {
    'max_iter': 200,
    'bandwidth_tol': 1e-7,
    'alpha_min': 1e-4,
    'perf_preference': 'balanced',
    'init_V': 1e11,
    'V_min': 1e9,
    'V_max': 1e13,
    'adapt_interval': 10
}

# Create test system state
device_positions = np.random.rand(num_devices, 2) * 100  # Random positions in 100x100 area
mec_positions = np.random.rand(num_servers, 2) * 100  # Random positions in 100x100 area
tasks = np.random.uniform(env_config['task_min_size'], env_config['task_max_size'], num_devices)

system_state = {
    'device_positions': device_positions.tolist(),
    'mec_positions': mec_positions.tolist(),
    'tasks': tasks.tolist()
}

# Create the solver
solver = OORAA_Solver(env_config, cfg)

# Test the solver
try:
    print("Testing OORAA_Solver...")
    print("=" * 50)
    
    # Test the solve method
    print("1. Testing solve method...")
    result = solver.solve(system_state)
    print(f"   Solve result keys: {list(result.keys())}")
    print(f"   Assignment: {result['assignment']}")
    print(f"   Power: {[round(p, 4) for p in result['power']]}")
    print(f"   Bandwidth shape: ({len(result['bandwidth'])}, {len(result['bandwidth'][0])})")
    print(f"   Frequency shape: ({len(result['freq'])}, {len(result['freq'][0])})")
    
    # Test default decision
    print("\n2. Testing default decision...")
    default_result = solver._get_default_decision(system_state)
    print(f"   Default assignment: {default_result['assignment']}")
    
    # Test channel gain update
    print("\n3. Testing channel gain update...")
    solver._update_channel_gain(device_positions, mec_positions)
    print(f"   Channel gain matrix shape: {solver.H.shape}")
    
    # Test task partition
    print("\n4. Testing task partition...")
    c_star = solver.subproblem_task_partition(tasks)
    print(f"   Task partition ratios: {[round(c, 4) for c in c_star]}")
    
    # Test local computation
    print("\n5. Testing local computation...")
    f_star = solver.subproblem_local_computation()
    print(f"   Local CPU frequencies: {[round(f/1e9, 4) for f in f_star]} GHz")
    
    print("\n=" * 50)
    print("All tests completed successfully!")
    print("OORAA_Solver is now compatible with the experiment framework.")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()