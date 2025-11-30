import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from solvers.NOMA_VEC_Solver import NOMA_VEC_Solver

# Test configuration
num_vehicles = 3
num_servers = 3

# Create a simple test case
test_tasks = [
    {
        'Din': 2.5,  # MB
        'Cv': 1e3,   # cycles per bit
        'Tmax': 1.0  # seconds
    }
    for _ in range(num_vehicles)
]

# Create TUE and F-AP coordinates
# TUE coordinates
tue_coords = np.array([
    [100, 200],  # TUE 0
    [300, 400],  # TUE 1
    [500, 600]   # TUE 2
])

# F-AP coordinates
fap_coords = np.array([
    [0, 0],      # F-AP 0
    [400, 400],  # F-AP 1
    [800, 800]   # F-AP 2
])

# Initialize the solver with correct parameter format
env_config = {
    'num_vehicles': num_vehicles,
    'num_servers': num_servers,
    'B': 1e6,         # Bandwidth (Hz)
    'sigma2': 1e-9,   # Noise power (W)
    'kappa': 1e-28,   # Capacitance coefficient
    'delta_TUE': 1.0, # TUE computing capacity (Hz)
    'delta_IUE': 0.8, # IUE computing capacity (Hz)
    'delta_FAP': 1.2, # F-AP computing capacity (Hz)
    'Cf_tot': 1e10,   # Total computational capacity of F-APs (cycles/s)
    'Pf_dBm': -8.0,   # Optical transmission power in dBm
    'Re': 1e9,        # Optical transmission rate (bits/s)
    'alpha': 2.8,     # Path loss exponent
    'K0': 1.0,        # Channel parameter
    'Pn_dBm': 26.0,   # Transmit power in dBm
    'tue_coords': tue_coords,
    'fap_coords': fap_coords
}

cfg = {
    'max_iter': 1000,
    'tol': 1e-5,
    'mu': 0.9,
    'init_xi': 1.0
}

# Create the solver
solver = NOMA_VEC_Solver(env_config, cfg)

# Test the solver
try:
    print("Testing NOMA_VEC_Solver with modifications...")
    print("=" * 50)
    
    # Test channel model calculation
    print("1. Testing channel model...")
    # Get distance between TUE 0 and F-AP 1
    distance = solver._calculate_distance(tue_coords[0], fap_coords[1])
    print(f"   Distance between TUE 0 and F-AP 1: {distance:.2f} m")
    
    # Test adjacency matrix
    print("2. Testing adjacency matrix calculation...")
    # Create dummy channel gain matrix
    g = np.zeros((num_vehicles, num_servers))
    for n in range(num_vehicles):
        for j in range(num_servers):
            d = solver._calculate_distance(tue_coords[n], fap_coords[j])
            g[n, j] = env_config['K0'] / (d ** env_config['alpha'])
    
    # Create initial assignment
    initial_assignment = [0, 1, 2]
    
    # Test Nash stable user association
    print("3. Testing user association optimization with Nash stability...")
    # First optimize power
    theta = solver._optimal_power_allocation(test_tasks, g, initial_assignment)
    print(f"   Optimal power allocation: {theta}")
    
    # Then optimize task splitting
    rho = solver._task_splitting_optimization(test_tasks, g, initial_assignment, theta)
    print(f"   Task splitting ratios shape: {rho.shape}")
    print(f"   Task splitting for TUE 0: ln={rho[0,0]:.3f}, un={rho[0,1]:.3f}, vn={rho[0,2]:.3f}")
    if rho.shape[1] > 3:
        print(f"   Auxiliary F-AP splitting ratios for TUE 0: {rho[0,3:]}")
    
    # Now test user association with Nash stability
    optimized_assignment = solver._user_association_optimization(test_tasks, g, initial_assignment, theta, rho)
    print(f"   Initial assignment: {initial_assignment}")
    print(f"   Optimized assignment: {optimized_assignment}")
    
    # Test total energy calculation
    print("4. Testing total energy calculation...")
    total_energy = solver._calculate_total_energy(test_tasks, g, optimized_assignment, theta, rho)
    print(f"   Total energy consumption: {total_energy:.6f} J")
    
    # Test individual energy calculation (new feature)
    print("5. Testing individual energy calculation...")
    for n in range(num_vehicles):
        individual_energy = solver._calculate_individual_energy(n, test_tasks, g, optimized_assignment, theta, rho)
        print(f"   Energy for TUE {n}: {individual_energy:.6f} J")
    
    print("=" * 50)
    print("All tests completed successfully!")
    
except Exception as e:
    print(f"Error during testing: {e}")
    import traceback
    traceback.print_exc()