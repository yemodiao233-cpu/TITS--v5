# solvers/NOMA_VEC_Solver.py
"""
NOMA-enabled multi-F-AP vehicle fog computing partial offloading solver.
Based on the paper's proposed joint algorithm (Algorithm 1 + Algorithm 2).
"""
import numpy as np
import logging
from typing import Dict, Any, List
from scipy.optimize import minimize
from abc import ABC

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the base solver class
from solvers.basesolver import BaseSolver

class NOMA_VEC_Solver(BaseSolver):
    """
    Solver for NOMA-enabled multi-F-AP vehicle fog computing partial offloading.
    Implements the three-step optimization logic: power allocation -> task splitting -> user association.
    """
    def __init__(self, env_config: Dict[str, Any], cfg: Dict[str, Any]):
        """
        Initialize the solver with environment and configuration parameters.
        
        Args:
            env_config: Configuration for the environment
            cfg: Configuration for the solver
        """
        super().__init__(env_config, cfg)
        
        # Store environment configuration
        self.num_vehicles = env_config.get('num_vehicles', 3)
        self.num_servers = env_config.get('num_servers', 2)
        
        # Set default values based on paper settings (Table III)
        self.Delta_t = env_config.get('Delta_t', 1.0)  # Slot time
        self.sigma2 = env_config.get('sigma2', 1e-9)  # Noise power (linear scale)
        self.B = env_config.get('B', 5e6)  # Bandwidth in Hz (5MHz)
        self.alpha = env_config.get('alpha', 4.0)  # Path loss exponent
        self.Pn_dBm = env_config.get('Pn_dBm', 26.0)  # TUE transmit power in dBm
        self.Pn = 10 ** ((self.Pn_dBm - 30) / 10)  # Convert to mW
        self.Pf_dBm = env_config.get('Pf_dBm', -8.0)  # F-AP optical power in dBm
        self.Pf = 10 ** ((self.Pf_dBm - 30) / 10)  # Convert to mW
        self.kappa = env_config.get('kappa', 1e-27)  # Compute constant
        self.Re = env_config.get('Re', 155.52e6)  # Optical fiber data rate in bps
        self.td = env_config.get('td', 2.0)  # Tolerable delay in seconds
        self.Delta_t = env_config.get('Delta_t', 1.0)  # Slot duration
        
        # Device capabilities
        self.delta_TUE = env_config.get('delta_TUE', 1e9)  # TUE computing capacity (Hz)
        self.delta_IUE = env_config.get('delta_IUE', 4e9)  # IUE computing capacity (Hz, 3-5GHz)
        self.delta_FAP = env_config.get('delta_FAP', 1e10)  # F-AP computing capacity (Hz)
        
        # Additional parameters for auxiliary F-APs
        self.Cf_tot = env_config.get('Cf_tot', 1e12)  # Total F-AP computing capacity in cycles per second
        self.fap_coverage = env_config.get('fap_coverage', 100.0)  # F-AP coverage radius in meters
        self.auxiliary_max_distance = env_config.get('auxiliary_max_distance', 500.0)  # Max distance for auxiliary F-APs (STM-1 standard)
        
        # Channel model parameters (论文 Section III-B)
        self.K0 = env_config.get('K0', 1e-3)  # 参考距离1m处的信道增益（-30dB）
        
        # Iteration parameters
        self.max_iter = cfg.get('max_iter', 1000)
        self.tol = cfg.get('tol', 1e-5)
        self.mu = cfg.get('mu', 0.9)
        self.init_xi = cfg.get('init_xi', 1.0)
        
        # 初始化F-AP邻接矩阵（将在solve方法中基于真实距离更新）
        self.fap_adjacency = np.zeros((self.num_servers, self.num_servers), dtype=bool)
        
        # Initialize decision variables
        self.assignment = None
        self.power = None
        self.bandwidth = None
        self.freq = None
        
        logger.info("NOMA_VEC_Solver initialized with parameters:")
        logger.info(f"  Vehicles: {self.num_vehicles}, Servers: {self.num_servers}")
        logger.info(f"  Bandwidth: {self.B/1e6} MHz, Tolerable delay: {self.td}s")
    
    def _calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            pos1: First point coordinates (x, y)
            pos2: Second point coordinates (x, y)
            
        Returns:
            distance: Euclidean distance between the two points
        """
        return np.sqrt(np.sum((pos1 - pos2) ** 2))
    
    def solve(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the resource allocation problem using the three-step optimization approach.
        
        Args:
            system_state: The current system state from the VEC_Environment
            
        Returns:
            A dictionary containing the resource allocation decisions
        """
        try:
            # Extract system state information
            V_set = system_state.get('V_set', list(range(self.num_vehicles)))
            J_set = system_state.get('J_set', list(range(self.num_servers)))
            tasks = system_state.get('tasks', [{'Din': 0.75, 'Cv': 750e3, 'kv': 0} for _ in V_set])
            
            # 计算信道增益矩阵（论文 Section III-B: g_{n,j} = K0 / d_{n,j}^alpha）
            # 获取TUE和F-AP的坐标，如果没有则生成随机坐标
            tue_positions = system_state.get('tue_positions', np.random.rand(self.num_vehicles, 2) * 1000)  # (N, 2) 坐标数组
            fap_positions = system_state.get('fap_positions', np.random.rand(self.num_servers, 2) * 1000)  # (F, 2) 坐标数组
            
            # 计算距离矩阵
            distance_matrix = np.zeros((self.num_vehicles, self.num_servers))
            for n in range(self.num_vehicles):
                for j in range(self.num_servers):
                    distance_matrix[n, j] = self._calculate_distance(tue_positions[n], fap_positions[j])
                    
            # 计算信道增益矩阵，确保距离不小于1m（避免除零错误）
            g = np.zeros((self.num_vehicles, self.num_servers))
            for n in range(self.num_vehicles):
                for j in range(self.num_servers):
                    distance = max(distance_matrix[n, j], 1.0)  # 确保距离至少为1m
                    g[n, j] = self.K0 / (distance ** self.alpha)
            
            # 更新F-AP邻接矩阵（论文 Section III-A: G_f̂ₙ = {f ∈ F | d_{f̂ₙ,f} ≤ 500m, f ≠ f̂ₙ}）
            self.fap_adjacency = np.zeros((self.num_servers, self.num_servers), dtype=bool)
            for f1 in range(self.num_servers):
                for f2 in range(self.num_servers):
                    if f1 == f2:
                        continue
                    distance = self._calculate_distance(fap_positions[f1], fap_positions[f2])
                    if distance <= self.auxiliary_max_distance:
                        self.fap_adjacency[f1, f2] = True
            
            # Initialize user association (Step 5.1)
            self.assignment = self._init_assignment()
            
            # Joint optimization loop (Step 5.2)
            energy_prev = float('inf')
            for iteration in range(self.max_iter):
                # Step 2: Optimal power allocation
                theta = self._optimal_power_allocation(tasks, g, self.assignment)
                
                # Step 3: Task splitting optimization (SCA-内点法)
                rho = self._task_splitting_optimization(tasks, g, self.assignment, theta)
                
                # Step 4: User association optimization (联盟博弈)
                self.assignment = self._user_association_optimization(tasks, g, self.assignment, theta, rho)
                
                # Calculate current total energy
                energy_current = self._calculate_total_energy(tasks, g, self.assignment, theta, rho)
                
                # Check convergence (Step 5.2)
                if abs(energy_current - energy_prev) < self.tol:
                    logger.info(f"Converged after {iteration+1} iterations")
                    break
                
                energy_prev = energy_current
                
                if iteration >= self.max_iter - 1:
                    logger.warning(f"Reached maximum iterations ({self.max_iter})")
            
            # Convert optimization results to environment-compatible format
            return self._convert_to_environment_decision(tasks, g, self.assignment, theta, rho)
            
        except Exception as e:
            logger.error(f"Error in NOMA_VEC_Solver: {e}")
            # Fallback to default assignment
            return self._get_default_decision(system_state)
    
    def _init_assignment(self) -> List[int]:
        """
        Initialize user association matrix I.
        Randomly assign TUEs to servers, ensuring each TUE is assigned to exactly one server.
        
        Returns:
            assignment: List of server indices for each TUE
        """
        assignment = np.random.randint(0, self.num_servers, self.num_vehicles)
        return assignment.tolist()
    
    def _optimal_power_allocation(self, tasks: List[Dict[str, Any]], g: np.ndarray, assignment: List[int]) -> np.ndarray:
        """
        Compute optimal power allocation coefficients theta_n* based on Proposition 1.
        
        Args:
            tasks: List of tasks for each TUE
            g: Channel gain matrix
            assignment: Current user association
            
        Returns:
            theta: Optimal power allocation coefficients
        """
        theta = np.zeros(self.num_vehicles)
        
        for n in range(self.num_vehicles):
            # Get assigned server
            j = assignment[n]
            
            # Task parameters
            Dn = tasks[n]['Din'] * 8e6  # Convert from MB to bits
            Cn = tasks[n]['Cv']  # Cycles per bit
            
            # Assume initial un value (will be optimized in task splitting)
            un = 0.25
            
            # Ensure channel gain is positive for numerical stability
            channel_gain = max(g[n, j], 1e-12)
            
            # Calculate IUE computation delay (Step 2.1)
            TIUE_comp = (un * Dn * Cn) / self.delta_IUE
            
            # Verify un feasibility (Step 2.2)
            SNR = (channel_gain * self.Pn) / self.sigma2
            log_term = np.log2(SNR + 1) if SNR > 1e-12 else 1e-12
            denominator = (Dn / log_term) + (self.B * Cn) / self.delta_IUE
            denominator = max(denominator, 1e-12)  # Ensure denominator is positive
            
            feasible_un = self.B * self.td / denominator
            if un >= feasible_un or un <= 0:
                un = min(feasible_un * 0.9, 0.99)
            
            # Calculate optimal theta_n* using closed-form solution (Step 2.3)
            delay_remaining = self.td - TIUE_comp
            
            # Ensure delay_remaining is positive
            if delay_remaining <= 0:
                delay_remaining = 1e-3  # Set a small positive value
                logger.warning(f"Delay remaining is negative, adjusting to {delay_remaining}")
            
            exponent = un * Dn / (self.B * delay_remaining)
            
            # Ensure exponent is not too large to avoid overflow
            if exponent > 30:
                exponent = 30
                logger.warning(f"Exponent too large, adjusting to {exponent}")
            
            numerator = 2 ** exponent - 1
            denominator = (channel_gain / self.sigma2) * self.Pn
            denominator = max(denominator, 1e-12)  # Ensure denominator is positive
            
            theta_n_star = numerator / denominator
            
            # Ensure theta_n* is within [0, 1] (Step 2.4)
            theta[n] = max(0.0, min(1.0, theta_n_star))
        
        return theta
    
    def get_auxiliary_FAPs(self, vehicle_index: int, assigned_fap: int) -> List[int]:
        """
        Get auxiliary F-APs for a given TUE and its assigned F-AP.
        According to paper Section III-A: G_f̂ₙ consists of F-APs within 500m (STM-1 standard) and connected via fiber.
        
        Args:
            vehicle_index: Index of the TUE
            assigned_fap: Index of the assigned F-AP
            
        Returns:
            auxiliary_faps: List of auxiliary F-AP indices
        """
        auxiliary_faps = []
        
        for f in range(self.num_servers):
            if f == assigned_fap:
                continue  # Skip the assigned F-AP itself
            
            # Check if F-AP f is connected to assigned_fap (within max distance)
            if self.fap_adjacency[assigned_fap, f]:
                auxiliary_faps.append(f)
        
        return auxiliary_faps
    
    def _task_splitting_optimization(self, tasks: List[Dict[str, Any]], g: np.ndarray, assignment: List[int], theta: np.ndarray) -> np.ndarray:
        """
        Optimize task splitting ratio using SCA-内点法 (Algorithm 1).
        Implements the SCA convex approximation for the non-convex term T_{TUE,̂f_n}^{n,trans} as per paper formulas (25)-(31).
        
        Args:
            tasks: List of tasks for each TUE
            g: Channel gain matrix
            assignment: Current user association
            theta: Optimal power allocation coefficients
            
        Returns:
            rho: Optimized task splitting ratios, shape (num_vehicles, 3 + max_aux_faps)
                - 前3列: ln, un, vn
                - 第4列及以后: 每个辅助F-AP的a_{nf}
        """
        # 计算每个TUE的辅助F-AP数量
        max_aux_faps = 0
        auxiliary_faps_list = []
        for n in range(self.num_vehicles):
            aux_faps = self.get_auxiliary_FAPs(n, assignment[n])
            auxiliary_faps_list.append(aux_faps)
            max_aux_faps = max(max_aux_faps, len(aux_faps))
        
        # Initialize task splitting ratios (ln, un, vn, a_{nf}) for each TUE
        rho = np.zeros((self.num_vehicles, 3 + max_aux_faps))  # [ln, un, vn, a_{nf1}, a_{nf2}, ...]
        
        # Calculate optimal ln values according to paper formula 22
        for n in range(self.num_vehicles):
            Dn = tasks[n]['Din'] * 8e6  # MB to bits
            Cn = tasks[n]['Cv']  # cycles per bit
            
            # Calculate ln_opt according to paper formula 22
            ln_opt = (self.td * self.delta_TUE) / (Dn * Cn)
            
            # Clamp ln_opt to [0, 1]
            rho[n, 0] = max(0.0, min(1.0, ln_opt))  # ln = rho[n, 0]
            
            # Initialize un, vn and a_{nf} with remaining capacity
            remaining = 1.0 - rho[n, 0]
            num_aux_faps = len(auxiliary_faps_list[n])
            
            if num_aux_faps == 0:
                # 没有辅助F-AP时，只分配un和vn
                rho[n, 1] = remaining * 0.5  # un = rho[n, 1]
                rho[n, 2] = remaining * 0.5  # vn = rho[n, 2]
            else:
                # 有辅助F-AP时，分配un、vn和a_{nf}
                share_per_component = remaining / (2 + num_aux_faps)  # un, vn + num_aux_faps个a_{nf}
                rho[n, 1] = share_per_component  # un
                rho[n, 2] = share_per_component  # vn
                
                # 分配辅助F-AP的a_{nf}
                for i in range(num_aux_faps):
                    rho[n, 3 + i] = share_per_component
        
        # Iteration parameters for SCA
        prev_energy = float('inf')
        
        # SCA迭代求解（论文Algorithm 1）
        for sca_iter in range(self.max_iter):
            # Store current iteration values for convex approximation
            current_rho = rho.copy()
            
            # Create objective function with SCA convex approximation for current iteration
            def objective(rho_flat):
                rho_reshaped = rho_flat.reshape((self.num_vehicles, 3 + max_aux_faps))
                total_energy = 0.0
                
                for n in range(self.num_vehicles):
                    ln = rho_reshaped[n, 0]
                    un = rho_reshaped[n, 1]
                    vn = rho_reshaped[n, 2]
                    a_nf = rho_reshaped[n, 3:3+len(auxiliary_faps_list[n])]
                    
                    # Ensure non-negative splitting ratios
                    if any(x < 0 for x in [ln, un, vn]) or any(x < 0 for x in a_nf):
                        return float('inf')
                    
                    # Check if sum of splitting ratios is approximately 1 (with small tolerance)
                    sum_ratios = ln + un + vn + np.sum(a_nf)
                    if abs(sum_ratios - 1.0) > 1e-3:
                        return float('inf')
                    
                    j = assignment[n]
                    Dn = tasks[n]['Din'] * 8e6  # MB to bits
                    Cn = tasks[n]['Cv']  # cycles per bit
                    
                    # Calculate TUE computing energy (论文公式10)
                    E_TUE_comp = self.kappa * (self.delta_TUE ** 2) * ln * Dn * Cn
                    
                    # Calculate IUE computing energy (论文公式11)
                    E_IUE_comp = self.kappa * (self.delta_IUE ** 2) * un * Dn * Cn
                    
                    # Calculate transmission energy to F-AP with SCA convex approximation
                    channel_gain = max(g[n, j], 1e-12)  # Ensure channel gain is positive
                    Pn = theta[n] * self.Pn
                    SNR = (Pn * channel_gain) / (self.sigma2 + 1e-12)
                    
                    # Calculate rate with numerical stability
                    rate = self.B * np.log2(SNR + 1) if SNR > 1e-12 else 1e-12
                    
                    # 获取当前迭代的vn值用于凸近似（论文公式25）
                    vn_k = current_rho[n, 2]
                    
                    # 计算非凸项 T_{TUE,̂f_n}^{n,trans} = vn Dn / rate 的凸近似
                    # 论文公式（25）的实现：
                    # T_{TUE,̂f_n}^{n,trans} ≈ vn Dn / rate + (vn_k Dn / rate^2) * (rate - rate_k)
                    # 其中 rate_k 是基于当前迭代点计算的速率
                    rate_k = self.B * np.log2(SNR + 1) if SNR > 1e-12 else 1e-12
                    
                    # 计算凸近似的传输时间（论文公式25的实现）
                    if rate_k > 1e-12:
                        tx_time_convex = (vn * Dn) / rate + (vn_k * Dn / (rate_k ** 2)) * (rate - rate_k)
                    else:
                        tx_time_convex = 1e3  # 数值稳定性处理
                    
                    # Ensure transmission time is not too large
                    tx_time_convex = min(tx_time_convex, 10.0)  # Cap at 10 seconds
                    
                    E_FAP_tx = Pn * tx_time_convex
                    
                    # Calculate Cf_tot for this iteration (论文公式12)
                    # 计算所有TUE对F-AP的总负载贡献
                    Cf_tot = 0.0
                    for m in range(self.num_vehicles):
                        vm = rho_reshaped[m, 2]
                        amf = rho_reshaped[m, 3:3+len(auxiliary_faps_list[m])]
                        Dm = tasks[m]['Din'] * 8e6
                        Cm = tasks[m]['Cv']
                        
                        for f in range(self.num_servers):
                            if f == assignment[m]:
                                Cf_tot += vm * Dm * Cm
                            elif f in auxiliary_faps_list[m]:
                                f_index = auxiliary_faps_list[m].index(f)
                                Cf_tot += amf[f_index] * Dm * Cm
                    
                    # Ensure Cf_tot is positive to avoid division by zero
                    Cf_tot = max(Cf_tot, 1e-12)
                    
                    # Calculate F-AP computing energy (论文公式14)
                    C_fn = vn * Dn * Cn  # Cycles required for F-AP computing
                    E_FAP_comp = self.kappa * (C_fn / Cf_tot) * (self.delta_FAP ** 2) * C_fn
                    
                    # Calculate optical transmission energy for auxiliary F-APs
                    E_optical = 0.0
                    for i, f in enumerate(auxiliary_faps_list[n]):
                        if a_nf[i] > 1e-12:
                            E_optical += self.Pf * (a_nf[i] * Dn) / self.Re
                    
                    # Calculate auxiliary F-APs computing energy (论文公式18)
                    E_auxiliary = 0.0
                    for i, f in enumerate(auxiliary_faps_list[n]):
                        C_fn_aux = a_nf[i] * Dn * Cn  # Cycles required for auxiliary F-AP computing
                        E_auxiliary += self.kappa * (C_fn_aux / Cf_tot) * (self.delta_FAP ** 2) * C_fn_aux
                    
                    # 总能耗的凸近似目标函数（论文公式31）
                    total_energy += E_TUE_comp + E_IUE_comp + E_FAP_tx + E_FAP_comp + E_optical + E_auxiliary
                
                return total_energy
            
            # Initial guess
            initial_guess = rho.flatten()
            
            # Define constraints
            constraints = []
            
            # 延迟约束验证（论文公式19d-19f）
            for n in range(self.num_vehicles):
                num_aux = len(auxiliary_faps_list[n])
                
                # 约束1: 分割比总和为1
                constraints.append({
                    'type': 'eq',
                    'fun': lambda x, n=n, num_aux=num_aux: 1.0 - sum(x[n*(3+max_aux_faps):n*(3+max_aux_faps)+3+num_aux])
                })
                
                # 约束2: TUE本地计算延迟约束（论文公式19d）
                def tue_delay_constraint(x, n=n):
                    rho_reshaped = x.reshape((self.num_vehicles, 3 + max_aux_faps))
                    ln = rho_reshaped[n, 0]
                    Dn = tasks[n]['Din'] * 8e6
                    Cn = tasks[n]['Cv']
                    return self.td - (ln * Dn * Cn / self.delta_TUE)
                
                constraints.append({
                    'type': 'ineq',
                    'fun': tue_delay_constraint
                })
                
                # 约束3: IUE子任务延迟约束（论文公式19e）
                def iue_delay_constraint(x, n=n):
                    rho_reshaped = x.reshape((self.num_vehicles, 3 + max_aux_faps))
                    un = rho_reshaped[n, 1]
                    Dn = tasks[n]['Din'] * 8e6
                    Cn = tasks[n]['Cv']
                    # IUE计算延迟 (忽略传输延迟，因为是预先配对的)
                    return self.td - (un * Dn * Cn / self.delta_IUE)
                
                constraints.append({
                    'type': 'ineq',
                    'fun': iue_delay_constraint
                })
                
                # 约束4: F-AP子任务延迟约束（论文公式19f）
                def fap_delay_constraint(x, n=n):
                    rho_reshaped = x.reshape((self.num_vehicles, 3 + max_aux_faps))
                    vn = rho_reshaped[n, 2]
                    a_nf = rho_reshaped[n, 3:3+len(auxiliary_faps_list[n])]
                    
                    j = assignment[n]
                    Dn = tasks[n]['Din'] * 8e6
                    Cn = tasks[n]['Cv']
                    
                    # 计算F-AP传输延迟（使用当前迭代点的凸近似）
                    channel_gain = max(g[n, j], 1e-12)
                    Pn = theta[n] * self.Pn
                    SNR = (Pn * channel_gain) / (self.sigma2 + 1e-12)
                    rate = self.B * np.log2(SNR + 1) if SNR > 1e-12 else 1e-12
                    
                    vn_k = current_rho[n, 2]
                    rate_k = self.B * np.log2(SNR + 1) if SNR > 1e-12 else 1e-12
                    
                    if rate_k > 1e-12:
                        tx_time_convex = (vn * Dn) / rate + (vn_k * Dn / (rate_k ** 2)) * (rate - rate_k)
                    else:
                        tx_time_convex = 1e3
                    
                    # 计算F-AP计算延迟（需要先计算Cf_tot）
                    Cf_tot = 0.0
                    for m in range(self.num_vehicles):
                        vm = rho_reshaped[m, 2]
                        amf = rho_reshaped[m, 3:3+len(auxiliary_faps_list[m])]
                        Dm = tasks[m]['Din'] * 8e6
                        Cm = tasks[m]['Cv']
                        
                        for f in range(self.num_servers):
                            if f == assignment[m]:
                                Cf_tot += vm * Dm * Cm
                            elif f in auxiliary_faps_list[m]:
                                f_index = auxiliary_faps_list[m].index(f)
                                Cf_tot += amf[f_index] * Dm * Cm
                    
                    Cf_tot = max(Cf_tot, 1e-12)
                    compute_time = (vn * Dn * Cn) / self.delta_FAP  # 论文公式13
                    
                    # 计算光传输延迟（如果有辅助F-AP）
                    optical_delay = 0.0
                    if len(auxiliary_faps_list[n]) > 0:
                        # 选择第一个辅助F-AP计算光传输延迟（简化）
                        optical_delay = (a_nf[0] * Dn) / self.Re  # 论文公式6
                    
                    total_fap_delay = tx_time_convex + (1 - (vn > 0)) * optical_delay + compute_time
                    
                    return self.td - total_fap_delay
                
                constraints.append({
                    'type': 'ineq',
                    'fun': fap_delay_constraint
                })
            
            # Bounds for each splitting ratio (0 <= rho <= 1)
            bounds = [(0.0, 1.0) for _ in range((3 + max_aux_faps) * self.num_vehicles)]
            
            # Solve optimization problem
            try:
                result = minimize(
                    objective,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 100, 'tol': 1e-6}
                )
                
                if result.success:
                    rho = result.x.reshape((self.num_vehicles, 3 + max_aux_faps))
                else:
                    logger.warning(f"SCA optimization failed: {result.message}")
                    break
            except Exception as e:
                logger.error(f"Error in SCA optimization: {e}")
                break
            
            # Calculate current energy
            current_energy = objective(result.x)
            
            # Check convergence
            if abs(current_energy - prev_energy) < self.tol:
                logger.info(f"SCA converged after {sca_iter+1} iterations")
                break
            
            prev_energy = current_energy
        
        return rho
    
    def _user_association_optimization(self, tasks: List[Dict[str, Any]], g: np.ndarray, assignment: List[int], theta: np.ndarray, rho: np.ndarray) -> List[int]:
        """
        Optimize user association using coalition game (Algorithm 2).
        Strictly verifies Nash stability: no user can reduce their individual energy by unilaterally switching F-APs.
        
        Args:
            tasks: List of tasks for each TUE
            g: Channel gain matrix
            assignment: Current user association
            theta: Optimal power allocation coefficients
            rho: Optimized task splitting ratios (shape: [num_vehicles, 3 + max_aux_faps])
            
        Returns:
            new_assignment: Optimized user association (Nash stable)
        """
        new_assignment = assignment.copy()
        
        # Step 1: Optimize assignments using greedy algorithm
        improved = True
        iterations = 0
        max_game_iter = self.num_vehicles * self.num_servers
        
        while improved and iterations < max_game_iter:
            improved = False
            
            for n in range(self.num_vehicles):
                current_server = new_assignment[n]
                current_energy = self._calculate_total_energy(tasks, g, new_assignment, theta, rho)
                
                # Try switching to all other servers
                for j in range(self.num_servers):
                    if j == current_server:
                        continue
                    
                    # Check coverage constraint (论文公式19j): TUE n must be within F-AP j's coverage
                    if g[n, j] < 1e-12:  # Threshold for coverage
                        continue  # Skip this F-AP, TUE not in coverage
                    
                    # Create a temporary assignment
                    temp_assignment = new_assignment.copy()
                    temp_assignment[n] = j
                    
                    # Calculate energy with temporary assignment
                    temp_energy = self._calculate_total_energy(tasks, g, temp_assignment, theta, rho)
                    
                    # If energy is reduced, update assignment
                    if temp_energy < current_energy - self.tol:
                        new_assignment = temp_assignment.copy()
                        current_energy = temp_energy
                        improved = True
            
            iterations += 1
        
        # Step 2: Strictly verify Nash stability
        nash_stable = False
        verification_iter = 0
        max_verification_iter = 5  # Maximum iterations for Nash stability verification
        
        while not nash_stable and verification_iter < max_verification_iter:
            nash_stable = True
            
            # Check each TUE for Nash stability
            for n in range(self.num_vehicles):
                current_server = new_assignment[n]
                
                # Calculate current individual energy for TUE n
                # Create a temporary assignment with only TUE n in the system
                single_tue_assignment = [-1] * self.num_vehicles
                single_tue_assignment[n] = current_server
                
                # Calculate energy when only TUE n is assigned to current_server
                current_individual_energy = self._calculate_individual_energy(n, tasks, g, single_tue_assignment, theta, rho)
                
                # Try switching to all other servers
                for j in range(self.num_servers):
                    if j == current_server:
                        continue
                    
                    # Check coverage constraint
                    if g[n, j] < 1e-12:
                        continue
                    
                    # Create a temporary assignment with only TUE n in the system
                    temp_single_tue_assignment = [-1] * self.num_vehicles
                    temp_single_tue_assignment[n] = j
                    
                    # Calculate individual energy with temporary assignment
                    temp_individual_energy = self._calculate_individual_energy(n, tasks, g, temp_single_tue_assignment, theta, rho)
                    
                    # If individual energy is reduced, the assignment is not Nash stable
                    if temp_individual_energy < current_individual_energy - self.tol:
                        logger.info(f"Nash stability violated: TUE {n} can reduce energy from {current_individual_energy} to {temp_individual_energy} by switching to F-AP {j}")
                        nash_stable = False
                        
                        # Update the assignment and re-optimize
                        new_assignment[n] = j
                        
                        # Re-optimize power allocation and task splitting with new assignment
                        theta = self._optimal_power_allocation(tasks, g, new_assignment)
                        rho = self._task_splitting_optimization(tasks, g, new_assignment, theta)
                        
                        # Break out of the loop and re-check all TUEs
                        break
                
                if not nash_stable:
                    break
            
            verification_iter += 1
        
        if nash_stable:
            logger.info("Nash stable assignment achieved")
        else:
            logger.warning("Nash stability verification reached maximum iterations")
        
        return new_assignment
    
    def _calculate_individual_energy(self, n: int, tasks: List[Dict[str, Any]], g: np.ndarray, assignment: List[int], theta: np.ndarray, rho: np.ndarray) -> float:
        """
        Calculate the individual energy consumption for a specific TUE.
        
        Args:
            n: TUE index
            tasks: List of tasks for each TUE
            g: Channel gain matrix
            assignment: User association
            theta: Power allocation coefficients
            rho: Task splitting ratios (shape: [num_vehicles, 3 + max_aux_faps])
            
        Returns:
            individual_energy: Energy consumption for TUE n
        """
        if assignment[n] == -1:
            return 0.0  # TUE not assigned to any F-AP
            
        ln = rho[n, 0]
        un = rho[n, 1]
        vn = rho[n, 2]
        
        # Get auxiliary F-APs for this TUE and its assignment
        aux_faps = self.get_auxiliary_FAPs(n, assignment[n])
        num_aux = len(aux_faps)
        a_nf = rho[n, 3:3+num_aux]
        
        j = assignment[n]
        Dn = tasks[n]['Din'] * 8e6  # MB to bits
        Cn = tasks[n]['Cv']  # cycles per bit
        
        # Calculate TUE computing energy (论文公式10)
        E_TUE_comp = self.kappa * (self.delta_TUE ** 2) * ln * Dn * Cn
        
        # Calculate IUE computing energy (论文公式11)
        E_IUE_comp = self.kappa * (self.delta_IUE ** 2) * un * Dn * Cn
        
        # Calculate transmission energy to F-AP
        channel_gain = max(g[n, j], 1e-12)  # Ensure channel gain is positive
        Pn = theta[n] * self.Pn
        SNR = (Pn * channel_gain) / (self.sigma2 + 1e-12)
        rate = self.B * np.log2(SNR + 1) if SNR > 1e-12 else 1e-12
        tx_time = (vn * Dn) / rate if rate > 1e-12 else 1e3
        tx_time = min(tx_time, 10.0)  # Cap at 10 seconds
        E_FAP_tx = Pn * tx_time
        
        # Calculate Cf_tot for individual energy calculation (论文公式12)
        # For individual energy, we consider only the tasks assigned to the current TUE
        # This is because each TUE's energy should only depend on its own load contribution
        Cf_tot_individual = vn * Dn * Cn  # Assigned F-AP cycles from this TUE
        for i, f in enumerate(aux_faps):
            C_fn_aux = a_nf[i] * Dn * Cn  # Auxiliary F-AP cycles from this TUE
            Cf_tot_individual += C_fn_aux
        
        # Ensure Cf_tot is positive to avoid division by zero
        Cf_tot_individual = max(Cf_tot_individual, 1e-12)
        
        # Calculate F-AP computing energy (论文公式14)
        C_fn = vn * Dn * Cn  # Cycles required for F-AP computing
        E_FAP_comp = self.kappa * (C_fn / Cf_tot_individual) * (self.delta_FAP ** 2) * C_fn
        
        # Calculate optical transmission energy for auxiliary F-APs (论文公式16)
        E_optical = 0.0
        for i, f in enumerate(aux_faps):
            if a_nf[i] > 1e-12:
                E_optical += self.Pf * (a_nf[i] * Dn) / self.Re
        
        # Calculate auxiliary F-APs computing energy (论文公式18)
        E_auxiliary = 0.0
        for i, f in enumerate(aux_faps):
            C_fn_aux = a_nf[i] * Dn * Cn  # Cycles required for auxiliary F-AP computing
            E_auxiliary += self.kappa * (C_fn_aux / Cf_tot_individual) * (self.delta_FAP ** 2) * C_fn_aux
        
        individual_energy = E_TUE_comp + E_IUE_comp + E_FAP_tx + E_FAP_comp + E_optical + E_auxiliary
        
        return individual_energy
    
    def _calculate_total_energy(self, tasks: List[Dict[str, Any]], g: np.ndarray, assignment: List[int], theta: np.ndarray, rho: np.ndarray) -> float:
        """
        Calculate the total system energy consumption according to paper formulas.
        
        Args:
            tasks: List of tasks for each TUE
            g: Channel gain matrix
            assignment: Current user association
            theta: Power allocation coefficients
            rho: Task splitting ratios (shape: [num_vehicles, 3 + max_aux_faps])
            
        Returns:
            total_energy: Total system energy consumption
        """
        # First, calculate Cf_tot (论文公式12): C_f^{tot} = Σ_{n ∈ N} [I_f^n v_n + Σ_{f' ∈ G_f} I_{f'}^n a_{n f'}] D_n C_n
        Cf_tot = 0.0
        for n in range(self.num_vehicles):
            ln = rho[n, 0]
            un = rho[n, 1]
            vn = rho[n, 2]
            
            # Get auxiliary F-APs for this TUE and its assignment
            aux_faps = self.get_auxiliary_FAPs(n, assignment[n])
            num_aux = len(aux_faps)
            a_nf = rho[n, 3:3+num_aux]
            
            Dn = tasks[n]['Din'] * 8e6  # MB to bits
            Cn = tasks[n]['Cv']  # cycles per bit
            
            # For each F-AP, check if it's assigned to this TUE or is an auxiliary F-AP
            for f in range(self.num_servers):
                if f == assignment[n]:
                    # This is the assigned F-AP, add vn Dn Cn
                    Cf_tot += vn * Dn * Cn
                elif f in aux_faps:
                    # This is an auxiliary F-AP, find its a_{nf} and add it
                    f_index = aux_faps.index(f)
                    Cf_tot += a_nf[f_index] * Dn * Cn
        
        # Ensure Cf_tot is positive to avoid division by zero
        Cf_tot = max(Cf_tot, 1e-12)
        
        total_energy = 0.0
        
        for n in range(self.num_vehicles):
            ln = rho[n, 0]
            un = rho[n, 1]
            vn = rho[n, 2]
            
            # Get auxiliary F-APs for this TUE and its assignment
            aux_faps = self.get_auxiliary_FAPs(n, assignment[n])
            num_aux = len(aux_faps)
            a_nf = rho[n, 3:3+num_aux]
            
            # Ensure non-negative splitting ratios
            if any(x < 0 for x in [ln, un, vn]) or any(x < 0 for x in a_nf):
                return float('inf')
            
            # Check if sum of splitting ratios is approximately 1 (with small tolerance)
            sum_ratios = ln + un + vn + np.sum(a_nf)
            if abs(sum_ratios - 1.0) > 1e-3:
                return float('inf')
            
            j = assignment[n]
            Dn = tasks[n]['Din'] * 8e6  # MB to bits
            Cn = tasks[n]['Cv']  # cycles per bit
            
            # Calculate TUE computing energy (论文公式10): E_TUE^n,comp = κ(δ_TUE^n)^2 l_n D_n C_n
            E_TUE_comp = self.kappa * (self.delta_TUE ** 2) * ln * Dn * Cn
            
            # Calculate IUE computing energy (论文公式11): E_IUE^n,comp = κ(δ_IUE^n)^2 u_n D_n C_n
            E_IUE_comp = self.kappa * (self.delta_IUE ** 2) * un * Dn * Cn
            
            # Calculate transmission energy to F-AP
            channel_gain = max(g[n, j], 1e-12)  # Ensure channel gain is positive
            Pn = theta[n] * self.Pn
            SNR = (Pn * channel_gain) / (self.sigma2 + 1e-12)
            rate = self.B * np.log2(SNR + 1) if SNR > 1e-12 else 1e-12
            tx_time = (vn * Dn) / rate if rate > 1e-12 else 1e3
            tx_time = min(tx_time, 10.0)  # Cap at 10 seconds
            E_FAP_tx = Pn * tx_time
            
            # Calculate F-AP computing energy (论文公式14): E_FAP^n,comp = κ(C_fn / C_f,tot)(δ_FAP)^2 C_fn
            C_fn = vn * Dn * Cn  # Cycles required for F-AP computing
            E_FAP_comp = self.kappa * (C_fn / Cf_tot) * (self.delta_FAP ** 2) * C_fn
            
            # Calculate optical transmission energy for auxiliary F-APs (论文公式16)
            E_optical = 0.0
            for i, f in enumerate(aux_faps):
                if a_nf[i] > 1e-12:
                    E_optical += self.Pf * (a_nf[i] * Dn) / self.Re
            
            # Calculate auxiliary F-APs computing energy (论文公式18): E_{FAP,f}^n,comp = κ(C_{f,n} / C_f,tot)(δ_FAP)^2 C_{f,n}
            E_auxiliary = 0.0
            for i, f in enumerate(aux_faps):
                C_fn_aux = a_nf[i] * Dn * Cn  # Cycles required for auxiliary F-AP computing
                E_auxiliary += self.kappa * (C_fn_aux / Cf_tot) * (self.delta_FAP ** 2) * C_fn_aux
            
            total_energy += E_TUE_comp + E_IUE_comp + E_FAP_tx + E_FAP_comp + E_optical + E_auxiliary
        
        return total_energy
    
    def _convert_to_environment_decision(self, tasks: List[Dict[str, Any]], g: np.ndarray, assignment: List[int], theta: np.ndarray, rho: np.ndarray) -> Dict[str, Any]:
        """
        Convert the optimization results to the format expected by the environment.
        
        Args:
            tasks: List of tasks for each TUE
            g: Channel gain matrix
            assignment: User association
            theta: Power allocation coefficients
            rho: Task splitting ratios
            
        Returns:
            decision: Dictionary containing the resource allocation decisions
        """
        # Initialize decision variables
        power = np.zeros(self.num_vehicles)
        bandwidth = np.zeros((self.num_vehicles, self.num_servers))
        freq = np.zeros((self.num_vehicles, self.num_servers))
        
        # Fill in the decision variables
        for n in range(self.num_vehicles):
            j = assignment[n]
            
            # Power allocation
            power[n] = theta[n] * self.Pn
            
            # Bandwidth allocation (only for assigned server)
            bandwidth[n, j] = self.B / 1e6  # Convert to MHz for environment
            
            # CPU frequency allocation
            # Based on task splitting ratio for F-AP computing
            vn = rho[n, 2]
            Dn = tasks[n]['Din'] * 8e6  # MB to bits
            Cn = tasks[n]['Cv']  # cycles per bit
            
            # Calculate required frequency for this TUE's task at the server
            required_cycles = vn * Dn * Cn
            freq[n, j] = required_cycles / self.Delta_t  # Cycles per second (Hz)
            
            # Ensure minimum frequency to avoid division by zero
            if freq[n, j] < 1e6:
                freq[n, j] = 1e6
        
        # Log decision values for debugging
        logger.debug(f"Assignment: {assignment}")
        logger.debug(f"Power: {power}")
        logger.debug(f"Bandwidth: {bandwidth}")
        logger.debug(f"Frequency: {freq}")
        logger.debug(f"Theta: {theta}")
        logger.debug(f"Rho: {rho}")
        
        return {
            'assignment': assignment,
            'power': power.tolist(),
            'bandwidth': bandwidth.tolist(),
            'freq': freq.tolist(),
            'debug': {
                'theta': theta.tolist(),
                'rho': rho.tolist()
            }
        }
    
    def _get_default_decision(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a default decision when the solver encounters an error.
        
        Args:
            system_state: The current system state from the VEC_Environment
            
        Returns:
            A dictionary containing default resource allocation decisions
        """
        V_set = system_state.get('V_set', list(range(self.num_vehicles)))
        J_set = system_state.get('J_set', list(range(self.num_servers)))
        
        # Simple round-robin assignment
        assignment = [v % len(J_set) for v in range(len(V_set))]
        power = [self.Pn] * len(V_set)
        bandwidth = [[self.B / 1e6] * len(J_set) for _ in V_set]  # MHz
        freq = [[self.delta_IUE] * len(J_set) for _ in V_set]  # Hz
        
        logger.info(f"Using default round-robin assignment: {assignment}")
        return {
            'assignment': assignment,
            'power': power,
            'bandwidth': bandwidth,
            'freq': freq,
            'debug': {
                'using_default': True
            }
        }