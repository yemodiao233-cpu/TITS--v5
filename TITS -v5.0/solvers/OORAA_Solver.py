import numpy as np
import logging
from typing import Dict, Any, List, Set, Tuple
from scipy.optimize import fsolve
from solvers.basesolver import BaseSolver

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OORAA_Solver(BaseSolver):
    """
    复刻论文《Energy_Efficiency_and_Delay_Tradeoff_in_an_MEC-Enabled_Mobile_IoT_Network》的OORAA算法
    新增：控制参数V的自适应选择逻辑，基于EE-时延偏好动态调整
    """
    def __init__(self, env_config: Dict[str, Any], cfg: Dict[str, Any]):
        super().__init__(env_config, cfg)
        
        # -------------------------- 原有参数初始化 --------------------------
        # 系统基础参数（Section III）
        self.M = env_config.get('num_servers', env_config.get('num_mec_servers', 2))  # MEC服务器数量（与环境保持一致）
        self.U = env_config.get('num_vehicles', env_config.get('num_devices', 3))     # 设备数量（与环境保持一致）
        self.N_max = env_config.get('N_max', 4)        # 每个MEC最大服务设备数（约束15c）
        self.tau = env_config.get('tau', 1e-3)         # 时间槽长度（s）
        self.omega = env_config.get('bandwidth', 1e6)  # 总上行带宽（Hz）
        self.sigma_sq = env_config.get('noise_variance', 10**(-174/10)*1e-3)  # 噪声功率
        self.chi = env_config.get('interference_power', 1e-13)  # 干扰功率
        self.g0 = env_config.get('path_loss_coeff', 10**(-40/10))  # 路径损耗系数
        self.d0 = env_config.get('reference_distance', 1.0)  # 参考距离（m）
        self.theta = env_config.get('path_loss_exponent', 4.0)  # 路径损耗指数
        self.kappa = env_config.get('capacitance_coeff', 1e-28)  # 电容系数
        self.L_u = env_config.get('comp_intensity', 737.5)  # 计算强度（cycles/bit）
        self.f_max = env_config.get('max_cpu_freq', 2.15e9)  # 最大CPU频率（Hz）
        self.P_max_tx = env_config.get('max_transmit_power', 1.0)  # 最大发射功率（W）
        self.A_min = env_config.get('task_min_size', 1e3)  # 最小任务大小（bit）
        self.A_max = env_config.get('task_max_size', 2e3)  # 最大任务大小（bit）
        
        # 求解器核心参数（Section V、Theorem 2）
        self.max_iter = cfg.get('max_iter', 200)   # 交替优化最大迭代次数
        self.zeta = cfg.get('bandwidth_tol', 1e-7) # 带宽分配精度阈值
        self.epsilon_m = cfg.get('alpha_min', 1e-4)# 最小带宽分配比例
        self.H = None  # 信道功率增益矩阵
        self.eta_EE = 0.0  # 实时EE参数（公式16）
        self.E_accum = 0.0  # 累积能耗
        self.D_accum = 0.0  # 累积完成任务量
        self.Q_l = np.zeros(self.U)  # 本地任务队列
        self.Q_o = np.zeros(self.U)  # 卸载任务队列

        # -------------------------- 新增：V自适应参数 --------------------------
        # 1. V的基础配置（用户可设置偏好）
        self.perf_preference = cfg.get('perf_preference', 'balanced')  # 性能偏好：'EE_first'/'delay_first'/'balanced'
        self.target_EE = cfg.get('target_EE', 1e-7)  # 目标EE（J/bit），参考论文仿真值
        self.target_delay = cfg.get('target_delay', 0.02)  # 目标时延（s），参考论文Fig.6（20ms）
        self.adapt_interval = cfg.get('adapt_interval', 10)  # V调整间隔（时间槽数）
        
        # 2. V的边界约束（避免极端值）
        self.V_min = cfg.get('V_min', 1e9)   # 最小V（时延优先时下限）
        self.V_max = cfg.get('V_max', 1e13)  # 最大V（EE优先时上限）
        
        # 3. V的调整状态
        self.current_V = cfg.get('init_V', 1e11)  # 初始V值
        self.time_slot_count = 0  # 时间槽计数器（用于触发调整）
        self.perf_history = []    # 性能历史记录（存储近N个时间槽的EE和时延）
        
        logger.info("OORAA_Solver初始化完成，初始V={:.1e}，性能偏好={}".format(
            self.current_V, self.perf_preference))

    # -------------------------- 原有核心方法（保持不变） --------------------------
    def _update_channel_gain(self, device_positions: np.ndarray, mec_positions: np.ndarray) -> None:
        self.H = np.zeros((self.U, self.M))
        for u in range(self.U):
            for m in range(self.M):
                d_um = np.linalg.norm(device_positions[u] - mec_positions[m])
                h_um = np.random.exponential(scale=1.0)
                self.H[u, m] = h_um * self.g0 * (self.d0 / max(d_um, 1e-3)) ** self.theta

    def _update_eta_EE(self) -> None:
        # 初始状态下设置一个非零的小值，避免资源分配计算结果为0
        self.eta_EE = self.E_accum / self.D_accum if self.D_accum > 1e-6 else 1e-10

    def subproblem_task_partition(self, A: np.ndarray) -> np.ndarray:
        c_star = np.zeros(self.U)
        for u in range(self.U):
            Q_l_u = self.Q_l[u]
            Q_o_u = self.Q_o[u]
            A_u = A[u]
            if Q_o_u <= Q_l_u - A_u:
                c_star[u] = 0.0
            elif Q_o_u >= Q_l_u + A_u:
                c_star[u] = 1.0
            else:
                c_star[u] = (Q_o_u + A_u - Q_l_u) / (2 * A_u)
        return c_star

    def subproblem_local_computation(self) -> np.ndarray:
        f_star = np.zeros(self.U)
        for u in range(self.U):
            # 初始状态下队列长度为0，添加一个小值确保非零计算
            numerator = max(self.Q_l[u], 1e-10) + self.current_V * self.eta_EE
            denominator = 3 * self.kappa * self.current_V * self.L_u
            f_opt = np.sqrt(numerator / denominator) if denominator > 1e-30 else 0.0
            f_star[u] = min(f_opt, self.f_max)
        return f_star

    def _calc_transmit_rate(self, alpha_um: float, p_um: float, H_um: float) -> float:
        if alpha_um <= 0:
            return 0.0
        signal_power = H_um * p_um
        noise_interference = self.chi + alpha_um * self.omega * self.sigma_sq
        return alpha_um * self.omega * np.log2(1 + signal_power / noise_interference) if noise_interference > 1e-30 else 0.0

    def subproblem_transmit_power(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        p_star = np.zeros((self.U, self.M))
        for u in range(self.U):
            for m in range(self.M):
                if x[u, m] == 0:
                    p_star[u, m] = 0.0
                    continue
                H_um = self.H[u, m]
                alpha_um = alpha[u, m]
                if H_um < 1e-30 or alpha_um < self.epsilon_m:
                    p_star[u, m] = 0.0
                    continue
                denominator_gamma = self.chi + alpha_um * self.omega * self.sigma_sq
                gamma_u = H_um / denominator_gamma
                # 初始状态下队列长度为0，添加一个小值确保非零计算
                B_u = (max(self.Q_o[u], 1e-10) + self.current_V * self.eta_EE) * alpha_um * self.omega
                if self.current_V >= (B_u * gamma_u) / np.log(2):
                    p_opt = 0.0
                else:
                    p_opt = (B_u / (self.current_V * np.log(2))) - (1 / gamma_u)
                p_star[u, m] = max(0.0, min(p_opt, self.P_max_tx))
        return p_star

    def _dr_dalpha(self, alpha_um: float, p_um: float, H_um: float) -> float:
        if alpha_um <= 0:
            return 0.0
        term1 = np.log2(1 + (H_um * p_um) / (self.chi + alpha_um * self.omega * self.sigma_sq))
        denominator = np.log(2) * (self.chi + alpha_um * self.omega * self.sigma_sq) * \
                      (self.chi + alpha_um * self.omega * self.sigma_sq + H_um * p_um)
        numerator = alpha_um * H_um * p_um * self.omega * self.sigma_sq
        term2 = numerator / denominator
        return self.omega * (term1 - term2)

    def _g_um(self, lambda_m: float, u: int, m: int, p_um: float) -> float:
        def equation(alpha):
            if alpha < self.epsilon_m:
                return 1e10
            dr = self._dr_dalpha(alpha, p_um, self.H[u, m])
            return (self.Q_o[u] + self.current_V * self.eta_EE) * self.tau * dr - lambda_m
        # 使用minimize函数求解带约束的优化问题，因为fsolve不支持bounds
        from scipy.optimize import minimize
        
        result = minimize(
            lambda alpha: abs(equation(alpha)),  # 最小化方程绝对值
            x0=0.5*(self.epsilon_m + 1.0),
            bounds=[(self.epsilon_m, 1.0)],
            method='L-BFGS-B'
        )
        
        return max(self.epsilon_m, min(1.0, result.x[0]))

    def subproblem_bandwidth_allocation(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        alpha_star = np.zeros((self.U, self.M))
        for m in range(self.M):
            associated_users = [u for u in range(self.U) if x[u, m] == 1]
            N_m = len(associated_users)
            if N_m == 0:
                continue
            lambda_L = 1e-10
            lambda_U = 1e10
            for u in associated_users:
                dr_L = self._dr_dalpha(1.0, p[u, m], self.H[u, m])
                lambda_L = min(lambda_L, (self.Q_o[u] + self.current_V * self.eta_EE) * self.tau * dr_L)
                dr_U = self._dr_dalpha(self.epsilon_m, p[u, m], self.H[u, m])
                lambda_U = max(lambda_U, (self.Q_o[u] + self.current_V * self.eta_EE) * self.tau * dr_U)
            for _ in range(self.max_iter):
                lambda_tmp = (lambda_L + lambda_U) / 2
                alpha_sum = 0.0
                alpha_candidate = []
                for u in associated_users:
                    alpha_u = self._g_um(lambda_tmp, u, m, p[u, m])
                    alpha_candidate.append(alpha_u)
                    alpha_sum += alpha_u
                if abs(alpha_sum - 1.0) < self.zeta:
                    break
                elif alpha_sum > 1.0:
                    lambda_L = lambda_tmp
                else:
                    lambda_U = lambda_tmp
            for idx, u in enumerate(associated_users):
                alpha_star[u, m] = alpha_candidate[idx]
        return alpha_star

    def subproblem_user_association(self, p: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        x_star = np.zeros((self.U, self.M), dtype=int)
        S = set((u, m) for u in range(self.U) for m in range(self.M))
        chi = set()
        chi_m = [set() for _ in range(self.M)]
        chi_u = [False for _ in range(self.U)]
        while S:
            max_gain = -1e10
            best_action = None
            for (u, m) in S:
                if chi_u[u] or len(chi_m[m]) >= self.N_max:
                    continue
                r_um = self._calc_transmit_rate(alpha[u, m], p[u, m], self.H[u, m])
                gain = r_um * self.tau
                if gain > max_gain:
                    max_gain = gain
                    best_action = (u, m)
            if best_action is None or max_gain <= 0:
                break
            u_opt, m_opt = best_action
            x_star[u_opt, m_opt] = 1
            chi_m[m_opt].add(u_opt)
            chi_u[u_opt] = True
            to_remove = set((u, m) for (u, m) in S if u == u_opt or m == m_opt)
            S -= to_remove
        return x_star

    def alternating_optimization(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.zeros((self.U, self.M), dtype=int)
        for u in range(self.U):
            m_rand = np.random.randint(0, self.M)
            x[u, m_rand] = 1
        for m in range(self.M):
            users_m = np.where(x[:, m] == 1)[0]
            if len(users_m) > self.N_max:
                x[users_m[self.N_max:], m] = 0
        alpha = np.ones((self.U, self.M)) / self.M
        p = np.ones((self.U, self.M)) * self.P_max_tx / 2
        for iter_k in range(self.max_iter):
            x_prev = x.copy()
            alpha_prev = alpha.copy()
            p_prev = p.copy()
            p = self.subproblem_transmit_power(x, alpha)
            alpha = self.subproblem_bandwidth_allocation(x, p)
            x = self.subproblem_user_association(p, alpha)
            if (np.sum(np.abs(x - x_prev)) < 1e-3 and
                np.max(np.abs(alpha - alpha_prev)) < 1e-3 and
                np.max(np.abs(p - p_prev)) < 1e-3):
                break
        return x, alpha, p

    def _calculate_performance_metrics(self, c_star: np.ndarray, f_star: np.ndarray, 
                                      x_star: np.ndarray, alpha_star: np.ndarray, p_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        D_l = np.zeros(self.U)
        D_o = np.zeros(self.U)
        E_l = np.zeros(self.U)
        E_o = np.zeros(self.U)
        for u in range(self.U):
            D_l[u] = self.tau * f_star[u] / self.L_u
            E_l[u] = self.tau * self.kappa * (f_star[u] ** 3)
            r_um_sum = 0.0
            p_um_sum = 0.0
            for m in range(self.M):
                if x_star[u, m] == 1:
                    r_um = self._calc_transmit_rate(alpha_star[u, m], p_star[u, m], self.H[u, m])
                    r_um_sum += r_um
                    p_um_sum = p_star[u, m]
            D_o[u] = r_um_sum * self.tau
            E_o[u] = p_um_sum * self.tau
        E_total = np.sum(E_l + E_o)
        D_total = np.sum(D_l + D_o)
        self.E_accum += E_total
        self.D_accum += D_total
        return D_l, D_o, E_total, D_total

    def _update_queues(self, D_l: np.ndarray, D_o: np.ndarray, A: np.ndarray, c_star: np.ndarray) -> None:
        for u in range(self.U):
            self.Q_l[u] = max(self.Q_l[u] - D_l[u], 0.0) + c_star[u] * A[u]
            self.Q_o[u] = max(self.Q_o[u] - D_o[u], 0.0) + (1 - c_star[u]) * A[u]

    # -------------------------- 新增：V自适应核心逻辑 --------------------------
    def _adapt_V_based_perf(self) -> None:
        """
        基于性能反馈自适应调整V（核心逻辑）
        理论依据：论文Theorem 2的[O(1/V), O(V)] EE-时延 tradeoff
        - V增大 → EE提升（趋近最优），时延增大
        - V减小 → 时延降低，EE下降
        """
        if len(self.perf_history) < self.adapt_interval:
            return  # 历史数据不足，不调整
        
        # 1. 计算近N个时间槽的平均性能
        avg_EE = np.mean([h['EE'] for h in self.perf_history[-self.adapt_interval:]])
        avg_delay = np.mean([h['delay'] for h in self.perf_history[-self.adapt_interval:]])
        logger.debug("V调整前：平均EE={:.3e} J/bit，平均时延={:.3f} s".format(avg_EE, avg_delay))

        # 2. 根据性能偏好确定调整策略
        V_step = self.current_V * 0.5  # V调整步长（当前值的50%，可自适应）
        if self.perf_preference == 'EE_first':
            # 策略1：EE优先 → 优先满足EE目标，允许时延略超
            if avg_EE > self.target_EE * 1.1:
                # EE优于目标10% → 适当减小V，降低时延（不低于V_min）
                new_V = max(self.current_V - V_step, self.V_min)
            elif avg_EE < self.target_EE * 0.9:
                # EE劣于目标10% → 增大V，提升EE（不超过V_max）
                new_V = min(self.current_V + V_step, self.V_max)
            else:
                # EE达标 → 维持V，观察时延（时延超30%目标时微调）
                new_V = self.current_V if avg_delay <= self.target_delay * 1.3 else max(self.current_V - V_step/2, self.V_min)
        
        elif self.perf_preference == 'delay_first':
            # 策略2：时延优先 → 优先满足时延目标，允许EE略降
            if avg_delay > self.target_delay * 1.1:
                # 时延超目标10% → 减小V，降低时延（不低于V_min）
                new_V = max(self.current_V - V_step, self.V_min)
            elif avg_delay < self.target_delay * 0.9:
                # 时延优于目标10% → 增大V，提升EE（不超过V_max）
                new_V = min(self.current_V + V_step, self.V_max)
            else:
                # 时延达标 → 维持V，观察EE（EE劣于30%目标时微调）
                new_V = self.current_V if avg_EE >= self.target_EE * 0.7 else min(self.current_V + V_step/2, self.V_max)
        
        else:  # balanced（均衡模式）
            # 策略3：均衡 → 同时约束EE和时延，优先调整偏差大的指标
            ee_deviation = abs(avg_EE - self.target_EE) / self.target_EE
            delay_deviation = abs(avg_delay - self.target_delay) / self.target_delay
            
            if ee_deviation > 0.1 or delay_deviation > 0.1:
                if ee_deviation > delay_deviation:
                    # EE偏差更大 → 调整V优化EE
                    new_V = min(self.current_V + V_step, self.V_max) if avg_EE < self.target_EE else max(self.current_V - V_step, self.V_min)
                else:
                    # 时延偏差更大 → 调整V优化时延
                    new_V = max(self.current_V - V_step, self.V_min) if avg_delay > self.target_delay else min(self.current_V + V_step, self.V_max)
            else:
                # 双指标均达标 → 维持V
                new_V = self.current_V

        # 3. 执行V更新（记录日志）
        if abs(new_V - self.current_V) / self.current_V > 0.05:  # 变化超5%才更新
            logger.info("V调整：{} → {}（平均EE={:.3e}→目标{:.3e}，平均时延={:.3f}→目标{:.3f}）".format(
                self.current_V, new_V, avg_EE, self.target_EE, avg_delay, self.target_delay))
            self.current_V = new_V
        else:
            logger.debug("V无需调整（变化量<5%），当前V={:.1e}".format(self.current_V))

    def _record_perf_history(self, current_EE: float, service_delay: float) -> None:
        """记录性能历史，用于V调整"""
        self.perf_history.append({
            'time_slot': self.time_slot_count,
            'EE': current_EE,
            'delay': service_delay,
            'V': self.current_V
        })
        # 保留最近100个时间槽的历史（避免内存占用）
        if len(self.perf_history) > 100:
            self.perf_history.pop(0)

    # -------------------------- 主求解函数（集成V自适应） --------------------------
    def solve(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # 1. 时间槽计数+1
            self.time_slot_count += 1
            
            # 2. 提取系统状态
            device_positions = np.array(system_state.get('device_positions', np.random.rand(self.U, 2)*100))
            mec_positions = np.array(system_state.get('mec_positions', np.random.rand(self.M, 2)*100))
            
            # 处理任务信息：将任务列表转换为计算量数组
            tasks = system_state.get('tasks', [])
            A = np.zeros(self.U)
            for u in range(self.U):
                if u < len(tasks):
                    # 从任务字典中提取计算量（假设任务的计算量信息在'Cv'字段中）
                    A[u] = float(tasks[u].get('Cv', np.random.uniform(self.A_min, self.A_max)))
                else:
                    # 如果没有足够的任务信息，使用随机值
                    A[u] = np.random.uniform(self.A_min, self.A_max)
            
            # 3. 更新信道增益和EE参数
            self._update_channel_gain(device_positions, mec_positions)
            self._update_eta_EE()
            
            # 4. 三大子问题求解（核心算法）
            c_star = self.subproblem_task_partition(A)
            f_star = self.subproblem_local_computation()
            x_star, alpha_star, p_star = self.alternating_optimization()
            
            # 5. 性能计算与队列更新
            D_l, D_o, E_total, D_total = self._calculate_performance_metrics(c_star, f_star, x_star, alpha_star, p_star)
            self._update_queues(D_l, D_o, A, c_star)
            
            # 6. 计算当前EE和时延（论文Section IV.A）
            current_EE = E_total / D_total if D_total > 0 else 0.0
            average_queue = np.mean(self.Q_l + self.Q_o)
            service_delay = average_queue / (np.mean(A) * self.U) if np.mean(A) > 0 else 0.0
            
            # 7. 记录性能历史，触发V自适应调整
            self._record_perf_history(current_EE, service_delay)
            if self.time_slot_count % self.adapt_interval == 0:
                self._adapt_V_based_perf()
            
            # 8. 转换结果为环境兼容格式
            decisions = self._convert_to_environment_decision(x_star, alpha_star, p_star, f_star, c_star)
            
            # 添加用户测试代码所需的额外信息
            decisions['current_V'] = self.current_V
            decisions['current_EE'] = current_EE
            decisions['service_delay'] = service_delay
            
            logger.info("时间槽{}求解完成：EE={:.3e} J/bit，时延={:.3f} ms，当前V={:.1e}".format(
                self.time_slot_count, current_EE, service_delay*1000, self.current_V))
            return decisions
            
        except Exception as e:
            logger.error(f"OORAA_Solver求解出错: {e}", exc_info=True)
            return self._get_default_decision(system_state)

    def _convert_to_environment_decision(self, x_star: np.ndarray, alpha_star: np.ndarray, p_star: np.ndarray, f_star: np.ndarray, c_star: np.ndarray) -> Dict[str, Any]:
        """
        Convert OORAA optimization results to the format expected by the environment.
        
        Args:
            x_star: User association matrix
            alpha_star: Bandwidth allocation ratios
            p_star: Transmit power
            f_star: Local CPU frequency
            c_star: Task splitting ratios
            
        Returns:
            decision: Dictionary containing the resource allocation decisions
        """
        # Initialize decision variables
        num_devices = self.U
        num_servers = self.M
        
        assignment = []
        power = np.zeros(num_devices)
        bandwidth = np.zeros((num_devices, num_servers))
        freq = np.zeros((num_devices, num_servers))
        
        # Convert assignment matrix to list of server indices
        for u in range(num_devices):
            if np.any(x_star[u] > 0.5):
                j = np.where(x_star[u] > 0.5)[0][0]
            else:
                j = 0  # Default to first server if no assignment
            assignment.append(j)
            
            # Power allocation
            power[u] = p_star[u, j]
            
            # Bandwidth allocation (convert to MHz)
            bandwidth[u, j] = alpha_star[u, j] * self.omega / 1e6
            
            # CPU frequency allocation for MEC server
            # Based on offloaded task portion
            offload_portion = 1 - c_star[u]
            # For MEC frequency, we'll use the server's capacity divided by number of devices
            # This is a simplified approach for compatibility
            freq[u, j] = self.f_max / self.N_max
            
        # Log decision values for debugging
        logger.debug(f"Assignment: {assignment}")
        logger.debug(f"Power: {power}")
        logger.debug(f"Bandwidth: {bandwidth}")
        logger.debug(f"Frequency: {freq}")
        
        return {
            'assignment': assignment,
            'power': power.tolist(),
            'bandwidth': bandwidth.tolist(),
            'freq': freq.tolist(),
            'debug': {
                'c_star': c_star.tolist(),
                'f_star': f_star.tolist(),
                'x_star': x_star.tolist(),
                'alpha_star': alpha_star.tolist(),
                'p_star': p_star.tolist(),
                'current_V': self.current_V,
                'perf_preference': self.perf_preference
            }
        }
    
    def _get_default_decision(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        num_devices = self.U
        num_servers = self.M
        
        # Initialize default decisions
        assignment = [0] * num_devices  # Assign all to first server
        power = np.zeros(num_devices).tolist()
        bandwidth = np.zeros((num_devices, num_servers)).tolist()
        freq = np.zeros((num_devices, num_servers)).tolist()
        
        # Set minimal values to avoid division by zero
        for u in range(num_devices):
            bandwidth[u][0] = 1.0  # 1 MHz minimum
            freq[u][0] = 1e6  # 1 MHz minimum
        
        # 在默认决策中也添加相同的字段，确保结构一致
        current_EE = 0.0
        average_queue = np.mean(self.Q_l + self.Q_o)
        service_delay = average_queue / (self.A_min * self.U) if self.A_min > 0 else 0.0
        
        return {
            'assignment': assignment,
            'power': power,
            'bandwidth': bandwidth,
            'freq': freq,
            'current_V': self.current_V,
            'current_EE': current_EE,
            'service_delay': service_delay,
            'debug': {
                'error': 'Default decision due to solver error',
                'current_V': self.current_V
            }
        }