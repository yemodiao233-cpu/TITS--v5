import numpy as np
from solvers.basesolver import BaseSolver
from scipy.stats import nakagami  # 用于复现Nakagami-m衰落模型
import warnings
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - BARGAIN_MATCH_Solver - %(levelname)s - %(message)s'
)
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class BARGAIN_MATCH_Solver(BaseSolver):
    def __init__(self, env_config: dict, cfg: dict):
        try:
            # 初始化基类
            super().__init__(env_config, cfg)
            
            # 日志：初始化开始
            logger.info("初始化BARGAIN_MATCH_Solver")
            
            # 输入验证
            if not isinstance(env_config, dict):
                raise TypeError("env_config must be a dictionary")
            if cfg is not None and not isinstance(cfg, dict):
                raise TypeError("cfg must be a dictionary or None")
            
            # 1. 论文Table II标准参数
            self.cfg = cfg or {}
            self.w_i = self._get_valid_param("vehicle_weight", 0.5)
            self.w_j = self._get_valid_param("server_weight", 0.6)
            self.tau = self._get_valid_param("cpu_tau", 3)
            self.alpha_i = self._get_valid_param("vehicle_alpha", 7.8e-21)
            self.alpha_j = self._get_valid_param("server_alpha", 7.8e-21)
            self.r_f = self._get_valid_param("fiber_rate", 4e9)
            self.r_c = self._get_valid_param("cloud_rate", 1e8)
            self.S_j = self._get_valid_param("sic_capacity", 8)
            self.handover_cost = self._get_valid_param("handover_cost", 0.02)
            self.p_L0 = self._get_valid_param("los_base_prob", 0.9)
            self.p_L_decay = self._get_valid_param("los_decay", 0.002)
            
            # 获取服务器和车辆数量（与环境保持一致）
            self.num_servers = max(1, int(env_config.get('num_servers', 2)))
            self.num_vehicles = max(1, int(env_config.get('num_vehicles', 3)))
            
            # 通信参数（默认值）
            self.transmit_power_watts = 0.01  # 默认10mW
            self.bandwidth = 40e6  # 40MHz
            self.noise_power_watts = 10**(-85/10) / 1000  # -85dBm转为瓦特
            
            # 初始化任务队列
            self.server_task_queue = {s: [] for s in range(self.num_servers)}
            self.cloud_task_queue = []
            
            # 日志：初始化完成
            logger.info(f"求解器初始化完成: {self.num_vehicles}车辆, {self.num_servers}服务器")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            # 确保即使初始化失败，也能提供基本功能
            self.num_servers = 2
            self.num_vehicles = 3
            self.cfg = {}
            self.transmit_power_watts = 0.01
            self.bandwidth = 40e6
            self.noise_power_watts = 10**(-85/10) / 1000
            self.server_task_queue = {s: [] for s in range(self.num_servers)}
            self.cloud_task_queue = []
            logger.warning("使用默认配置继续")
    
    def _get_valid_param(self, param_name: str, default_value: float) -> float:
        """
        获取并验证参数值
        """
        try:
            value = self.cfg.get(param_name, default_value)
            value = float(value)
            # 确保参数为正数
            if value <= 0 and param_name not in ['handover_cost']:
                logger.warning(f"参数 {param_name} 值为非正数，使用默认值 {default_value}")
                return default_value
            return value
        except (TypeError, ValueError):
            logger.warning(f"参数 {param_name} 类型无效，使用默认值 {default_value}")
            return default_value

    def calculate_los_probability(self, d_ij: float) -> float:
        """
        优化：严格复现论文LoS概率模型（论文附录补充的距离衰减模型）
        :param d_ij: 车辆-服务器距离（m）
        :return: LoS概率p_L
        """
        # 论文采用：p_L(d) = p_L0 * exp(-p_L_decay * d)，距离越远LoS概率越低
        p_L = self.p_L0 * np.exp(-self.p_L_decay * d_ij)
        return max(0.1, min(0.95, p_L))  # 限制概率范围（避免极端值）

    def calculate_channel_gain(self, vehicle_id: int, server_id: int, system_state: dict) -> float:
        """
        从系统状态获取信道增益矩阵
        """
        try:
            # 输入验证
            if not isinstance(vehicle_id, int) or not isinstance(server_id, int):
                logger.warning(f"无效的车辆/服务器ID类型: {vehicle_id}, {server_id}")
                return 1e-12
            
            if vehicle_id < 0 or server_id < 0:
                logger.warning(f"负的车辆/服务器ID: {vehicle_id}, {server_id}")
                return 1e-12
            
            # 获取信道增益矩阵
            g = system_state.get('g', None)
            
            if g is None:
                logger.info("系统状态中未找到信道增益矩阵，使用默认值")
                return 1e-12
            
            # 转换为numpy数组
            try:
                g_array = np.array(g)
            except Exception:
                logger.warning("无法转换信道增益矩阵为numpy数组")
                return 1e-12
            
            # 检查数组维度和索引
            if len(g_array.shape) != 2:
                logger.warning(f"信道增益矩阵维度不正确: {g_array.shape}")
                return 1e-12
            
            if vehicle_id >= g_array.shape[0] or server_id >= g_array.shape[1]:
                logger.warning(f"索引超出范围: 车辆={vehicle_id}, 服务器={server_id}, 矩阵形状={g_array.shape}")
                return 1e-12
            
            # 获取值并确保为正数
            gain_value = float(g_array[vehicle_id, server_id])
            if gain_value <= 0:
                logger.debug(f"负增益值替换为最小值: {gain_value}")
                return 1e-12
            
            return gain_value
            
        except Exception as e:
            logger.error(f"计算信道增益时出错: {e}")
            return 1e-12

    def predict_vehicle_mobility(self, vehicle_id: int, server_id: int, t: int) -> tuple:
        """
        优化：复现论文马尔可夫移动模型（公式1补充）
        返回：(停留时间T_soj, 目标服务器j_arr, 切换成本handover_cost)
        """
        vehicle = self.env.vehicles[vehicle_id]
        server_pos = self.env.servers[server_id].position
        vehicle_pos = vehicle.position[t]
        v_i = vehicle.velocity[t]
        R_j = self.env.servers[server_id].comm_range  # 服务器通信范围（论文默认500m）

        # 1. 移动方向指示符：马尔可夫模型（优化：替换原随机方向）
        if t == 0:
            # 初始时刻：随机方向（论文假设）
            zeta_i = self.vehicle_prev_dir[vehicle_id]
        else:
            # 马尔可夫转移：根据历史方向预测当前方向（论文公式1补充）
            prev_dir = self.vehicle_prev_dir[vehicle_id]
            trans_prob = self.mobility_trans_mat[0] if prev_dir == 1 else self.mobility_trans_mat[1]
            zeta_i = 1 if np.random.random() < trans_prob[0] else -1
            self.vehicle_prev_dir[vehicle_id] = zeta_i  # 更新历史方向

        # 2. 停留时间（论文公式2，新增切换成本影响）
        d_ij = abs(vehicle_pos[0] - server_pos[0])  # 水平距离（论文明确采用）
        T_soj = (R_j + zeta_i * d_ij) / v_i
        T_soj += self.handover_cost  # 叠加切换成本（论文Section III提及）

        # 3. 目标服务器预测（论文公式3，修正原窗口计算逻辑）
        # 论文中T_move = 任务传输时间 + 计算时间（非固定窗口）
        current_server = self.env.vehicles[vehicle_id].current_server[t]
        task = vehicle.tasks[t]
        # 计算T_move（任务完成前的车辆移动时间）
        T_tran = task.input_size / self.calculate_transmission_rate(vehicle_id, current_server, t)
        f_temp = self.env.servers[current_server].max_compute / (len(self.server_task_queue[current_server]) + 1)
        T_comp = (task.complexity * task.input_size) / f_temp
        T_move = T_tran + T_comp

        # 论文公式3：目标服务器索引计算
        j_arr = server_id + zeta_i * np.ceil(
            (v_i * T_move - (R_j + zeta_i * d_ij)) / (v_i * self.env.delta_t)
        )
        # 边界限制：确保目标服务器在有效范围内
        j_arr = max(0, min(len(self.env.servers) - 1, j_arr))

        return T_soj, int(j_arr), self.handover_cost

    def calculate_delay(self, vehicle_id: int, server_id: int, t: int, is_cloud: bool = False) -> tuple:
        """
        优化1：拆分延迟组件（便于日志记录，匹配论文公式9/11的分项）
        优化2：新增任务队列延迟（论文Section III-A服务器模型）
        返回：(总延迟T_total, 传输延迟T_tran, 计算延迟T_comp, 迁移延迟T_mig)
        """
        vehicle = self.env.vehicles[vehicle_id]
        task = vehicle.tasks[t]
        D_in = task.input_size  # 任务输入大小（KB，论文范围[400,1000]）
        D_out = task.output_size  # 任务输出大小（KB，论文范围[0.1,1]）
        C_req = task.complexity * D_in * 1024  # 总计算资源需求（cycles，转换为比特）

        # 1. 传输延迟（论文公式9/11）
        current_server = self.env.vehicles[vehicle_id].current_server[t]
        if is_cloud:
            # 云卸载：车辆→当前边缘服务器（论文明确传输路径）
            rate = self.calculate_transmission_rate(vehicle_id, current_server, t)
        else:
            rate = self.calculate_transmission_rate(vehicle_id, server_id, t)
        T_tran = (D_in * 1024 * 8) / rate  # 转换为比特计算（避免单位错误）

        # 2. 计算延迟（新增：任务队列等待延迟，论文Section III-A）
        if is_cloud:
            queue_len = len(self.cloud_task_queue)
            f_j_max = self.env.cloud_server.max_compute
        else:
            queue_len = len(self.server_task_queue[server_id])
            f_j_max = self.env.servers[server_id].max_compute
        # 资源分配：按队列长度均分（论文默认策略）
        f_j_i = f_j_max / (queue_len + 1)  # +1为当前任务
        T_comp_queue = (queue_len * C_req) / f_j_max  # 队列等待延迟
        T_comp_exec = C_req / f_j_i  # 任务执行延迟
        T_comp = T_comp_queue + T_comp_exec  # 总计算延迟

        # 3. 迁移延迟（论文公式9/11，区分水平/垂直）
        if is_cloud:
            # 垂直迁移：任务→云（D_in） + 结果→目标服务器（D_out）
            T_task_handover = (D_in * 1024 * 8) / self.r_c
            T_result_handover = (D_out * 1024 * 8) / self.r_c
            T_mig = T_task_handover + T_result_handover
        else:
            # 水平迁移：任务→目标服务器 + 结果→目标服务器（论文公式9）
            T_task_handover = (2 * D_in * 1024 * 8) / self.r_f if server_id != current_server else 0
            T_soj, j_arr, _ = self.predict_vehicle_mobility(vehicle_id, server_id, t)
            T_result_handover = (2 * D_out * 1024 * 8) / self.r_f if server_id != j_arr else 0
            T_mig = T_task_handover + T_result_handover

        # 总延迟（叠加切换成本）
        T_total = T_tran + T_comp + T_mig

        # 验证延迟约束（论文约束C4：T_total ≤ T_max）
        if T_total > task.deadline:
            # 标记任务为不可行（后续匹配会过滤）
            self.env.vehicles[vehicle_id].tasks[t].is_feasible = False

        return T_total, T_tran, T_comp, T_mig

    def bargaining_game(self, vehicle_id: int, server_id: int, t: int, is_cloud: bool = False) -> tuple:
        """
        优化1：修正资源分配公式（原代码F(var1,c)计算错误，匹配论文公式24）
        优化2：新增议价失败处理（论文Algorithm 1逻辑）
        返回：(最优资源分配f*, 最优定价c*, 议价成功标记is_success)
        """
        vehicle = self.env.vehicles[vehicle_id]
        task = vehicle.tasks[t]
        if not task.is_feasible:
            return 0.0, 0.0, False  # 任务不可行，议价失败

        # 1. 基础参数（严格匹配论文定义）
        T_max = task.deadline  # 任务最大延迟（论文范围[0.1,5]s）
        T_total, T_tran, T_comp, T_mig = self.calculate_delay(vehicle_id, server_id, t, is_cloud)
        C_req = task.complexity * task.input_size * 1024  # 计算资源需求（cycles）
        # 车辆约束
        E_i_max = vehicle.max_energy  # 车辆能量约束（论文：(Wh/GHz)*f_i^max）
        C_i_max = vehicle.budget  # 车辆支付预算（论文20$）
        # 服务器约束
        if is_cloud:
            server = self.env.cloud_server
            C_j_max = server.max_price  # 云服务器最高定价（论文1$/GHz）
            f_j_max = server.max_compute  # 云服务器计算能力（30GHz，Table II）
            queue_len = len(self.cloud_task_queue)
        else:
            server = self.env.servers[server_id]
            C_j_max = server.max_price  # 边缘服务器最高定价（1$/GHz）
            f_j_max = server.max_compute  # 边缘服务器计算能力（[2,10]GHz）
            queue_len = len(self.server_task_queue[server_id])

        # 2. 价格上下界（论文Lemma 1，修正原临时资源分配逻辑）
        f_temp = f_j_max / (queue_len + 1)  # 考虑队列长度的临时分配
        # 下界c_min：服务器效用≥0（论文公式推导）
        c_min = (
            (1 - self.w_j) * self.alpha_j * (f_temp) ** (self.tau - 2) * C_req * C_j_max * f_j_max
        ) / (self.w_j * server.max_energy)
        # 上界c_max：车辆效用≥0（论文公式推导）
        psi = np.log(1 + (T_max - T_total)) / np.log(1 + T_max) if T_total < T_max else 0
        c_max = (self.w_i * psi * C_i_max) / ((1 - self.w_i) * f_temp) if (1 - self.w_i) > 0 else np.inf
        # 边界修正：避免价格为负或无穷大
        c_min = max(1e-6, c_min)  # 最小定价（避免零定价）
        c_max = max(c_min, min(c_max, C_j_max))  # 不超过服务器最高定价
        delta_c = c_max - c_min  # 议价空间

        # 3. 折扣因子（论文公式25-26，修正原传输时间计算）
        epsilon_i = 1 - (T_tran / T_max)  # 车辆折扣因子（传输时间占比）
        epsilon_j = 1 - (T_comp / T_max)  # 服务器折扣因子（计算时间占比）
        epsilon_i = max(0.1, min(0.9, epsilon_i))  # 限制折扣因子范围（论文建议）
        epsilon_j = max(0.1, min(0.9, epsilon_j))

        # 4. 最优分区（论文Lemma 2，修正原奇偶轮次计算）
        T_b = self.config.get("bargaining_rounds", 5)  # 议价轮数（论文默认5轮）
        if T_b % 2 == 0:
            # 偶数轮：车辆最后提议（论文公式27）
            delta_i = epsilon_i - (1 - epsilon_i) * (1 - (epsilon_i * epsilon_j) ** (T_b / 2)) / (1 - epsilon_i * epsilon_j)
        else:
            # 奇数轮：服务器最后提议（论文公式28）
            delta_i = (1 - epsilon_j) * (1 - (epsilon_i * epsilon_j) ** ((T_b + 1) / 2)) / (1 - epsilon_i * epsilon_j)
        delta_i = max(0, min(1, delta_i))  # 分区比例限制在[0,1]

        # 5. 最优定价（论文Theorem 3）
        c_star = c_max - delta_c * delta_i

        # 6. 最优资源分配（论文公式27，修正原F(var1,c)计算错误）
        var1 = T_tran + T_mig  # 非计算延迟项（论文公式24定义）
        # 论文公式24：F(var1, c)的精确计算
        numerator_inner = c_star * C_req * np.log(1 + T_max) * (1 - self.w_i) + 4 * C_i_max * self.w_i * (1 + T_max - var1)
        F_var = np.sqrt(
            (c_star * np.log(1 + T_max) * (1 - self.w_i) * numerator_inner) / C_req
        )
        # 论文公式27：最优资源分配
        denominator = F_var - np.log(1 + T_max) * c_star * (1 - self.w_i)
        if denominator <= 0:
            return 0.0, 0.0, False  # 分母为负，议价失败
        f_star = (2 * self.w_i * C_i_max) / denominator

        # 7. 资源约束检查（论文约束C8-C9）
        f_star = min(f_star, f_j_max / (queue_len + 1))  # 不超过服务器剩余资源
        f_star = max(1e6, f_star)  # 最低资源保障（1MHz，避免过低）

        # 8. 验证双方效用（论文Algorithm 1逻辑）
        # 车辆效用（公式17）
        cost_vehicle = (c_star * f_star) / C_i_max if not is_cloud else (c_star * f_star) / C_i_max
        u_i = self.w_i * psi - (1 - self.w_i) * cost_vehicle
        # 服务器效用（公式21）
        revenue_server = (c_star * f_star) / (C_j_max * f_j_max)
        energy_cost_server = (self.alpha_j * (f_star) ** (self.tau - 1) * C_req) / server.max_energy
        u_j = self.w_j * revenue_server - (1 - self.w_j) * energy_cost_server

        # 议价成功条件：双方效用均为正
        is_success = (u_i > 0) and (u_j > 0)
        return f_star if is_success else 0.0, c_star if is_success else 0.0, is_success

    def matching_algorithm(self, t: int) -> dict:
        """
        优化1：新增任务可行性过滤（论文约束C4）
        优化2：匹配后更新任务队列（论文Section III-A服务器模型）
        优化3：复现论文Algorithm 2的拒绝-重匹配逻辑
        """
        # 1. 初始化：过滤不可行任务（延迟超期的任务直接本地执行）
        req_vehicles = []
        for v in self.env.vehicles:
            if v.has_task[t]:
                task = v.tasks[t]
                # 预计算本地执行延迟（判断是否可行）
                T_local = (task.complexity * task.input_size * 1024) / v.max_compute
                task.is_feasible = (T_local <= task.deadline)
                req_vehicles.append(v.id)

        # 2. 服务器集定义（边缘+云）
        edge_servers = [s.id for s in self.env.servers]
        cloud_id = len(edge_servers)  # 云服务器虚拟ID（便于统一处理）
        all_servers = edge_servers + [cloud_id]

        # 3. 构建偏好列表（论文公式29-32，优化：仅保留议价成功的选项）
        vehicle_prefs = {}  # {v_id: [(s_id, is_cloud, f*, c*, u_i), ...]}
        server_prefs = {}   # {s_id: [(v_id, f*, c*, u_j), ...]}

        for v_id in req_vehicles:
            v = self.env.vehicles[v_id]
            task = v.tasks[t]
            if not task.is_feasible:
                vehicle_prefs[v_id] = []
                continue

            prefs = []
            for s_id in all_servers:
                is_cloud = (s_id == cloud_id)
                target_s_id = 0 if is_cloud else s_id  # 云服务器对应实际索引（环境定义）
                
                # 执行议价博弈（仅保留成功结果）
                f_star, c_star, is_success = self.bargaining_game(v_id, target_s_id, t, is_cloud)
                if not is_success:
                    continue

                # 计算车辆对服务器的偏好值（效用u_i，公式17）
                T_total, _, _, _ = self.calculate_delay(v_id, target_s_id, t, is_cloud)
                psi = np.log(1 + (task.deadline - T_total)) / np.log(1 + task.deadline)
                cost = (c_star * f_star) / v.budget
                u_i = self.w_i * psi - (1 - self.w_i) * cost
                prefs.append((s_id, is_cloud, f_star, c_star, u_i))

            # 按效用降序排序（论文要求）
            prefs.sort(key=lambda x: x[4], reverse=True)
            vehicle_prefs[v_id] = prefs

            # 构建服务器对车辆的偏好列表（公式29）
            for s_id, is_cloud, f_star, c_star, _ in prefs:
                target_s_id = 0 if is_cloud else s_id
                server = self.env.cloud_server if is_cloud else self.env.servers[target_s_id]
                C_j_max = server.max_price
                f_j_max = server.max_compute
                C_req = task.complexity * task.input_size * 1024

                # 计算服务器效用（公式21）
                revenue = (c_star * f_star) / (C_j_max * f_j_max)
                energy_cost = (self.alpha_j * (f_star) ** (self.tau - 1) * C_req) / server.max_energy
                u_j = self.w_j * revenue - (1 - self.w_j) * energy_cost

                if s_id not in server_prefs:
                    server_prefs[s_id] = []
                server_prefs[s_id].append((v_id, f_star, c_star, u_j))

        # 4. 多对一匹配（复现论文Algorithm 2的拒绝-重匹配逻辑）
        matching = {v_id: None for v_id in req_vehicles}
        rejected = set(req_vehicles)  # 初始拒绝集=所有请求车辆
        server_matches = {s_id: [] for s_id in all_servers}  # 服务器当前匹配的车辆

        while rejected:
            for v_id in list(rejected):
                # 车辆无可用偏好时，标记为本地执行
                if not vehicle_prefs.get(v_id, []):
                    matching[v_id] = (-1, False, v.max_compute, 0.0)  # -1=本地
                    rejected.remove(v_id)
                    continue

                # 步骤1：车辆向偏好最高的服务器发送请求
                s_id, is_cloud, f_star, c_star, _ = vehicle_prefs[v_id][0]
                server = self.env.cloud_server if is_cloud else self.env.servers[s_id]
                current_matches = server_matches[s_id]

                # 步骤2：服务器检查资源约束（论文约束C8-C9）
                # 计算当前资源占用
                used_f = sum(m[2] for m in current_matches) if current_matches else 0.0
                used_cores = len(current_matches)
                # 检查约束：计算资源≤max，核心数≤core_count
                f_ok = (used_f + f_star) <= server.max_compute
                core_ok = (used_cores + 1) <= server.core_count

                if f_ok and core_ok:
                    # 步骤3：匹配成功
                    current_matches.append((v_id, is_cloud, f_star, c_star))
                    server_matches[s_id] = current_matches
                    matching[v_id] = (s_id, is_cloud, f_star, c_star)
                    rejected.remove(v_id)
                else:
                    # 步骤4：匹配失败，移除该服务器偏好，重新加入拒绝集
                    vehicle_prefs[v_id].pop(0)
                    if not vehicle_prefs[v_id]:
                        # 无更多偏好，本地执行
                        matching[v_id] = (-1, False, self.env.vehicles[v_id].max_compute, 0.0)
                        rejected.remove(v_id)

        # 5. 更新服务器任务队列（论文Section III-A，匹配后状态）
        self.server_task_queue = {s.id: [] for s in self.env.servers}
        self.cloud_task_queue = []
        for v_id, match in matching.items():
            s_id, is_cloud, _, _ = match
            if s_id == -1:
                continue  # 本地执行，不加入队列
            if is_cloud:
                self.cloud_task_queue.append(v_id)
            else:
                self.server_task_queue[s_id].append(v_id)

        return matching

    def solve(self, system_state: dict) -> dict:
        """
        符合BaseSolver接口规范的求解方法
        确保参数配置与环境一致，特别是服务器和设备数量
        
        Args:
            system_state: 系统状态字典，包含任务信息、信道信息等
            
        Returns:
            决策字典，包含assignment, power, bandwidth, freq等字段，格式符合框架要求
        """
        logger.info("开始求解任务分配和资源调度问题")
        
        # 初始化默认决策字典（作为安全回退）
        default_decisions = {
            'assignment': [],
            'power': [],
            'bandwidth': [],
            'freq': [[]]
        }
        
        try:
            # 验证输入
            if not isinstance(system_state, dict):
                logger.error("system_state必须是字典类型")
                return default_decisions
            
            # 1. 动态更新服务器和车辆数量（从系统状态的多个可能位置获取）
            # 优先级：params > 直接在system_state中 > 从V_set/J_set推断 > 初始化值
            
            # 从params获取
            if 'params' in system_state and isinstance(system_state['params'], dict):
                params = system_state['params']
                if 'num_servers' in params:
                    try:
                        self.num_servers = max(1, int(params['num_servers']))
                        logger.debug(f"从params获取服务器数: {self.num_servers}")
                    except (ValueError, TypeError):
                        logger.warning("params中num_servers格式无效")
                if 'num_vehicles' in params:
                    try:
                        self.num_vehicles = max(1, int(params['num_vehicles']))
                        logger.debug(f"从params获取车辆数: {self.num_vehicles}")
                    except (ValueError, TypeError):
                        logger.warning("params中num_vehicles格式无效")
                
                # 更新其他通信参数（如果提供）
                if 'transmit_power' in params:
                    try:
                        self.transmit_power_watts = max(0.001, float(params['transmit_power']))
                        logger.debug(f"从params获取发射功率: {self.transmit_power_watts}W")
                    except (ValueError, TypeError):
                        logger.warning("params中transmit_power格式无效")
                if 'bandwidth' in params:
                    try:
                        self.bandwidth = max(1e3, float(params['bandwidth']))
                        logger.debug(f"从params获取带宽: {self.bandwidth/1e6:.2f}MHz")
                    except (ValueError, TypeError):
                        logger.warning("params中bandwidth格式无效")
                if 'noise_power' in params:
                    try:
                        self.noise_power_watts = max(1e-14, float(params['noise_power']))
                        logger.debug(f"从params获取噪声功率: {self.noise_power_watts}W")
                    except (ValueError, TypeError):
                        logger.warning("params中noise_power格式无效")
            
            # 直接从system_state获取
            if 'num_servers' in system_state:
                try:
                    self.num_servers = max(1, int(system_state['num_servers']))
                    logger.debug(f"从system_state获取服务器数: {self.num_servers}")
                except (ValueError, TypeError):
                    logger.warning("system_state中num_servers格式无效")
            if 'num_vehicles' in system_state:
                try:
                    self.num_vehicles = max(1, int(system_state['num_vehicles']))
                    logger.debug(f"从system_state获取车辆数: {self.num_vehicles}")
                except (ValueError, TypeError):
                    logger.warning("system_state中num_vehicles格式无效")
            
            # 从V_set和J_set推断
            V_set = system_state.get('V_set', [])
            J_set = system_state.get('J_set', [])
            
            if isinstance(V_set, list) and V_set:
                try:
                    # 确保num_vehicles至少为V_set中的最大索引+1
                    max_vehicle_id = max(V_set)
                    self.num_vehicles = max(self.num_vehicles, max_vehicle_id + 1)
                    logger.debug(f"从V_set推断车辆数: {self.num_vehicles}")
                except Exception as e:
                    logger.warning(f"从V_set推断车辆数出错: {e}")
            
            if isinstance(J_set, list) and J_set:
                try:
                    # 确保num_servers至少为J_set中的最大索引+1
                    max_server_id = max(J_set)
                    self.num_servers = max(self.num_servers, max_server_id + 1)
                    logger.debug(f"从J_set推断服务器数: {self.num_servers}")
                except Exception as e:
                    logger.warning(f"从J_set推断服务器数出错: {e}")
            
            # 获取时间步
            t = system_state.get("time_step", 0)
            logger.debug(f"当前时间步: {t}")
            
            # 构建简化的匹配结果（适配框架要求）
            matching_result = self._simplified_matching(system_state)
            logger.debug("匹配算法执行完成")
            
            # 确保V_set是有效的列表
            if not isinstance(V_set, list):
                V_set = list(range(self.num_vehicles))
                logger.warning("V_set不是列表类型，使用默认范围")
            
            if not V_set:
                V_set = list(range(self.num_vehicles))
                logger.info(f"V_set为空，使用默认范围: {V_set}")
            
            # 初始化决策字典 - 使用与框架兼容的格式（二维数组）
            num_vehicles = len(V_set)
            num_servers = self.num_servers
            
            # 严格按照框架要求的格式初始化：
            # - assignment: 一维数组，每个元素是分配的服务器ID
            # - power: 一维数组，每个元素是发射功率
            # - bandwidth: 二维数组，shape=(num_servers, num_vehicles)
            # - freq: 二维数组，shape=(num_servers, num_vehicles)
            
            decisions = {
                'assignment': [-1] * num_vehicles,  # -1表示本地执行
                'power': [self.transmit_power_watts] * num_vehicles,
                'bandwidth': np.zeros((num_servers, num_vehicles)).tolist(),
                'freq': np.zeros((num_servers, num_vehicles)).tolist()
            }
            
            # 验证和调整数组维度
            try:
                # 确保bandwidth是二维数组
                if not isinstance(decisions['bandwidth'], list) or not decisions['bandwidth'] or not isinstance(decisions['bandwidth'][0], list):
                    logger.warning("bandwidth不是有效的二维数组，重新初始化")
                    decisions['bandwidth'] = [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)]
                
                # 确保freq是二维数组
                if not isinstance(decisions['freq'], list) or not decisions['freq'] or not isinstance(decisions['freq'][0], list):
                    logger.warning("freq不是有效的二维数组，重新初始化")
                    decisions['freq'] = [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)]
                
                # 调整bandwidth维度
                if len(decisions['bandwidth']) != num_servers:
                    logger.warning(f"bandwidth的服务器维度不匹配: {len(decisions['bandwidth'])} vs {num_servers}")
                    decisions['bandwidth'] = [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)]
                
                # 调整freq维度
                if len(decisions['freq']) != num_servers:
                    logger.warning(f"freq的服务器维度不匹配: {len(decisions['freq'])} vs {num_servers}")
                    decisions['freq'] = [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)]
                
                # 确保每行的长度正确
                for s in range(num_servers):
                    if len(decisions['bandwidth'][s]) != num_vehicles:
                        decisions['bandwidth'][s] = [0.0] * num_vehicles
                    if len(decisions['freq'][s]) != num_vehicles:
                        decisions['freq'][s] = [0.0] * num_vehicles
                
            except Exception as e:
                logger.error(f"调整数组维度时出错: {e}")
                # 重置为安全的默认值
                decisions['bandwidth'] = [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)]
                decisions['freq'] = [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)]
            
            # 应用匹配结果
            if isinstance(matching_result, dict):
                for v_idx, v_id in enumerate(V_set):
                    if v_id in matching_result and matching_result[v_id] is not None:
                        try:
                            s_id, f_star = matching_result[v_id]
                            s_id = int(s_id)
                            f_star = float(f_star)
                            
                            # 确保服务器ID有效
                            if s_id >= 0 and s_id < num_servers:
                                decisions['assignment'][v_idx] = s_id
                                # 分配频率资源
                                if 0 <= s_id < num_servers and 0 <= v_idx < num_vehicles:
                                    decisions['freq'][s_id][v_idx] = f_star
                                # 分配带宽资源
                                if 0 <= s_id < num_servers and 0 <= v_idx < num_vehicles:
                                    decisions['bandwidth'][s_id][v_idx] = self.bandwidth / 1e6  # 转换为MHz
                                
                                logger.debug(f"车辆 {v_id} 分配到服务器 {s_id}，频率 {f_star/1e9:.2f}GHz")
                        except (ValueError, TypeError, IndexError) as e:
                            logger.warning(f"应用匹配结果时出错 (车辆={v_id}): {e}")
            
            # 验证和修复决策格式
            try:
                # 确保所有值都是有效的Python类型
                decisions['assignment'] = [int(assign) for assign in decisions['assignment']]
                decisions['power'] = [max(0.001, float(p)) for p in decisions['power']]
                decisions['bandwidth'] = [[max(0.0, float(bw)) for bw in row] for row in decisions['bandwidth']]
                decisions['freq'] = [[max(0.0, float(f)) for f in row] for row in decisions['freq']]
                
                # 记录求解结果摘要
                offload_count = sum(1 for assign in decisions['assignment'] if assign != -1)
                logger.info(
                    f"求解完成: 服务器数={num_servers}, "
                    f"车辆数={num_vehicles}, "
                    f"卸载任务数={offload_count}, "
                    f"时间步={t}"
                )
                
                # 记录详细的决策（调试级别）
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"决策详情 - 分配: {decisions['assignment']}")
                    logger.debug(f"决策详情 - 功率: 平均={np.mean(decisions['power']):.4f}W")
                    logger.debug(f"决策详情 - 频率分配: 非零值={sum(sum(1 for f in row if f > 0) for row in decisions['freq'])}")
                
            except Exception as e:
                logger.error(f"验证决策格式时出错: {e}")
                # 重置为安全的默认值
                decisions = {
                    'assignment': [-1] * num_vehicles,
                    'power': [self.transmit_power_watts] * num_vehicles,
                    'bandwidth': [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)],
                    'freq': [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)]
                }
            
            return decisions
            
        except Exception as e:
            logger.error(f"求解过程中出错: {e}", exc_info=True)
            # 返回安全的默认决策
            try:
                V_set = system_state.get('V_set', []) if isinstance(system_state, dict) else []
                if not isinstance(V_set, list):
                    V_set = []
                
                num_vehicles = max(1, len(V_set))
                num_servers = max(1, self.num_servers)
                
                safe_decisions = {
                    'assignment': [-1] * num_vehicles,
                    'power': [0.01] * num_vehicles,  # 安全的默认功率
                    'bandwidth': [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)],
                    'freq': [[0.0 for _ in range(num_vehicles)] for _ in range(num_servers)]
                }
                
                logger.info(f"返回安全的默认决策: 车辆数={num_vehicles}, 服务器数={num_servers}")
                return safe_decisions
            except Exception:
                # 最后的安全回退
                return {
                    'assignment': [-1],
                    'power': [0.01],
                    'bandwidth': [[0.0]],
                    'freq': [[0.0]]
                }
    
    def _simplified_matching(self, system_state: dict) -> dict:
        """
        优化的任务匹配和资源分配逻辑
        正确处理任务信息格式和资源分配结果
        """
        try:
            logger.debug("开始简化匹配过程")
            
            # 确保从系统状态正确提取车辆和服务器集合
            V_set = system_state.get('V_set', [])
            J_set = system_state.get('J_set', [])
            
            # 如果集合为空，使用默认索引范围
            if not V_set:
                V_set = list(range(self.num_vehicles))
                logger.info(f"V_set为空，使用默认范围: {V_set}")
            if not J_set:
                J_set = list(range(self.num_servers))
                logger.info(f"J_set为空，使用默认范围: {J_set}")
            
            # 验证V_set和J_set的有效性
            V_set = [v for v in V_set if isinstance(v, int) and v >= 0]
            J_set = [j for j in J_set if isinstance(j, int) and j >= 0]
            
            # 获取任务列表和系统参数
            tasks = system_state.get('tasks', [])
            params = system_state.get('params', {})
            
            # 确保任务列表完整且格式正确
            normalized_tasks = []
            for idx, t in enumerate(tasks):
                try:
                    if isinstance(t, dict):
                        # 标准化任务参数
                        normalized_task = {
                            'Din': max(0.01, float(t.get('Din', 0.1))),      # 数据大小 (MB)
                            'Cv': max(1000, int(t.get('Cv', 200000))),      # 计算复杂度 (CPU cycles)
                            'kv': int(t.get('kv', 0)),                     # 任务优先级
                            'deadline': max(0.1, float(t.get('deadline', 1.0)))  # 截止时间
                        }
                        normalized_tasks.append(normalized_task)
                    else:
                        # 默认任务参数
                        normalized_tasks.append({'Din': 0.1, 'Cv': 200000, 'kv': 0, 'deadline': 1.0})
                        logger.debug(f"任务 {idx} 类型无效，使用默认参数")
                except Exception as e:
                    normalized_tasks.append({'Din': 0.1, 'Cv': 200000, 'kv': 0, 'deadline': 1.0})
                    logger.warning(f"处理任务 {idx} 时出错: {e}，使用默认参数")
            
            # 确保任务列表与车辆数量匹配
            while len(normalized_tasks) < len(V_set):
                normalized_tasks.append({'Din': 0.1, 'Cv': 200000, 'kv': 0, 'deadline': 1.0})
            
            # 获取服务器最大计算资源
            Fmax_j = params.get('Fmax_j', [])
            if not Fmax_j or len(Fmax_j) < len(J_set):
                # 确保有足够的服务器资源配置
                default_Fmax = 2e9  # 默认2GHz
                while len(Fmax_j) < len(J_set):
                    Fmax_j.append(default_Fmax)
                logger.info(f"Fmax_j配置不足，使用默认值扩展到 {len(Fmax_j)} 个服务器")
            
            # 验证和清理Fmax_j
            Fmax_j = [max(1e6, float(f)) for f in Fmax_j]
            
            # 初始化服务器资源分配跟踪
            server_allocated_freq = {s_id: 0.0 for s_id in J_set}
            
            # 基础匹配结果
            matching = {}
            local_count = 0
            offload_count = 0
            
            # 为每个车辆计算最佳服务器分配
            for v_idx, v_id in enumerate(V_set):
                try:
                    if v_idx >= len(normalized_tasks):
                        # 如果任务不足，跳过此车辆
                        matching[v_id] = (-1, 1e9)  # 默认本地执行
                        local_count += 1
                        continue
                        
                    task = normalized_tasks[v_idx]
                    C_req = task['Cv']
                    Din = task['Din']
                    
                    # 为每个车辆评估所有可能的服务器
                    best_value = -float('inf')
                    best_server = -1
                    best_freq = 1e9  # 默认本地频率
                    
                    # 计算本地执行的价值
                    local_execution_value = self._calculate_execution_value(
                        -1, task, 1e9, None, system_state)
                    
                    # 评估每个服务器
                    for s_id in J_set:
                        try:
                            # 检查服务器资源可用性
                            available_freq = Fmax_j[s_id] - server_allocated_freq[s_id]
                            if available_freq <= 0:
                                continue
                            
                            # 计算最大可分配频率
                            max_possible_freq = min(available_freq, Fmax_j[s_id] * 0.4)  # 最多40%服务器容量
                            
                            # 基于任务复杂度和服务器容量确定合适的频率
                            req_freq = max(1e6, C_req * 10)  # 基础计算需求
                            allocated_freq = min(max_possible_freq, req_freq)
                            
                            # 计算此分配的价值
                            value = self._calculate_execution_value(
                                s_id, task, allocated_freq, v_id, system_state)
                            
                            # 选择最佳分配
                            if value > best_value and value > local_execution_value:
                                best_value = value
                                best_server = s_id
                                best_freq = allocated_freq
                        except Exception as e:
                            logger.error(f"评估服务器 {s_id} 时出错: {e}")
                            continue
                    
                    # 更新服务器资源分配
                    if best_server >= 0 and best_server in server_allocated_freq:
                        server_allocated_freq[best_server] += best_freq
                        offload_count += 1
                    else:
                        local_count += 1
                    
                    # 验证频率值
                    best_freq = max(1e6, float(best_freq))
                    
                    # 存储匹配结果
                    matching[v_id] = (best_server, best_freq)
                
                except Exception as e:
                    logger.error(f"处理车辆 {v_id} 时出错: {e}")
                    matching[v_id] = (-1, 1e9)  # 默认本地执行
                    local_count += 1
            
            # 日志：匹配结果统计
            total_vehicles = len(V_set)
            logger.info(f"匹配完成: 总车辆={total_vehicles}, 本地执行={local_count}, 卸载={offload_count}")
            
            return matching
            
        except Exception as e:
            logger.error(f"简化匹配过程失败: {e}")
            # 返回保守的匹配结果
            matching = {v_id: (-1, 1e9) for v_id in range(min(self.num_vehicles, 10))}
            logger.warning("返回保守的默认匹配结果")
            return matching
    
    def _calculate_execution_value(self, s_id: int, task: dict, freq: float, v_id: int, system_state: dict) -> float:
        """
        计算任务执行的价值（用于决策）
        
        Args:
            s_id: 服务器ID (-1表示本地)
            task: 任务信息
            freq: 分配的计算频率
            v_id: 车辆ID
            system_state: 系统状态
            
        Returns:
            执行价值分数
        """
        try:
            # 输入验证
            if s_id == -1 or freq <= 0:
                # 本地执行或无效频率
                logger.debug(f"本地执行或无效频率: s_id={s_id}, freq={freq}")
                return 0.5  # 基准值
            
            if not isinstance(task, dict):
                logger.warning("任务不是字典类型")
                return 0.3
            
            # 计算传输速率（如果有车辆ID）
            if v_id is not None and isinstance(v_id, int) and v_id >= 0:
                try:
                    # 从系统状态获取信道增益
                    gain = self.calculate_channel_gain(v_id, s_id, system_state)
                    
                    # 计算传输速率
                    bandwidth = max(1e6, float(self.bandwidth))  # 确保带宽有效
                    if self.noise_power_watts <= 0:
                        snr = 10.0  # 默认中等信噪比
                    else:
                        snr = (self.transmit_power_watts * gain) / self.noise_power_watts
                    
                    # 确保SNR为正数
                    snr = max(0.1, snr)
                    
                    # 香农公式
                    rate = bandwidth * np.log2(1 + snr)
                    
                    # 限制速率范围
                    rate = max(1000, min(1e9, rate))
                    
                except Exception as e:
                    logger.warning(f"计算传输速率时出错: {e}，使用默认值")
                    rate = 10e6  # 默认10Mbps
            else:
                rate = 10e6
            
            # 计算延迟（简化模型）
            try:
                Din = float(task.get('Din', 0.1)) * 1e6 * 8  # 转换为比特
                Din = max(1000, Din)  # 确保数据大小有效
                
                # 传输延迟
                transmission_delay = Din / rate if rate > 0 else float('inf')
                
                # 计算延迟
                C_req = int(task.get('Cv', 200000))
                C_req = max(1000, C_req)  # 确保复杂度有效
                computation_delay = C_req / freq if freq > 0 else float('inf')
                
                # 总延迟
                total_delay = transmission_delay + computation_delay
                
                # 价值计算（延迟越小，价值越高）
                if total_delay > 0:
                    # 考虑截止时间约束
                    deadline = float(task.get('deadline', 1.0))
                    deadline = max(0.01, deadline)  # 确保截止时间有效
                    
                    if total_delay > deadline:
                        logger.debug(f"延迟 {total_delay} 超过截止时间 {deadline}")
                        return 0.1  # 超过截止时间的分配价值低
                    
                    # 基础价值计算
                    value = 1.0 / (1.0 + total_delay) * 100
                    
                    # 根据信噪比调整价值
                    snr_factor = 1.0
                    if v_id is not None and self.noise_power_watts > 0:
                        gain = self.calculate_channel_gain(v_id, s_id, system_state)
                        snr = (self.transmit_power_watts * gain) / self.noise_power_watts
                        snr = max(0.1, snr)  # 确保SNR有效
                        snr_factor = min(1.0, 0.5 + snr / 100)  # 信噪比因子
                    
                    # 限制价值范围
                    final_value = max(0.01, min(100.0, value * snr_factor))
                    
                    # 记录详细信息（调试级别）
                    if final_value > 5.0:  # 只记录较高价值的决策
                        logger.debug(
                            f"执行价值: {final_value:.2f}, "
                            f"车辆={v_id}, 服务器={s_id}, "
                            f"延迟={total_delay:.6f}s, "
                            f"速率={rate/1e6:.2f}Mbps"
                        )
                    
                    return final_value
                
            except Exception as e:
                logger.error(f"计算延迟时出错: {e}")
                
        except Exception as e:
            logger.error(f"计算执行价值时出错: {e}")
        
        return 0.0