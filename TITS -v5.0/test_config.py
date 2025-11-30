import numpy as np
from solvers.OORAA_Solver import OORAA_Solver

# 环境配置（参考论文仿真参数）
env_config = {
    'num_mec_servers': 3,
    'num_devices': 10,
    'N_max': 4,
    'tau': 1e-3,  # 1ms
    'bandwidth': 1e6,  # 1MHz
    'noise_variance': 10**(-174/10)*1e-3,  # -174dBm/Hz转功率
    'interference_power': 1e-13,  # 1e-13 W
    'path_loss_coeff': 10**(-40/10),  # -40dB
    'reference_distance': 1.0,  # 1m
    'path_loss_exponent': 4.0,
    'capacitance_coeff': 1e-28,  # 1e-28 F
    'comp_intensity': 737.5,  # cycles/bit
    'max_cpu_freq': 2.15e9,  # 2.15GHz
    'max_transmit_power': 1.0,  # 1W
    'task_min_size': 1e3,  # 1e3 bit
    'task_max_size': 2e3   # 2e3 bit
}

# 求解器配置（V自适应参数）
cfg = {
    'max_iter': 200,
    'bandwidth_tol': 1e-7,
    'alpha_min': 1e-4,
    'init_V': 1e11,  # 初始V值
    'perf_preference': 'balanced',  # 均衡模式
    'target_EE': 1e-7,  # 目标EE（1e-7 J/bit）
    'target_delay': 0.02,  # 目标时延（20ms）
    'adapt_interval': 10,  # 每10个时间槽调整V
    'V_min': 1e9,
    'V_max': 1e13
}

# 初始化求解器
solver = OORAA_Solver(env_config, cfg)

# 模拟系统状态（设备位置、MEC位置、任务到达）
system_state = {
    'device_positions': np.random.rand(10, 2)*100,  # 10个设备在100×100m区域
    'mec_positions': np.array([[25,25], [50,75], [75,25]]),  # 3个MEC位置
    'tasks': np.random.uniform(1e3, 2e3, 10)  # 10个设备的任务到达量
}

# 求解（单时间槽）
result = solver.solve(system_state)
print("当前V值：", result['current_V'])
print("当前EE：", result['current_EE'])
print("当前时延：", result['service_delay']*1000, "ms")