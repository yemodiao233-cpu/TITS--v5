#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BARGAIN_MATCH_Solver 测试脚本
"""

import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from solvers.BARGAIN_MATCH_Solver import BARGAIN_MATCH_Solver
    print("✓ 成功导入 BARGAIN_MATCH_Solver")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_solver_initialization():
    """测试求解器初始化"""
    print("\n=== 测试求解器初始化 ===")
    try:
        # 创建模拟的环境配置
        env_config = {
            'num_servers': 2,
            'num_vehicles': 3
        }
        
        # 创建配置字典
        cfg = {
            'vehicle_weight': 0.5,
            'server_weight': 0.6,
            'cpu_tau': 3
        }
        
        # 初始化求解器
        solver = BARGAIN_MATCH_Solver(env_config, cfg)
        print(f"✓ 求解器初始化成功")
        print(f"  - 服务器数量: {solver.num_servers}")
        print(f"  - 车辆数量: {solver.num_vehicles}")
        
        return solver
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return None

def create_test_system_state(num_vehicles=3, num_servers=2):
    """创建测试用的系统状态"""
    # 创建信道增益矩阵
    g = np.random.random((num_vehicles, num_servers)) * 0.1 + 0.01  # 0.01-0.11的随机增益
    
    # 创建任务列表
    tasks = []
    for i in range(num_vehicles):
        tasks.append({
            'Din': 0.1 + i * 0.05,  # 0.1-0.25 MB
            'Cv': 200000 + i * 50000,  # 200K-350K CPU cycles
            'kv': i % 3
        })
    
    # 创建系统状态
    system_state = {
        'time_step': 1,
        'V_set': list(range(num_vehicles)),
        'J_set': list(range(num_servers)),
        'g': g.tolist(),
        'tasks': tasks,
        'params': {
            'time_slot': 0.1,  # 100ms
            'noise_power': 1e-10,
            'Fmax_j': [2e9, 2e9]  # 每个服务器2GHz
        }
    }
    
    return system_state

def test_solve_method(solver):
    """测试solve方法"""
    print("\n=== 测试solve方法 ===")
    try:
        # 创建测试系统状态
        system_state = create_test_system_state(
            num_vehicles=solver.num_vehicles,
            num_servers=solver.num_servers
        )
        
        # 调用solve方法
        decision = solver.solve(system_state)
        
        # 验证决策格式
        required_keys = ['assignment', 'power', 'bandwidth', 'freq']
        for key in required_keys:
            if key not in decision:
                print(f"✗ 缺少必要的决策键: {key}")
                return False
        
        print(f"✓ solve方法调用成功，返回决策包含所有必要字段")
        
        # 验证assignment
        print(f"  Assignment: {decision['assignment']}")
        
        # 验证power
        print(f"  Power: {decision['power'][:3]}..." if len(decision['power']) > 3 else f"  Power: {decision['power']}")
        
        # 验证bandwidth形状
        bandwidth_shape = (len(decision['bandwidth']), len(decision['bandwidth'][0]) if decision['bandwidth'] else 0)
        print(f"  Bandwidth shape: {bandwidth_shape}")
        
        # 验证freq
        print(f"  Freq: {decision['freq']}")
        
        return True
    except Exception as e:
        print(f"✗ solve方法执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases(solver):
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试1: 空的系统状态
    try:
        empty_state = {}
        decision = solver.solve(empty_state)
        print(f"✓ 成功处理空系统状态")
    except Exception as e:
        print(f"✗ 处理空系统状态失败: {e}")
    
    # 测试2: 错误的信道增益矩阵
    try:
        bad_g_state = create_test_system_state()
        bad_g_state['g'] = "invalid"
        decision = solver.solve(bad_g_state)
        print(f"✓ 成功处理错误的信道增益矩阵")
    except Exception as e:
        print(f"✗ 处理错误的信道增益矩阵失败: {e}")
    
    # 测试3: 任务数量不足
    try:
        few_tasks_state = create_test_system_state()
        few_tasks_state['tasks'] = []
        decision = solver.solve(few_tasks_state)
        print(f"✓ 成功处理任务数量不足的情况")
    except Exception as e:
        print(f"✗ 处理任务数量不足的情况失败: {e}")

def main():
    """主测试函数"""
    print("BARGAIN_MATCH_Solver 测试脚本")
    print("=" * 50)
    
    # 测试1: 初始化
    solver = test_solver_initialization()
    if not solver:
        print("测试失败: 求解器初始化失败")
        return False
    
    # 测试2: solve方法
    solve_success = test_solve_method(solver)
    if not solve_success:
        print("测试失败: solve方法执行失败")
    
    # 测试3: 边界情况
    test_edge_cases(solver)
    
    print("\n" + "=" * 50)
    print(f"测试完成: {'通过' if solve_success else '部分失败'}")
    return solve_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)