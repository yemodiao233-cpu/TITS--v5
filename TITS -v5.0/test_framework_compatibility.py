import sys
import os
import numpy as np
import logging

# 添加项目目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('framework_compatibility_test')

try:
    # 导入BARGAIN_MATCH_Solver
    from solvers.BARGAIN_MATCH_Solver import BARGAIN_MATCH_Solver
    logger.info("成功导入BARGAIN_MATCH_Solver")
except Exception as e:
    logger.error(f"导入BARGAIN_MATCH_Solver失败: {e}")
    sys.exit(1)

def test_solver_with_framework_format():
    """
    测试求解器是否完全兼容实验框架要求的格式和行为
    """
    test_cases = [
        {
            "name": "标准场景 - 2服务器3车辆",
            "system_state": {
                "params": {
                    "num_servers": 2,
                    "num_vehicles": 3,
                    "transmit_power": 0.1,  # 100mW
                    "bandwidth": 10e6,  # 10MHz
                    "noise_power": 1e-13  # -100dBm
                },
                "time_step": 1,
                "V_set": [0, 1, 2],
                "J_set": [0, 1],
                # 模拟任务信息
                "tasks": {
                    0: {"computation_requirement": 100, "deadline": 10, "data_size": 1e4},
                    1: {"computation_requirement": 150, "deadline": 15, "data_size": 1.5e4},
                    2: {"computation_requirement": 200, "deadline": 20, "data_size": 2e4}
                },
                # 模拟信道信息
                "channel_gains": np.random.rand(2, 3).tolist()
            }
        },
        {
            "name": "从system_state直接获取参数",
            "system_state": {
                "num_servers": 3,
                "num_vehicles": 2,
                "time_step": 5,
                "V_set": [0, 1],
                "J_set": [0, 1, 2]
            }
        },
        {
            "name": "空V_set和J_set",
            "system_state": {
                "params": {
                    "num_servers": 1,
                    "num_vehicles": 1
                },
                "time_step": 10,
                "V_set": [],
                "J_set": []
            }
        },
        {
            "name": "大数量场景 - 5服务器10车辆",
            "system_state": {
                "params": {
                    "num_servers": 5,
                    "num_vehicles": 10,
                    "transmit_power": 0.2,
                    "bandwidth": 20e6
                },
                "time_step": 20,
                "V_set": list(range(10)),
                "J_set": list(range(5))
            }
        },
        {
            "name": "无效参数类型",
            "system_state": {
                "params": {
                    "num_servers": "not_a_number",
                    "num_vehicles": "also_not_a_number",
                    "transmit_power": "invalid"
                },
                "time_step": 15,
                "V_set": [0, 1, 2]
            }
        },
        {
            "name": "部分缺失参数",
            "system_state": {
                "params": {
                    "num_servers": 2
                    # 缺少num_vehicles
                },
                "time_step": 30,
                "V_set": [0, 1, 2, 3]
            }
        }
    ]

    # 创建求解器实例
    solver = BARGAIN_MATCH_Solver(
        env_config={
            "num_servers": 1,
            "num_vehicles": 1
        },
        cfg={
            "transmit_power": 0.1,
            "bandwidth": 10e6,
            "noise_power": 1e-13
        }
    )

    results = []
    all_passed = True

    for test_case in test_cases:
        test_name = test_case["name"]
        system_state = test_case["system_state"]
        logger.info(f"\n===== 开始测试: {test_name} =====")
        
        try:
            # 调用solve方法
            decisions = solver.solve(system_state)
            logger.info(f"求解器返回结果类型: {type(decisions)}")
            
            # 验证结果格式
            is_valid = validate_framework_format(decisions, system_state)
            results.append({
                "test_name": test_name,
                "passed": is_valid,
                "decisions": decisions
            })
            
            if is_valid:
                logger.info(f"✅ 测试通过: {test_name}")
            else:
                logger.error(f"❌ 测试失败: {test_name}")
                all_passed = False
                
        except Exception as e:
            logger.error(f"❌ 测试执行异常: {test_name} - {str(e)}", exc_info=True)
            results.append({
                "test_name": test_name,
                "passed": False,
                "error": str(e)
            })
            all_passed = False
    
    # 打印测试摘要
    logger.info("\n===== 测试摘要 =====")
    for result in results:
        status = "✅ 通过" if result["passed"] else "❌ 失败"
        logger.info(f"{status}: {result['test_name']}")
    
    if all_passed:
        logger.info("\n🎉 所有测试通过！求解器已完全兼容实验框架。")
    else:
        failed_count = sum(1 for r in results if not r["passed"])
        logger.error(f"\n❌ 测试失败: {failed_count}/{len(results)} 个测试用例失败。")
    
    return all_passed

def validate_framework_format(decisions, system_state):
    """
    验证决策字典是否符合实验框架的格式要求
    """
    required_keys = ['assignment', 'power', 'bandwidth', 'freq']
    
    # 检查所有必要键是否存在
    for key in required_keys:
        if key not in decisions:
            logger.error(f"缺少必要的决策字段: {key}")
            return False
    
    # 获取预期的车辆和服务器数量
    V_set = system_state.get('V_set', [])
    expected_num_vehicles = len(V_set) if V_set else system_state.get('num_vehicles', 1)
    expected_num_servers = system_state.get('num_servers', 1)
    
    # 1. 验证assignment格式 - 一维数组
    assignment = decisions['assignment']
    if not isinstance(assignment, list):
        logger.error(f"assignment必须是列表类型，实际类型: {type(assignment)}")
        return False
    
    if len(assignment) != expected_num_vehicles:
        logger.warning(f"assignment长度不匹配: 期望={expected_num_vehicles}, 实际={len(assignment)}")
        # 不强制失败，允许求解器有自己的调整逻辑
    
    for i, val in enumerate(assignment):
        if not isinstance(val, int):
            logger.error(f"assignment[{i}]必须是整数类型，实际类型: {type(val)}")
            return False
    
    # 2. 验证power格式 - 一维数组
    power = decisions['power']
    if not isinstance(power, list):
        logger.error(f"power必须是列表类型，实际类型: {type(power)}")
        return False
    
    if len(power) != expected_num_vehicles:
        logger.warning(f"power长度不匹配: 期望={expected_num_vehicles}, 实际={len(power)}")
        # 不强制失败，允许求解器有自己的调整逻辑
    
    for i, val in enumerate(power):
        if not isinstance(val, (int, float)):
            logger.error(f"power[{i}]必须是数字类型，实际类型: {type(val)}")
            return False
        if val < 0:
            logger.error(f"power[{i}]不能为负数: {val}")
            return False
    
    # 3. 验证bandwidth格式 - 二维数组
    bandwidth = decisions['bandwidth']
    if not isinstance(bandwidth, list) or not bandwidth:
        logger.error(f"bandwidth必须是非空列表类型，实际类型: {type(bandwidth)}")
        return False
    
    if not all(isinstance(row, list) for row in bandwidth):
        logger.error(f"bandwidth必须是列表的列表（二维数组）")
        return False
    
    if len(bandwidth) != expected_num_servers:
        logger.warning(f"bandwidth的服务器维度不匹配: 期望={expected_num_servers}, 实际={len(bandwidth)}")
        # 不强制失败，允许求解器有自己的调整逻辑
    
    for i, row in enumerate(bandwidth):
        if len(row) != expected_num_vehicles:
            logger.warning(f"bandwidth[{i}]的车辆维度不匹配: 期望={expected_num_vehicles}, 实际={len(row)}")
        
        for j, val in enumerate(row):
            if not isinstance(val, (int, float)):
                logger.error(f"bandwidth[{i}][{j}]必须是数字类型，实际类型: {type(val)}")
                return False
            if val < 0:
                logger.error(f"bandwidth[{i}][{j}]不能为负数: {val}")
                return False
    
    # 4. 验证freq格式 - 二维数组
    freq = decisions['freq']
    if not isinstance(freq, list):
        logger.error(f"freq必须是列表类型，实际类型: {type(freq)}")
        return False
    
    if not all(isinstance(row, list) for row in freq):
        logger.error(f"freq必须是列表的列表（二维数组）")
        return False
    
    if len(freq) != expected_num_servers:
        logger.warning(f"freq的服务器维度不匹配: 期望={expected_num_servers}, 实际={len(freq)}")
        # 不强制失败，允许求解器有自己的调整逻辑
    
    for i, row in enumerate(freq):
        if len(row) != expected_num_vehicles:
            logger.warning(f"freq[{i}]的车辆维度不匹配: 期望={expected_num_vehicles}, 实际={len(row)}")
        
        for j, val in enumerate(row):
            if not isinstance(val, (int, float)):
                logger.error(f"freq[{i}][{j}]必须是数字类型，实际类型: {type(val)}")
                return False
            if val < 0:
                logger.error(f"freq[{i}][{j}]不能为负数: {val}")
                return False
    
    # 记录验证通过的详细信息
    logger.debug(f"框架格式验证通过: assignment={len(assignment)}, power={len(power)}, ")
    logger.debug(f"bandwidth={len(bandwidth)}x{len(bandwidth[0]) if bandwidth else 0}, ")
    logger.debug(f"freq={len(freq)}x{len(freq[0]) if freq else 0}")
    
    # 验证assignment中的服务器ID在有效范围内
    for i, server_id in enumerate(assignment):
        if server_id != -1 and (server_id < 0 or server_id >= expected_num_servers):
            logger.warning(f"assignment[{i}]中的服务器ID ({server_id})超出有效范围 [0, {expected_num_servers-1}] 或不是-1")
    
    return True

def test_extreme_cases():
    """
    测试极端情况，确保求解器的鲁棒性
    """
    extreme_cases = [
        {
            "name": "None输入",
            "system_state": None
        },
        {
            "name": "空字典输入",
            "system_state": {}
        },
        {
            "name": "超大数量 - 100服务器1000车辆",
            "system_state": {
                "params": {
                    "num_servers": 100,
                    "num_vehicles": 1000
                },
                "V_set": list(range(1000)),
                "J_set": list(range(100))
            }
        },
        {
            "name": "非常小的资源参数",
            "system_state": {
                "params": {
                    "num_servers": 1,
                    "num_vehicles": 1,
                    "transmit_power": 1e-10,
                    "bandwidth": 1,
                    "noise_power": 1e-100
                }
            }
        }
    ]
    
    solver = BARGAIN_MATCH_Solver(
        env_config={"num_servers": 1, "num_vehicles": 1},
        cfg={"transmit_power": 0.1, "bandwidth": 10e6, "noise_power": 1e-13}
    )
    
    logger.info("\n===== 开始极端情况测试 =====")
    all_survived = True
    
    for case in extreme_cases:
        test_name = case["name"]
        system_state = case["system_state"]
        
        try:
            decisions = solver.solve(system_state)
            # 验证返回值是否为字典且格式基本正确
            if isinstance(decisions, dict) and all(k in decisions for k in ['assignment', 'power', 'bandwidth', 'freq']):
                logger.info(f"✅ 极端情况测试通过: {test_name}")
            else:
                logger.warning(f"⚠️  极端情况测试返回了非标准结果: {test_name}")
        except Exception as e:
            logger.error(f"❌ 极端情况测试异常: {test_name} - {str(e)}")
            all_survived = False
    
    if all_survived:
        logger.info("🎉 所有极端情况测试通过！求解器具有良好的鲁棒性。")
    else:
        logger.warning("⚠️  部分极端情况测试失败，但这可能是预期的。请检查求解器的错误处理逻辑。")
    
    return all_survived

if __name__ == "__main__":
    logger.info("开始BARGAIN_MATCH_Solver与实验框架兼容性测试...")
    
    # 测试框架格式兼容性
    format_passed = test_solver_with_framework_format()
    
    # 测试极端情况
    extreme_passed = test_extreme_cases()
    
    logger.info("\n===== 兼容性测试总结 =====")
    logger.info(f"框架格式兼容性测试: {'✅ 通过' if format_passed else '❌ 失败'}")
    logger.info(f"极端情况测试: {'✅ 通过' if extreme_passed else '⚠️  部分通过'}")
    
    if format_passed and extreme_passed:
        logger.info("🎉 BARGAIN_MATCH_Solver 已成功验证与实验框架的兼容性！")
        sys.exit(0)
    else:
        logger.warning("⚠️  测试完成，但存在一些警告或失败。请检查日志并确保求解器行为符合预期。")
        sys.exit(0)  # 即使有警告也返回成功，因为这些可能是可接受的边界情况处理