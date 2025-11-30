# TITS -v5.0 项目说明文档

## 项目简介
TITS -v5.0 是一个用于比较不同求解器在车辆边缘计算(VEC)环境中性能的实验框架。该项目支持多种求解器的对比测试，可以评估不同算法在资源分配、能量效率、延迟等方面的表现。

## 项目结构

```
TITS -v5.0/
├── generate_tex_tables.py    # 生成TeX表格的脚本
├── logs/                     # 实验结果存储目录
├── main.py                   # 主要实验运行器（支持多求解器对比）
├── plots/                    # 图表输出目录
├── requirements.txt          # 项目依赖
├── run_experiment.py         # 命令行实验运行工具
├── run_plot.py               # 绘图工具
├── scenario.py               # 场景定义
├── simulation_outputs/       # 模拟输出目录
├── solvers/                  # 求解器实现目录
│   ├── A3C_GCN_Seq2Seq_Adapter.py    # A3C-GCN-Seq2Seq适配器
│   ├── BARGAIN_MATCH_Solver.py       # 优化的Bargaining Matching求解器
│   ├── NOMA_VEC_Solver.py            # NOMA使能的多F-AP车辆边缘计算求解器
│   ├── OLMA_Solver_perfect.py        # 完美OLMA求解器（默认）
│   ├── OORAA_Solver.py               # 能效和延迟权衡求解器
│   ├── basesolver.py                 # 求解器基类
│   ├── environment.py                # 环境实现
│   └── ...
├── test_*.py                 # 测试文件
├── utils/                    # 工具函数目录
│   ├── metrics_logger.py     # 指标记录器
│   ├── plot_utils.py         # 绘图工具
│   └── solver_adapter.py     # 求解器适配器
└── 可视化.txt               # 可视化说明
```

## 环境配置

### 安装依赖
项目依赖可通过以下命令安装：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- numpy: 数值计算
- scipy: 科学计算
- cvxpy: 凸优化
- torch: 深度学习框架（用于A3C-GCN-Seq2Seq）
- torch_geometric: 图神经网络库
- networkx: 网络分析库
- gym: 强化学习环境

### Python版本要求
建议使用Python 3.8或更高版本。

## 实验执行

### 方法一：使用main.py运行多求解器对比
main.py是一个单文件运行器，支持自动运行多个求解器并比较它们的性能：

```bash
python main.py
```

默认情况下，main.py会运行以下求解器：
- OLMA_Solver_perfect.OLMA_Solver (默认论文求解器)
- A3C_GCN_Seq2Seq_Adapter (A3C-GCN-Seq2Seq适配器)
- NOMA_VEC_Solver (NOMA使能的多F-AP车辆边缘计算求解器)
- OORAA_Solver (能效和延迟权衡求解器)
- BARGAIN_MATCH_Solver (优化后的Bargaining Matching求解器)

你可以在main.py中修改`SOLVERS_TO_RUN`列表来自定义要运行的求解器。

### 方法二：使用run_experiment.py运行指定求解器
run_experiment.py提供了更灵活的命令行接口：

```bash
python run_experiment.py --solver solvers.OLMA_Solver_perfect.OLMA_Solver --slots 100
```

参数说明：
- `--solver`: 求解器类路径（必需）
- `--slots`: 运行的时间槽数量（默认100）
- `--cfg`: 可选的JSON配置文件路径，包含环境、求解器和输出目录配置

### 配置文件示例
如果使用配置文件，可以创建一个JSON文件，例如`config.json`：

```json
{
  "env": {
    "num_servers": 2,
    "num_vehicles": 3,
    "Delta_t": 1.0
  },
  "solver": {},
  "out_dir": "logs"
}
```

然后使用：

```bash
python run_experiment.py --solver solvers.OLMA_Solver_perfect.OLMA_Solver --slots 100 --cfg config.json
```

## 实验结果

### 结果存储位置
实验结果默认存储在`logs/`目录下。

### 结果文件说明

#### 1. 整体摘要文件
- `logs/summary.json`: 包含所有实验的总体性能指标

#### 2. 求解器特定结果
每个求解器的结果存储在单独的目录中，例如：
```
logs/solvers_OLMA_Solver_perfect_OLMA_Solver/
```

该目录包含：
- `OLMA_Solver_output.txt`: 求解器输出日志
- `per_slot_minimal.csv`: 每个时间槽的最小化数据
- `summary.json`: 该求解器的性能指标摘要

### 性能指标说明
实验结果中包含以下主要性能指标：

| 指标名称 | 描述 |
|---------|------|
| C_mean | 平均成本 |
| C_std | 成本标准差 |
| E_tx_total | 总传输能耗 |
| E_srv_total | 总服务器能耗 |
| E_mean | 平均能耗 |
| Energy_per_bit | 每比特能耗 |
| Energy_per_task | 每个任务的能耗 |
| EE_proxy_bit_per_J | 能量效率代理（比特/焦耳） |
| avg_delay_mean | 平均延迟均值 |
| avg_delay_median | 平均延迟中位数 |
| delay_95 | 95%延迟分位数 |
| delay_99 | 99%延迟分位数 |
| SLA_violation_rate | SLA违反率 |
| Avg_queue | 平均队列长度 |
| Acc_rate | 接受率 |
| Drop_rate | 丢弃率 |
| Handover_count_total | 切换总数 |
| CPU_util_mean | 平均CPU利用率 |
| BW_util_mean | 平均带宽利用率 |
| Migration_freq_total | 迁移频率总数 |
| Migration_cost_total | 迁移成本总数 |
| DecisionTime_ms_mean | 平均决策时间（毫秒） |
| DecisionTime_ms_max | 最大决策时间（毫秒） |

## 可视化
项目提供了`run_plot.py`脚本用于生成实验结果的可视化图表。图表输出存储在`plots/`目录中。

此外，项目根目录下的`queue_length_plot.png`和`total_cost_plot.png`是预设的可视化结果示例。

## 测试
项目包含多个测试文件，用于验证各个求解器的功能：
- `test_bargain_solver.py`: 测试BARGAIN_MATCH_Solver
- `test_ooraa_solver.py`: 测试OORAA_Solver
- `test_solver.py`: 通用求解器测试
- `test_framework_compatibility.py`: 框架兼容性测试

## 常见问题

1. **如何添加新的求解器？**
   - 在`solver`目录下创建新的求解器文件
   - 继承`BaseSolver`类并实现`solve`方法
   - 在`main.py`的`SOLVERS_TO_RUN`列表中添加新求解器的路径

2. **如何修改环境参数？**
   - 可以通过配置文件修改环境参数
   - 也可以直接在`environment.py`中修改默认参数

3. **如何更改实验结果存储位置？**
   - 在运行命令时通过配置文件指定`out_dir`
   - 或在`main.py`中修改`OUT_DIR`变量

## 注意事项

- 实验运行时间取决于求解器复杂度和指定的时间槽数量
- 使用GPU可以加速A3C_GCN_Seq2Seq_Adapter求解器的运行
- 对于大规模实验，建议适当调整内存使用设置
