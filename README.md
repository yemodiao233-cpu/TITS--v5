# TITS -v6.0 项目说明文档

## 项目简介
TITS -v6.0 是一个用于比较不同求解器在车辆边缘计算(VEC)环境中性能的实验框架。该项目支持多种求解器的对比测试，可以评估不同算法在资源分配、能量效率、延迟等方面的表现。

### v6.0 更新内容
- 修复了OORAA_Solver中的类型错误，确保正确处理浮点数和字典类型
- 修复了BARGAIN_MATCH_Solver的执行问题，确保所有求解器能正常运行
- 增强了test_results目录的组织结构，确保每个求解器的结果正确保存
- 完善了sensitivity目录，添加了多种参数敏感性分析的结果

## 项目结构

```
TITS -v6.0/
├── copy_summary.py          # 复制摘要文件的脚本
├── generate_tex_tables.py   # 生成TeX表格的脚本
├── logs/                    # 实验结果存储目录
├── main.py                  # 主要实验运行器（支持多求解器对比）
├── plots/                   # 图表输出目录
│   ├── ablation/            # 消融实验图表
│   ├── kpi/                 # KPI性能指标图表
│   └── sensitivity/         # 参数敏感性分析图表
├── requirements.txt         # 项目依赖
├── run_bargain_match.py     # 运行BARGAIN_MATCH_Solver的专用脚本
├── run_experiment.py        # 命令行实验运行工具
├── run_multiple_experiments.py # 运行多组实验的脚本
├── run_ooraa_solver.py      # 运行OORAA_Solver的专用脚本
├── run_plot.py              # 绘图工具
├── scenario.py              # 场景定义
├── simulation_outputs/      # 模拟输出目录
├── solvers/                 # 求解器实现目录
│   ├── A3C_GCN_Seq2Seq_Adapter.py    # A3C-GCN-Seq2Seq适配器
│   ├── BARGAIN_MATCH_Solver.py       # 优化的Bargaining Matching求解器
│   ├── NOMA_VEC_Solver.py            # NOMA使能的多F-AP车辆边缘计算求解器
│   ├── OLMA_Solver_perfect.py        # 完美OLMA求解器（默认）
│   ├── OORAA_Solver.py               # 能效和延迟权衡求解器
│   ├── basesolver.py                 # 求解器基类
│   ├── environment.py                # 环境实现
│   └── ...
├── test_*.py                # 测试文件
├── test_results/            # 测试结果目录
├── utils/                   # 工具函数目录
│   ├── metrics_logger.py    # 指标记录器
│   ├── plot_utils.py        # 绘图工具
│   └── solver_adapter.py    # 求解器适配器
└── 可视化.txt              # 可视化说明
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
实验结果存储在两个主要目录下：
- `logs/`: 默认实验结果目录
- `test_results/`: 测试运行的结果目录，每个求解器有独立的运行目录

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

### 评价指标存储说明

根据`评价指标.md`中定义的关键评价指标，本部分详细说明这些指标在项目中的具体存储位置、格式和获取方式。

#### 1. 主要评价指标及存储位置

##### 1.1 长期平均系统成本 (Overall Long-Term Average Cost)

**存储位置：**
- **主要位置：** 各求解器的`summary.json`文件中的`"C_mean"`字段
- **具体路径示例：**
  - `logs/summary.json` - 整体汇总结果
  - `test_results/[求解器名称]_run_1/summary.json` - 单个求解器测试结果
  - `test_results/[求解器名称]_summary.json` - 求解器全局汇总结果

**数据类型：** 浮点数 (float)

**示例：**
```json
"C_mean": 0.821648354554072
```

##### 1.2 平均端到端时延 (Average End-to-End Delay)

**存储位置：**
- **主要位置：** 各求解器的`summary.json`文件中的`"avg_delay_mean"`字段
- **具体路径示例：** 与长期平均系统成本相同

**数据类型：** 浮点数 (float)

**示例：**
```json
"avg_delay_mean": 0.8701811573036433
```

##### 1.3 队列稳定性 (Queue Stability)

**存储位置：**
- **主要位置：** 各求解器的`summary.json`文件中的`"Avg_queue"`字段
- **具体路径示例：** 与长期平均系统成本相同

**数据类型：** 浮点数 (float)

**示例：**
```json
"Avg_queue": 0.0
```

##### 1.4 单时隙决策时延 (Decision Time per Slot)

**存储位置：**
- **总体统计：** 各求解器的`summary.json`文件中的`"DecisionTime_ms_mean"`字段
- **详细数据：** 各求解器的`per_slot_minimal.csv`文件中的`solve_time_s`列
- **具体路径示例：**
  - `test_results/[求解器名称]_run_1/solvers_[求解器名称]_[求解器名称]/per_slot_minimal.csv`

**数据类型：**
- 浮点数 (float)
- `DecisionTime_ms_mean`以毫秒为单位
- `solve_time_s`以秒为单位

**示例：**
```json
"DecisionTime_ms_mean": 74.99771118164062
```

CSV数据示例：
```csv
solve_time_s
0.06905889511108398
0.09007549285888672
0.08199906349182129
```

#### 2. 成本-时延权衡曲线数据

**存储位置：**
- **主要数据源：** `logs/sensitivity_results.csv`
- **可视化结果：** `plots/sensitivity/sensitivity_V.png`

**数据形式：**
- CSV格式，包含不同V值下的系统成本和时延数据
- 横轴为控制参数V，纵轴分别为系统成本(C_mean)和平均时延(avg_delay_mean)

#### 3. per_slot_minimal.csv 文件格式详解

`per_slot_minimal.csv`文件包含每个时间槽的详细运行数据，是计算各种平均指标的基础数据来源：

**主要字段说明：**
- `slot`: 时间槽编号
- `arrival_tasks`: 到达的任务数
- `completed_tasks`: 完成的任务数
- `dropped_tasks`: 丢弃的任务数
- `effective_bits`: 有效处理的数据比特数
- `energy_srv`: 服务器能耗
- `energy_tx`: 传输能耗
- `queue_mean`: 平均队列长度（用于计算队列稳定性）
- `solve_time_s`: 单时隙决策时间（秒）
- `total_cost`: 当前时间槽的总成本（用于计算长期平均系统成本）

#### 4. 主要性能指标汇总

| 指标名称 | 描述 | 存储位置 |
|---------|------|----------|
| C_mean | 平均成本 | summary.json |
| C_std | 成本标准差 | summary.json |
| E_tx_total | 总传输能耗 | summary.json |
| E_srv_total | 总服务器能耗 | summary.json |
| E_mean | 平均能耗 | summary.json |
| Energy_per_bit | 每比特能耗 | summary.json |
| Energy_per_task | 每个任务的能耗 | summary.json |
| EE_proxy_bit_per_J | 能量效率代理（比特/焦耳） | summary.json |
| avg_delay_mean | 平均延迟均值 | summary.json |
| avg_delay_median | 平均延迟中位数 | summary.json |
| delay_95 | 95%延迟分位数 | summary.json |
| delay_99 | 99%延迟分位数 | summary.json |
| SLA_violation_rate | SLA违反率 | summary.json |
| Avg_queue | 平均队列长度 | summary.json |
| Acc_rate | 接受率 | summary.json |
| Drop_rate | 丢弃率 | summary.json |
| Handover_count_total | 切换总数 | summary.json |
| CPU_util_mean | 平均CPU利用率 | summary.json |
| BW_util_mean | 平均带宽利用率 | summary.json |
| Migration_freq_total | 迁移频率总数 | summary.json |
| Migration_cost_total | 迁移成本总数 | summary.json |
| DecisionTime_ms_mean | 平均决策时间（毫秒） | summary.json |
| DecisionTime_ms_max | 最大决策时间（毫秒） | summary.json |

## 可视化
项目提供了`run_plot.py`脚本用于生成实验结果的可视化图表。图表输出存储在`plots/`目录中，包含以下子目录：

### 敏感性分析结果
`sensitivity`目录包含各种参数敏感性分析的图表和数据：
- **参数F_max**：最大CPU频率参数的敏感性分析 (sensitivity_F_max.csv/png)
- **参数I_bcd**：BCD算法迭代次数参数的敏感性分析 (sensitivity_I_bcd.csv/png)
- **参数P_max**：最大功率参数的敏感性分析 (sensitivity_P_max.csv/png)
- **参数V**：控制参数的敏感性分析 (sensitivity_V.csv/png)
- **参数epsilon**：精度参数的敏感性分析 (sensitivity_epsilon.csv/png)
- **参数weights_E**：能量权重参数的敏感性分析 (sensitivity_weights_E.csv/png)
- **gamma曲线**：gamma参数影响曲线 (gamma_curve.png)

### 消融实验结果
`ablation`目录包含消融实验的图表结果。

### KPI性能指标
`kpi`目录包含关键性能指标的图表结果。

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

4. **如何运行特定求解器的测试？**
   - 使用专用脚本：`python run_ooraa_solver.py` 或 `python run_bargain_match.py`
   - 或使用通用测试文件：`python test_solver.py`

## 注意事项

- 实验运行时间取决于求解器复杂度和指定的时间槽数量
- 使用GPU可以加速A3C_GCN_Seq2Seq_Adapter求解器的运行
- 对于大规模实验，建议适当调整内存使用设置
- 所有求解器已修复并测试，确保能正常运行并生成结果
- 敏感性分析图表已更新，提供更全面的参数影响分析
- 确保使用最新版本的Python (3.8+)以获得最佳兼容性

## 鸣谢 (Acknowledgments)

本研究的顺利完成，离不开多位同仁、导师以及支持者的无私帮助与慷慨支持。在此，作者向所有为本工作提供帮助的人员致以最诚挚的感谢。

* 特别感谢 **王尚鹏** 导师，感谢其在项目的概念构建阶段提供的富有启发性的指导和关键性的学术洞察，这些见解对本研究的方向起到了至关重要的作用。

* 我们衷心感谢 **陈铃** 女士，其在复杂的仿真实验环境搭建和数据处理过程中提供了专业的工具支持和不懈的技术援助，确保了实验结果的准确性。

* 感谢 **张殊赫**，他对本论文算法的逻辑结构和复现思路提出了多次重要修改，代码结构以及健壮性优化，修复了求解器凸近似精确性以及环境仿真问题。
  （同样重要的是，我们衷心感谢 **杨烨宇**。在我遭遇车祸意外的困难时期，给予了我巨大的精神鼓励和坚定支持，使得研究项目能够克服个人挑战，持续推进并最终完成。）
  
* 感谢 **李梓烨**，他对本论文对比算法的实现提供了坚实基础，优化了主函数逻辑，在结果图绘制及求解器日志写入，csv文件输入输出以及管理方面做出了突出贡献。

