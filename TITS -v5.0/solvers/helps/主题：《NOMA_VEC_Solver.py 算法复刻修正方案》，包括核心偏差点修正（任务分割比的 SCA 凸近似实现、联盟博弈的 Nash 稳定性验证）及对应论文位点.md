# 主题：《NOMA_VEC_Solver.py 算法复刻修正方案》，包括核心偏差点修正（任务分割比的 SCA 凸近似实现、联盟博弈的 Nash 稳定性验证）及对应论文位点

# [NOMA_VEC_Solver.py](NOMA_VEC_Solver.py) 算法复刻修正方案文档

## 一、文档说明

本文档针对 `NOMA_VEC_Solver.py` 与论文《Energy-Efficient Cooperative Task Offloading in NOMA-Enabled Vehicular Fog Computing》算法的偏差点，提供逐处详细修改方案，每个修改位点均标注对应论文公式及章节，确保代码完全复刻论文逻辑。

## 二、核心偏差点修正方案

### 2.1 任务分割比的SCA凸近似实现（补充论文Algorithm 1关键逻辑）

#### 偏差描述

代码未按论文公式（25）-（31）实现非凸项  $T_{TUE,\hat{f}_{n}}^{n,trans}$  的凸近似，直接依赖优化器求解，缺失SCA迭代近似过程。

#### 修改方案

在 `_task_splitting_optimization` 方法中添加SCA凸近似模块，步骤如下：

```Python

```

#### 对应论文位点

- 公式（24）：非凸项  $T_{TUE,\hat{f}_{n}}^{n,trans}$  拆分为  $y_1(u_n)y_2(u_n)$ ；

- 公式（25）： $T_{TUE,\hat{f}_{n}}^{n,trans}$  的凸近似表达式  $\tilde{T}_{TUE,\hat{f}_{n}}^{n,trans}(u_n;u_n^{(k)})$ ；

- 公式（31）：总能耗的凸近似目标函数  $\tilde{E}_{TUE}^{n,trans}$ ；

- Algorithm 1：SCA+内点法的迭代求解流程。

### 2.2 联盟博弈的Nash稳定性验证（修正论文Algorithm 2逻辑）

#### 偏差描述

代码简化了联盟博弈的“联盟划分”和“效用函数”定义，未完全复刻论文中“单TUE交换→能耗降低→迭代直至Nash稳定”的逻辑。

#### 修改方案

重写 `_user_association_optimization` 方法，补充联盟划分和效用函数：

```Python

```

#### 对应论文位点

- Section IV-C：联盟博弈定义（玩家TUE、联盟结构U、效用函数Γ(U)）；

- Definition 1：交换规则（TUE单边移动后能耗降低则执行交换）；

- Definition 2：Nash稳定（无TUE可通过单边交换降低能耗）；

- Proposition 3：Algorithm 2有限迭代收敛至Nash稳定；

- Algorithm 2：联合任务分配与用户关联的迭代流程。

### 2.3 辅助F-AP负载比例分配（修正论文公式12-14逻辑）

#### 偏差描述

代码中`Cf_tot`（F-AP总负载）为固定值，未按论文公式（12）实时计算，导致辅助F-AP的资源分配不符合“按负载比例分配”的假设。

#### 修改方案

在 `_calculate_total_energy` 中补充 `Cf_tot` 的动态计算：

```Python

```

#### 对应论文位点

- 公式（12）：F-AP总负载  $C_f^{tot} = \sum_{n \in \mathcal{N}} \left(I_f^n v_n + \sum_{f' \in \mathcal{G}_f} I_{f'}^n a_{n f'}\right) D_n C_n$ ；

- 公式（13）：F-AP计算延迟  $T_f^{n,comp} = \frac{C_f^{tot}}{\delta_f}$ （隐含在能耗计算中，负载影响资源分配）；

- 公式（14）：F-AP计算能耗  $E_f^{n,comp} = \kappa \frac{C_{f,n}}{C_f^{tot}} (\delta_f)^2 C_{f,n}$ ；

- 公式（18）：辅助F-AP计算能耗  $E_f^n = E_f^{n,comp}$ 。

### 2.4 延迟约束显式验证（补充论文公式19d-19f）

#### 偏差描述

代码未单独验证各子任务的延迟约束，可能出现能耗最优但延迟超标的情况，不符合论文优化问题的约束条件。

#### 修改方案

在 `_task_splitting_optimization` 的约束中添加延迟验证：

```Python

```

#### 对应论文位点

- 公式（19d）：TUE本地计算延迟约束  $T_{TUE}^{n,comp} \leq t_d$ ；

- 公式（19e）：IUE子任务延迟约束  $T_{TUE,IUE}^{n,trans} + T_{IUE}^{n,comp} \leq t_d$ ；

- 公式（19f）：F-AP子任务延迟约束  $T_{TUE,\hat{f}_n}^{n,trans} + (1-I_f^n)T_{\hat{f}_n,f}^{n,trans} + T_f^{n,comp} \leq t_d$ ；

- 公式（6）：主F-AP到辅助F-AP的光传输延迟  $T_{\hat{f}_n,f}^{n,trans} = \frac{a_{nf}D_n}{R_e}$ 。

## 三、其他优化建议（提升与论文的贴合度）

### 3.1 TUE与IUE预先配对（论文Section III-A假设）

在 `__init__` 中添加IUE-TUE配对逻辑，符合论文“每个TUE预先配对一个IUE”的假设：

```Python

```

### 3.2 功率分配中un的动态初始值（论文Proposition 1可行域）

在 `_optimal_power_allocation` 中按论文Proposition 1计算un的初始值，而非固定0.25：

```Python

```

#### 对应论文位点

- Proposition 1：un的可行域  $0 < u_n < \frac{B t_d}{\frac{D_n}{\log_2(H_2 P_n + 1)} + \frac{B C_n}{\delta_{IUE}^n}}$ ；

- Appendix A：H2的定义  $H_2 = \frac{h_{IUE}^n}{\sigma^2}$ 。

## 四、修改后验证建议

1. **能耗对比验证**：运行修改后的代码，对比论文Fig.3（不同TUE数量下的能耗），确保在TUE=50时，能耗比单F-AP基线低约37%（符合论文1-143描述）；

2. **延迟约束验证**：输出各子任务的延迟，确保所有延迟≤t_d（2s），符合论文Fig.9的延迟保障效果；

3. **Nash稳定验证**：打印最终联盟结构，确认无TUE可通过单边切换F-AP降低能耗（符合Proposition 3）。

通过上述修改，代码可完全复刻论文的核心算法逻辑，实现“功率分配→任务分割→用户关联”的联合优化，且满足所有约束条件与性能指标。
> （注：文档部分内容可能由 AI 生成）