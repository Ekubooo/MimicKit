# G1 Dodgeball 控制因果测试（2026-11-05）

目的：用已有 full 策略区分持续向膝关节下限施压与改变目标后仍无法离开限位。
这是一轮诊断，不是训练，也不把人工释放动作作为修复后的策略。

## 已核实的起点

- 远端 `C:\Project\MimicKit`，`mimickit-newton`，RTX A5000。
- Newton 1.0.0 / Warp 1.12.0 / MuJoCo 3.5.0 / MuJoCo-Warp 3.5.0.2。
- 旧模型 `output/a5000_g1_dodgeball_2048_full_r01/model.pt`，复用其保存配置与原200k prior。
- 旧诊断首球后左/右膝 near-limit 比例 86.5% / 53.1%，目标越界 85.7% / 57.4%。
- 膝关节物理范围 [-0.087267, 2.8798] rad；原配置动作范围约 ±4.03172 rad。

## 对照设计

每组32环境，每环境只统计第一回合、最多20秒；reference随机相位，seed 123、456。
共3组×2种子×32=192首回合，最多115,200环境控制步，RL更新数为0。

1. baseline：原策略不干预。
2. release：首次发球后，某膝实际角度连续0.5秒低于下限+0.03 rad，且策略目标低于下限，
   将该膝目标改为0.6 rad，保持1秒，再交还策略；每膝每回合最多一次。
3. clip：整回合所有关节目标截断至各自物理角度范围。

baseline和clip也检测相同事件、统计1秒自然响应窗口，但不施加屈膝脉冲。
原有命中、触地和超时终止保持。末端失败帧包含在记录中；提前结束使响应窗口删失，不能当作已观测满1秒的无响应。
相同seed核对初始关节状态；发生干预或不同终止后，不保证后续来球序列逐步相同。

记录30Hz膝关节位置、速度、策略目标、发送目标、实际广义驱动力矩和终止标志。
力矩来自MuJoCo `qfrc_actuator`，按关节名映射并逐步核对位置索引，仅代表每个控制步最后一个物理子步。
现有10Hz全关节/姿态诊断、raw SDS、回合终止统计仍保留。

响应定义：1秒窗口内实际角度超过下限+0.15 rad。
出现响应说明该事件并非不可逆锁死；没有响应仍需结合负载、接触、力矩和窗口删失解释。
clip改善限位比例不等于平衡或躲避改善；人工释放后的存活不等于学会恢复。

## 启动与完成

在远端已激活 `mimickit-newton` 的CMD运行（输出目录必须为空）：

```bat
cd /d C:\Project\MimicKit
python -B tools/g1_dodgeball/run_control_probe.py --smoke --out_dir output/g1_control_probe_smoke_20261105_r01
python -B tools/g1_dodgeball/run_control_probe.py --out_dir output/g1_control_probe_20261105_r01
```

先通过2环境通路检查，再启动完整对照。runner串行运行，不争用GPU；单组20分钟上限，
非零退出、非有限状态或约束容量溢出立即终止整套测试。每组保存实际配置、模型hash和命令。

完成标志：根目录 `runtime.txt` 的EXIT_CODE=0、`summary.json` 的status=completed且六组均完成；
这只表示诊断运行完成，不代表技能通过。
主要反馈：根目录 `summary.json`、`runtime.txt`，各组 `control_probe.json`、`diagnostics.txt`；
需要定位单次事件时读取 `control_trace.csv`、`state_trace.csv`。测试不生成新策略权重。

```bat
type output\g1_control_probe_20261105_r01\runtime.txt
notepad output\g1_control_probe_20261105_r01\summary.json
```

启动后不持续监测，用户通知结束后再分析。首轮实际耗时待远端通路检查估算。
