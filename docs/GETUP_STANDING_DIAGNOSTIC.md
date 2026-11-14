# Getup 分阶段诊断：先学站稳

2026-11-14。目标仍是默认 Humanoid 完整起身；本轮只拆分并检验站稳子问题。
代码准备后在 main 提交，用户 push 后助手远端拉取启动，不提前部署未推送源码。

## 新证据

前一轮 4096 环境、100,007,936 samples 于 11-13 21:11 正常结束，耗时 3960.78s。
独立评测仰卧 0/50、俯卧 0/50、增广倒地 0/100，全部未连续站稳 5s。
远端结果：`output/humanoid_getup_newton17_recovery_pilot_r01/result_review_20261114.json`。

进一步只读探针加载同一最终策略，从两段参考动作的 25%、50%、75%、95% 和中立站姿
分别运行 20s（共 10 条确定性诊断轨迹，不作为统计成功率估计）。所有轨迹最大连续
站稳时间均为 0。95% 处初始头高约 1.275/1.361m，中立头高约 1.342m，最后均倒下。
prior、策略和归一化指纹前后不变。证据：
`output/getup_phase_probe_20261114_r01/summary.json`。
此前固定中立 PD 能保持站稳 10s，故优先分离“策略保持站立”与“从地面探索起身”。

## 本轮实验

新初始化仅用于诊断配置：中立站姿，关节高斯扰动 std=0.03rad，限制在坐标边界内，
重新计算全身地面抬升，根部/关节初速度为 0，历史十帧均为相同静态状态。
20s 回合、5s 连续站稳标准、原高度/速度任务公式不变，倒下仍不提前终止。

两组从相同 seed=123 的随机 RL 权重分别训练，不互相加载训练结果：

| 组 | task/SMP 权重 | 用途 |
| --- | --- | --- |
| task_only | 1.0 / 0.0 | 判断无先验奖励时控制策略是否能学会站稳 |
| frozen_smp | 0.5 / 0.5 | 判断加入现有冻结先验后的影响 |

两组均加载同一旧 400k prior 并计算 SDS，但 task_only 的 SMP 得分不进入总奖励。
不训练/微调 diffusion，不修改特征维数、SDS scale=8 或 [22,15,8]。
task_only 是排障对照，不能将它成功称为 SMP 起身成功。

Getup agent 新增绝对 minibatch：actor=4096、critic=2048。此前 `actor_batch_size: 4`
会乘以环境数，4096 环境时实际为 16384。当前 actor 对齐论文表 8 给出的绝对值；
critic 2048 延续旧配置在 1024 环境下的比例含义，不能宣称论文单独指定此值。
4096×32 rollout 与论文的 1024×32 仍不同。只在 Getup agent 中启用该选项，其他 PPO
任务保留原行为；若通路测试 rollout 不足一个 minibatch，则上限取实际 rollout。

每组 4096×32×77 = **10,092,544 samples**，两组共 20,185,088 samples，最后 iteration=76。
输出间隔 50。预估两组连同评测 **30～60 分钟**，实际 minibatch 更新更多，启动后用
远端吞吐修正；2048 回退总预算相同。不得据此前约 1 小时/亿简单线性保证完成时间。

训练后每组分别评测：
- 带关节扰动的 standing 初始化，100 回合；
- 两段参考动作 90%～100% 的 reference 初始化，100 回合；
- 仰卧/俯卧 starts，合计 100 回合。

分别保存指标，绝不把 standing 成功率混入完整起身成功率。
站稳成功且回合结束仍稳定的比例达到 90%，作为后续近站立训练的工程准入标准。
如果 task_only 成功而 frozen_smp 失败，优先诊断先验冲突；两者均失败则排查控制、
动作探索和优化；两者均成功但 late_reference 失败，再推进参考动作后半段课程。
此 runner **不会自动开始完整起身或额外加训**，完成后等待用户反馈并分析结果。

## 启动与复查

远端 CMD 已激活 WarpSim；先核对新提交及真实 prior 哈希，再运行单更新检查。
正式成对诊断由助手按既有 WMI 隐藏后台方式启动，保留 UU 会话。

```bat
cd /d C:\Project\MimicKit
python tools/getup/standing_diagnostic.py --num_envs 4096 --max_samples 10000000 --out_dir output/humanoid_getup_standing_pair_r01
```

总状态及分组结果在 `output/humanoid_getup_standing_pair_r01/summary.json`，
每组的实际样本数、prior 哈希、有效 minibatch、初始化类型在各自 `run_manifest.json`。
不覆盖以前输出，不使用失败策略续训。

可独立分阶段评测任何兼容的旧策略，例如：

```bat
python tools/getup/evaluate_getup.py --model_file output/humanoid_getup_newton17_recovery_pilot_r01/model.pt --engine_config data/engines/newton_1_7_engine.yaml --agent_config data/agents/smp_getup_humanoid_recovery_agent.yaml --init reference --phase_range 0.9 1.0 --episodes 100 --out_dir output/getup_old_late_reference_check
```

## 本地验证

WarpSim / Newton 1.7 / Warp 1.18 / MuJoCo 3.15：20 项测试通过，包括真实 CUDA
初始化、局部重置、历史一致性、参考 phase 隔离以及固定 PD 站稳 6s。
新 standing 配置已完成 8 环境、256 samples 的一次训练更新，EXIT0，指标有限。
随后两回合 reference 评测正常完成，实际初始化 phase 为 0.98468 / 0.95990，
均明确标记为非 standing 初始化。此处使用本地极短训练的 fixture prior，
仅验证执行通路，不能据此判断真实 400k prior 或策略质量。
远端启动前仍须核对真实 prior 哈希并做单更新检查。

## 复现边界

论文参考：[D.5 Getup](https://arxiv.org/html/2512.03028v3#A4.SS5)、
[公开 SMP 代码](https://github.com/xbpeng/MimicKit/blob/main/docs/README_SMP.md)。
原 Getup 配置和旧 Newton 后端仍保留。新增站立初始化、扰动、课程阶段检查和上述
两种奖励权重均是诊断适配，不是论文原版完整训练。最终仍须回到完整倒地分布验收。
