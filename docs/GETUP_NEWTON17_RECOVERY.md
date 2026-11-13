# Humanoid Getup：新版 Newton 诊断与冻结先验短训

2026-11-13。本轮先用默认 Humanoid 排查起身失败，不迁移 G1、不重训 diffusion。
短训需本地 main 提交、用户 push 后再远端拉取启动；本文件不是已启动记录。

## 诊断结论

远端 A5000 / WarpSim，Newton 1.7.0、Warp 1.18.0、MuJoCo/MuJoCo-Warp 3.15.0。
以旧 `output/a5000_getup_2048_full_r01/model.pt` 及 iteration 5000 进行只读诊断。
冻结先验为 `output/a5000_getup_prior_joint_400k_scratch_r01/model.pt`，SHA256：
`9e718c27750e30032e557ff5e74b6b411593052618a6abdc5b6205a9a78cbbc0`。

结果位于远端 `output/getup_newton17_audit_20261113_r01/`：
`summary.json`、各策略 `trajectory.csv`、`terminal_scores.json`、`standing_scores.json`。
评分使用旧 RL checkpoint 保存的 SDS 归一化，重复四组噪声；没有更新先验或归一化。

| 真实物理轨迹 | task 均值 | SMP 均值 | 原 0.2/0.8 组合 | 新 0.5/0.5 重算 |
| --- | ---: | ---: | ---: | ---: |
| 最终旧策略，仰卧起点，20s | 0.496234 | 0.574800 | 0.559087 | 0.535517 |
| 最终旧策略，俯卧起点，20s | 0.502376 | 0.549063 | 0.539726 | 0.525720 |
| 中立姿态固定 PD，持续站立 10s | 1.000000 | 0.232532 | 0.386025 | 0.616266 |

两条旧策略轨迹均未成功起身，头部平均高度约 0.105/0.122m。
站立评分排除最初九步参考历史，只评分真实仿真产生的十帧窗口；整个 agent state
指纹在测试前后相同。两种起点及固定站姿只是诊断样本，不是策略成功率或全局最优性证明。

明确问题是：**当前自训练 Getup 先验、已有归一化与原权重组合，给失败贴地行为的奖励
高于已验证的稳定站立行为。** 这不等于 SMP 论文有同样问题，也不能推断唯一根因。
静态仰卧起点 SMP 约 0.995；静态两段终点约 0.554/0.795；中立静态候选约 0.235。
不能仅以 reference 评分高或 diffusion loss 小认定先验支持持续站立。

旧策略向上速度奖励平均约 0.71，而整个回合根部净高度变化仅 -0.039/+0.051m。
将记录轨迹的速度替换成控制周期高度差分后，task 均值仍有约 0.351/0.356。
说明只改速度取样不足以消除奖励冲突；本轮不改此项，不改观测或物理引擎。
奖励在头高 0.6m 处也有明显跃变，这是原配方结构，保留作为后续待验证问题。

本地 WarpSim 实际物理探针（`output/getup_newton17_physics_audit_r02/summary.json`）：
- 中立姿态 PD 可连续站稳约 9.97s，未发现“新版引擎无法站立”的证据。
- 原动作末帧固定 PD 最长仅站稳约 0.53/0.27s，随后倒下。
- 原动作开环 PD 回放未完成起身；开环失败不证明闭环 RL 不可实现。
- 参考姿态有约 5.8/6.7cm 最大地面抬升修正；旧环境已有抬升逻辑，本轮未改。

## 本轮改动及复现边界

1. Getup preflight 同时接受 `newton` 和 `newton_1_7`，旧后端保留。
2. 独立实验 agent `data/agents/smp_getup_humanoid_recovery_agent.yaml`，相对旧 agent
   **只有 task/SMP 权重从 0.2/0.8 改为 0.5/0.5**。旧配置不覆盖。
3. 冻结旧 400k prior；RL 从随机权重开始，不接着训练已失败策略。
4. 保持 SDS scale 8、[22,15,8]、90% 随机参考帧/10% 增广倒地、20s 回合、
   5s 连续站稳评测、原数据/PPO/归一化计划不变。
5. 增加只读物理与冻结评分工具、末帧评分对照；启动器支持 pilot、环境数、
   总样本数和向上对齐完整 rollout，保存实际后端、权重、预算及 prior 哈希。

这是定位并修复失败的受控实验，**不是严格论文参数复现**。若该实验能稳定起身，
下一步再评估如何在原权重下解决先验的站立支持问题。不能承诺改权重必然成功。

提交前验证：17 项 Getup 测试通过（含新版 Newton 实际倒地初始化、历史帧、局部 reset
与成功统计）；本地 8 环境/256 samples 单更新 smoke 正常退出，约 46s。
本地 smoke 复用已有两次更新的小型测试 prior，只验证执行通路，不代表真实先验质量。
新版物理审计工具的可选冻结评分路径也已运行通过，agent state 指纹不变。
远端上表诊断使用真实 400k prior 和旧 RL 权重，不能与本地 smoke 混用。

## Push 后执行

远端 CMD 已激活 WarpSim，工作目录 `C:\Project\MimicKit`。助手先拉取并核对提交、
模型哈希、GPU 空闲状态，再做真实先验 smoke 和负载检查。出现资源问题回退 2048，
逻辑错误先修复；不将本地小型测试 prior 的 smoke 当作真实先验验证。

短训：4096 × 32 × 763 = **100,007,936 samples**，最后 Iteration 762，seed 123。
历史全规模预算保持原值；此次约 1 亿是工程诊断预算，不是论文 Getup 精确规模。
按近期 A5000 吞吐估算 **2～3 小时**，含初始化与周期评测余量；启动后给出实际日历时间。
2048 回退也是 100,007,936 samples，共 1526 更新。

```bat
cd /d C:\Project\MimicKit
python tools/getup/train_getup.py --stage pilot --num_envs 4096 --max_samples 100000000 --engine_config data/engines/newton_1_7_engine.yaml --agent_config data/agents/smp_getup_humanoid_recovery_agent.yaml --seed 123 --out_dir output/humanoid_getup_newton17_recovery_pilot_r01 --check_only
python tools/getup/train_getup.py --stage pilot --num_envs 4096 --max_samples 100000000 --engine_config data/engines/newton_1_7_engine.yaml --agent_config data/agents/smp_getup_humanoid_recovery_agent.yaml --seed 123 --out_dir output/humanoid_getup_newton17_recovery_pilot_r01
```

助手实际启动沿用已验证的独立后台进程方式，保留 UU 会话。启动后不持续监视超过十分钟；
等待用户结束反馈。完成必须核对 runtime EXIT_CODE=0、samples、checkpoint 和有限日志。
验收分仰卧、俯卧、增广倒地起身率/起身时间/连续站稳 5s，不用组合回报替代动作验收。

```bat
python tools/getup/evaluate_getup.py --model_file output/humanoid_getup_newton17_recovery_pilot_r01/model.pt --agent_config data/agents/smp_getup_humanoid_recovery_agent.yaml --engine_config data/engines/newton_1_7_engine.yaml --init starts --episodes 100 --seed 123 --out_dir output/humanoid_getup_newton17_recovery_pilot_r01/eval_starts
python tools/getup/evaluate_getup.py --model_file output/humanoid_getup_newton17_recovery_pilot_r01/model.pt --agent_config data/agents/smp_getup_humanoid_recovery_agent.yaml --engine_config data/engines/newton_1_7_engine.yaml --init augmented --episodes 100 --seed 123 --out_dir output/humanoid_getup_newton17_recovery_pilot_r01/eval_augmented
python tools/getup/evaluate_getup.py --model_file output/humanoid_getup_newton17_recovery_pilot_r01/model.pt --agent_config data/agents/smp_getup_humanoid_recovery_agent.yaml --engine_config data/engines/newton_1_7_engine.yaml --init starts --episodes 2 --visualize --out_dir output/humanoid_getup_newton17_recovery_pilot_r01/view_starts
```

只读复查命令（输出目录须为空或不存在）：

```bat
python tools/getup/diagnose_getup.py --model_files output/a5000_getup_2048_full_r01/model.pt --engine_config data/engines/newton_1_7_engine.yaml --out_dir output/getup_newton17_old_policy_recheck
python tools/getup/audit_physics.py --model_file output/a5000_getup_2048_full_r01/model.pt --out_dir output/getup_newton17_physics_recheck
```
