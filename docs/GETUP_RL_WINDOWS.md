# Getup RL：Newton 1.0.0 / Windows / A5000

本轮目标：一个独立策略从仰卧、俯卧及增广倒地姿态起身，并连续站稳至少 5 秒。
使用已完成的联合 **400k diffusion prior**，冻结它进行 SMP reward 评分；RL 从头训练。
暂不添加站立数据、不重训 diffusion、不切换其他任务策略、不升级物理后端。

## 已补齐的部分

| 部分 | 文件 / 行为 |
| --- | --- |
| 独立起身环境 | `mimickit/envs/task_getup_env.py`，注册为 `task_getup` |
| 地面放置 | `mimickit/envs/getup_util.py`，按当前 Humanoid 碰撞形状计算最低点，避免初始化穿地 |
| Agent | `mimickit/learning/smp_getup_agent.py`，复用 SMP/PPO，评测后恢复训练初始化模式 |
| 环境参数 | `data/envs/smp_getup_humanoid_env.yaml` |
| PPO / prior 参数 | `data/agents/smp_getup_humanoid_agent.yaml` |
| 完整预算 | `args/smp_getup_humanoid_args.txt` |
| 训练入口 | `tools/getup/train_getup.py`，预检、控制台日志、起止时间、退出码与版本记录 |
| 精确并行评测 | `tools/getup/evaluate_getup.py`，默认 100 环境各取首回合，合计恰好 100 回合 |
| 反馈汇总 | `tools/getup/collect_feedback.py`，输出便于发送的 `feedback.txt` |

## 参数与论文的对应

按 [SMP v3 附录 D.5](https://arxiv.org/html/2512.03028v3#A4.SS5) 对齐：

- 训练初始化：90% 随机参考动作帧，10% 增广倒地状态；数据中的两段 motion 等权采样。
- task reward：0.3 × 高度项 + 0.7 × 向上速度项。
- 总 reward：0.2 × task + 0.8 × SMP；SDS scale = 8，评分噪声步 `[22,15,8]`。
- head 目标高度 1.2m；head 低于 0.6m 时鼓励 root 向上速度达到 0.25m/s。
- 原 prior 的 10 帧、30Hz、身体顺序和特征接口保持一致，关闭 GSI。

**公式差异必须记录：**论文公式 (30)/(31) 印为 `min(target-current, 0)`，
其字面行为会给低高度/低向上速度满分，与文字“抬高身体”的目标相反。
本实现默认 `getup_reward_mode: deficit`，采用 `max(target-current, 0)`，
这是依据文字目标作出的实现判断，未经作者确认，不宣称逐公式复现。
配置可显式改为 `paper_literal` 做字面公式对照；首轮采用默认值，不混用实验。

论文未公开完整的增广状态分布，本实现从各动作前 0.4 秒采样，加入 0.1rad 关节噪声、
限幅和随机整体朝向，按碰撞几何放置到地面上方 5mm；拒绝 head≥0.6m 或 root≥0.5m 的姿态，
最多尝试 12 次，未接受的槽回退到原动作起点。增广状态速度清零，10 帧历史设为实际初始姿态；
普通参考帧保留动作速度与前序历史。整体旋转按 Euler 分量均匀采样，不是 SO(3) 均匀采样。
这是受参考动作约束的增广，不保证覆盖任意倒地姿态，也未做独立自穿插筛选。

每回合 20 秒；手、膝、头、身体触地以及再次摔倒均不会直接终止。超时重置，
非有限物理状态、root 远离原点超过 20m 或异常高度则安全终止。
这些兜底与下述成功指标属于本项目适配；论文使用 Isaac Gym，本轮使用 Newton 1.0.0。

成功指标：head≥1.2m、root≥0.75m、躯干朝上点积≥0.8、至少一脚有 >1N 地面接触力、
root 速度≤0.5m/s、角速度≤1rad/s，**连续满足 5 秒**。
它只用于评测，不是额外奖励；成功后仍运行到回合结束。
分别记录“曾经站稳 5 秒”与“结束前连续站稳 5 秒”，以识别起身后再次摔倒。
恢复时间指从回合开始到完成这 5 秒保持的时刻。

训练日志字段采用固定宽度兼容的短名称：`Getup_Train_Success_Rate` 为站稳成功率，
`Getup_Train_End_Rate` 为结束仍站稳比例，`Hold_S` 为平均最长连续站稳时间，
`Success_S` 为成功回合平均完成时间，`Aug_Episodes/Aug_Success` 为增广子集数量/成功率，
`Nan_Failures` 为数值失败数；`Getup_Test_*` 为最近一次定期评测。
Train 指标覆盖上一次保存日志以来结束的回合，Test 指标覆盖最近一次测试；
回合数为零时 rate/时间填 0，应先看 Episodes，不能将其理解为已评测后失败。

已有离线检查 log14 显示静态站姿 SDS 明显高于动态参考，存在 task/prior 冲突风险。
这不是禁止 RL 的门槛，也不能直接预测成功率；本轮用当前 prior 建立实际物理基线。
全程 20 秒存活或较高 task return 都不等于成功起身，需结合上述指标和可视观察。

## 规模和预计耗时

| 项目 | 完整训练 |
| --- | --- |
| GPU / 后端 | 单卡 A5000 24GB / Newton 1.0.0，维持现有依赖 |
| 并行数 / rollout | 2048 / 32 步，每次更新 65,536 samples |
| 总预算 | **1,000,013,824 samples，15,259 次更新** |
| 最终日志 Iteration | **15258**，迭代从 0 开始 |
| 保存间隔 | 每 200 迭代及最终；`model.pt` 是最新保存的策略 |
| 耗时估计 | 历史其他任务 13k–17k samples/s 对应 16.3–21.4 小时，安排预留 **20–30 小时** |

这是本项目完整实验预算，未核实为论文 Getup 的精确总样本数。
2048 环境下 PPO 实际 minibatch 随环境数扩大，亦非论文表格的 1024 环境逐参数复现。
Getup 接触计算与定期评测会影响耗时，以远端实测吞吐修正。
Iteration 4000 代表已完成 4001 次更新，约 262M samples，历史吞吐约 4.3–5.6 小时。
定期评测沿用框架行为：2048 环境至少各完成 1 回合，`test_episodes:32` 实际至少评测 2048 回合，
每 200 更新一次；这些测试采样不计入训练 samples。独立工具则精确统计指定回合数。

## 1. 本地提交、推送，远端拉取

代码由用户提交。本地打开 Anaconda Prompt / CMD：

```bat
cd /d D:\ComputerScience\PROJECT\MimicKit
git status --short
git diff --check
git add mimickit/envs/env_builder.py mimickit/envs/getup_util.py mimickit/envs/task_getup_env.py mimickit/learning/agent_builder.py mimickit/learning/smp_getup_agent.py data/envs/smp_getup_humanoid_env.yaml data/agents/smp_getup_humanoid_agent.yaml args/smp_getup_humanoid_args.txt tools/getup/common.py tools/getup/train_getup.py tools/getup/evaluate_getup.py tools/getup/collect_feedback.py tests/test_getup_env.py docs/GETUP_RL_WINDOWS.md docs/GETUP_SHORT_MOTION_ANALYSIS.md plan.md
git diff --cached --stat
git commit -m "Add standalone SMP getup RL training and evaluation"
git push origin main
```

远端打开 **Anaconda Prompt**，以下命令均为 CMD 语法，每行可单独回车执行：

```bat
cd /d C:\Project\MimicKit
git status --short
git pull --ff-only origin main
conda activate mimickit-newton
git rev-parse HEAD
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
dir output\a5000_getup_prior_joint_400k_scratch_r01\model.pt
dir output\a5000_getup_prior_joint_400k_scratch_r01\diffusion_config.yaml
dir output\a5000_getup_prior_joint_400k_scratch_r01\env_config.yaml
python tools/getup/train_getup.py --stage full --check_only
```

`output/` 被 Git 忽略，拉取不会下载 prior；这里必须仍有实际训练的 400k 文件。
`PREFLIGHT OK` 表示路径、特征接口和 CUDA 检查通过，不代表已经完成长训练或质量验证。
若远端有未提交代码，先保留并核对；不要执行 `reset --hard` 清除。
无需重新安装 Newton/Warp，也无需重新训练 diffusion。

## 2. 先做单次更新短测

```bat
python tools/getup/train_getup.py --stage smoke
type output\a5000_getup_8_smoke_r01\runtime.txt
dir output\a5000_getup_8_smoke_r01\model.pt
python tools/getup/collect_feedback.py --run_dir output/a5000_getup_8_smoke_r01
```

8 环境 × 32 步 = 256 samples，只做一次 PPO 更新，验证真实 400k prior 加载、仿真、SDS、反向更新和保存。
短测跳过定期评测；首次编译可能多等数分钟。正常结束应显示 `EXIT_CODE=0`、Iteration 0、Samples 256，
且生成 `model.pt`。这轮不用于判断能否起身。失败时先反馈此目录的 `feedback.txt`。

测试目录和正式目录完全分开；正式训练不会加载短测策略。
工具拒绝写入已有非空目录，重复尝试用新名字，例如：

```bat
python tools/getup/train_getup.py --stage smoke --out_dir output/a5000_getup_8_smoke_r02
```

## 3. 完整训练，从头开始

短测退出正常后，在同一窗口执行：

```bat
python tools/getup/train_getup.py --stage full
```

这一个命令会持续占用当前窗口直至训练结束，并自动记录控制台、版本、起止时间与退出码。
不需要在训练占用的窗口继续输入命令；查看进度另开一个 Anaconda Prompt：

```bat
cd /d C:\Project\MimicKit
conda activate mimickit-newton
python -c "from pathlib import Path; p=Path('output/a5000_getup_2048_full_r01/log.txt'); print('\n'.join(p.read_text().splitlines()[-3:]) if p.exists() else 'Waiting for first saved log')"
dir output\a5000_getup_2048_full_r01\model.pt
```

训练前保持供电、关闭自动睡眠。锁屏或关闭屏幕可继续训练，避免休眠、重启及关闭训练窗口。
GPU CSV 不强制收集。可在另一窗口查看即时负载或每 5 秒刷新，Ctrl+C 仅结束监控：

```bat
nvidia-smi
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv -l 5
```

最终应满足 Samples=1000013824、Iteration=15258、`runtime.txt` 中 EXIT_CODE=0。
中途 Ctrl+C 只保留上次定期保存的策略，不是精确断点恢复；checkpoint 不含优化器/RNG/累计进度。
单独的 `args` 预设也能直接启动，但不会自动生成 wrapper 的控制台/起止时间文件，推荐统一使用上面的入口。

## 4. 查看策略效果

完整训练结束后，从两段原动作的倒地起点查看，不会随机从站立末段开始：

```bat
python mimickit/run.py --arg_file args/smp_getup_humanoid_args.txt --mode test --num_envs 4 --visualize true --model_file output/a5000_getup_2048_full_r01/model.pt --test_episodes 8
```

两种动作起点各两个环境；成功起身不会立刻重置，观察能否维持到 20 秒回合结束。
需要看增广倒地姿态，可使用带可视化的精确评测：

```bat
python tools/getup/evaluate_getup.py --model_file output/a5000_getup_2048_full_r01/model.pt --init augmented --episodes 4 --visualize --out_dir output/a5000_getup_2048_full_r01/view_augmented_seed123
```

可比较中间 checkpoint；例如 Iteration 4000 的模型，把上述 `--model_file` 改为：
`output/a5000_getup_2048_full_r01/int_models/model_0000004000.pt`。
长训练还在运行时不建议同时开启大规模可视化/评测，以免占用同一张卡。

## 5. 两组精确 100 回合评测

```bat
python tools/getup/evaluate_getup.py --model_file output/a5000_getup_2048_full_r01/model.pt --init starts --episodes 100 --seed 123 --out_dir output/a5000_getup_2048_full_r01/evaluation_starts_seed123
python tools/getup/evaluate_getup.py --model_file output/a5000_getup_2048_full_r01/model.pt --init augmented --episodes 100 --seed 123 --out_dir output/a5000_getup_2048_full_r01/evaluation_augmented_seed123
type output\a5000_getup_2048_full_r01\evaluation_starts_seed123\summary.txt
type output\a5000_getup_2048_full_r01\evaluation_augmented_seed123\summary.txt
```

每组使用 100 个并行环境，每个只统计首回合，含失败，不用框架的回合数向上取整。
starts：仰卧/俯卧各 50 次，采用原 motion 第一帧及其原始速度。
确定性策略在重复相同起点时可能产生相同结果；这组验证两种已知姿态，不代表 100 种独立倒地状态。
augmented：各 motion 种子各 50 次并随机化倒地姿态，速度为零，用于检查泛化；
必要时再换 seed456，输出目录也换名。mixed 仅作为训练分布诊断，不用于代替倒地起点成功率。
`summary.txt` / `evaluation.json` 含总体和分 motion 成功率，`episodes.csv` 含每回合明细。
评测 task return 不是加上 SMP 的训练总 reward，不能直接与 Train_Return 横比。

## 6. 给我的反馈清单

```bat
python tools/getup/collect_feedback.py --run_dir output/a5000_getup_2048_full_r01
notepad output\a5000_getup_2048_full_r01\feedback.txt
notepad output\a5000_getup_2048_full_r01\evaluation_starts_seed123\summary.txt
notepad output\a5000_getup_2048_full_r01\evaluation_augmented_seed123\summary.txt
```

请发：

1. `output/a5000_getup_2048_full_r01/feedback.txt`：含版本、实际配置、起止时间、退出码、日志摘要和控制台尾部。
2. 两组评测的 `summary.txt`；若有错误，再发对应 `console.txt` 尾部。保留完整 `log.txt`，需要时提供。
3. 目测：仰卧/俯卧分别能否起身、是否持续站稳、是否反复躺下再起、是否弹跳/扭曲；可附短视频。

无需上传 `model.pt` 或整份 GPU CSV。完整模型和阶段 checkpoint 留在远端；
不要因最终模型效果一般就覆盖本轮数据，中间模型可能更好。

## 本地验证边界

本地仅用相同结构的 2 次更新短测 prior 验证运行流程，实际 400k 权重仍在 A5000。
本地流程通过不证明 400k reward 下的策略质量、A5000 吞吐或随机倒地恢复率。
远端必须先完成上述真实权重 smoke，再启动完整训练。

2026-10-20 本地验证：6 项 Getup 单元检查、2 项真实 Newton 重置/历史检查，以及原有
5 项 prior 特征契约检查通过；8 环境单次 PPO 更新成功，保存后重新加载进行 8 环境增广评测、
100 环境各首回合评测成功，后者严格输出 100 行回合数据、两种 motion 各 50 行。
训练反馈汇总已验证日志列数一致且无非有限值。测试策略不具备起身能力，不作为训练结果发布。
本地文件位于 Git 忽略的 `output/getup_rl_local_check/`，不会混入远端完整训练配置或提交。
