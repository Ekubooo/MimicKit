# G1 29DoF 回旋踢：SMP prior → Newton RL → 评测与反馈

本轮目标是 G1 回旋踢仿真策略。已有 G1 模型、网格和动作，无需先下载机器人；
尚无 G1 SMP diffusion 权重。现有 `deepmimic_g1_spinkick_model.pt` 是其他算法的 RL 示例，不能当作 SMP prior。
保留 Newton 1.0.0 / Warp 1.12.0 / MuJoCo-Warp 当前环境；不在本轮升级后端或部署真机。

## 1. 为什么需要训练 G1 prior

之前 Newton 的 Humanoid 四任务直接使用公开冻结权重：回旋踢 `smp_prior_spinkick.pt`，
Location / Steering / Dodgeball 默认共用 `smp_prior_lafan.pt`。换物理引擎时并未重新训练它们。
上游说明后者是 LaFAN1 prior；单动作配置指向 `humanoid_spinkick.pkl`。
另有 `smp_prior_locomotion.pt` 文件，但不是当前三任务默认加载的权重。
公开多动作训练 YAML 指向 `dataset_humanoid_locomotion.yaml`（包括跑走/SIE 片段等），
不能仅凭该默认 YAML 就宣称 LaFAN 权重的完整原始训练清单或实际迭代数已经核实。
来源：https://github.com/xbpeng/MimicKit/blob/main/docs/README_SMP.md

### 本轮按什么规模对齐（2026-10-29 核实）

不只读取 YAML：已直接读取下载的公开 prior 权重内的 `ema_dmodel.step`，
结合源码每次 optimizer 更新后调用一次 EMA.update()，得到以下记录：

| 公开权重 | EMA step | 归一化统计帧数 | 对应 10 帧统计窗口数 |
| --- | --- | --- | --- |
| smp_prior_spinkick.pt | **50,000** | 100,000 | **10,000** |
| smp_prior_lafan.pt | 200,000 | 200,000 | 20,000 |
| smp_prior_locomotion.pt | 200,000 | 200,000 | 20,000 |

因此 G1 回旋踢默认 50k 对齐的是公开回旋踢权重记录的完整更新数，并非缩短的试训预算。
统计 count 按帧计数，不能误读成 100k 个窗口。
论文 v3 第 6.2 节整体描述 400k–800k 更新，与公开权重记录存在差异；
不能同时宣称 50k 与该正文区间逐项一致。需要论文区间实验时可使用下文独立 400k 选项。
论文来源：https://arxiv.org/html/2512.03028v3#S6.SS2

RL 严格按公开 `data/logs/smp_humanoid_spinkick_log.txt` 最后一条记录对齐：
**380,239,872 samples**。原日志 4096×32×2901，G1 改用 2048×32×5802，
所以最终 Iteration=5801。原日志最后输出是 Iteration2900，不能凭它编造未记录的最终预算。
此前拟定的 400,031,744 是约 4 亿的项目预算，本轮按用户要求改为公开可核实的精确样本数。
权重/日志 SHA256 与计数保存在 `tools/g1_spinkick/reference_budget.json`，并复制到每轮输出目录。

本轮 G1 使用 29 个驱动关节、31 个身体节点、10 帧 × 210 特征 = 2100 维窗口；
原 Humanoid prior 窗口为 1140 维，无法直接加载。这里准备的是全身 G1 的回旋踢专用 prior，
不是只有腿部的 10/12 动作行走策略，也不是一份覆盖 G1 所有技能的通用 prior。
以后其他三个任务可另训并共享兼容的 G1 locomotion prior，不要求每个任务单独训 diffusion。

## 2. 动作数据的一项适配，先看清楚

原动作 `data/motions/g1/g1_spinkick.pkl`：60Hz，78 帧，约 1.283s，35 个原始姿态分量。
检查发现 `left_ankle_roll_joint` 全部 78 帧低于模型下限，最大超限 0.46814rad。
`workflow.py --stage check` 自动生成一个派生副本：

`output/g1_spinkick_assets/g1_spinkick_joint_limited.pkl`

采用逐帧最小限位投影（clip），保留原文件和模型真实限位。当前只有左踝横滚被改变，
其全段被限幅到下限 -0.2618rad，**该关节原有变化因此丢失**。
修改的是参考动作；策略仍控制完整 29DoF，没有锁死该关节，也没有放宽模型限位。
这是透明、可追溯的首轮数据修整，不是完整 IK 重定向，也不证明支撑接触或动力学已经可行。
先预览，再做短程 RL；若修整明显破坏支撑或动作，应重新做约束重定向，不直接追加完整预算。
原始/派生 SHA256 和修改量写入 `output/g1_spinkick_assets/preparation.json` 及每轮 `motion_audit.json`。
工具会保留已被用户另行修改的派生文件并报错，不静默覆盖。

## 3. 预算与时间

以下为单卡 A5000 24GB 估算，G1 长训练尚无本机实测。prior 的输出开销含每 2000 更新导出 GIF。

| 阶段 | 规模 | 最后 Iteration | 时间预留 |
| --- | --- | --- | --- |
| prior-smoke | 2 次更新，batch512 | 1 | 首次约数分钟 |
| prior（默认） | **50,000 次更新，batch512，lr1e-4** | 49999 | **20–45 分钟** |
| rl-smoke | 8 环境 × 32 步，1 次 PPO 更新 | 0 | 首次约数分钟 |
| pilot | 2048 环境，**26,214,400 samples / 400 更新** | 399 | **0.5–1.5 小时** |
| full | 2048 环境，**380,239,872 samples / 5802 更新** | 5801 | **8–16 小时** |

50k 是公开 `tinymdm_single_clip.yaml` 的完整默认预算，适合作为本轮单动作基线；
不是把 Getup 的 400k 方案机械复用，也不称为论文所有实验的统一完整规格。
可选 400k prior 约 3–6 小时；800k 约 6–12 小时，属于未测外推，并非必须。
数据动作短不会免去这些更新；50k×512 = 25,600,000 次重复窗口抽样，不是同等数量独立动作。
prior 不运行物理仿真，无 2048 环境参数。

RL 对齐公开回旋踢日志的全部已记录样本，不保证 G1 达到同等效果。
pilot 和 full 都从头初始化 RL；full 不加载 pilot，二者预算不混计。
若 pilot 出现持续僵直、重复摔倒或静止得分过高，先反馈并排查，暂不扩大到 full。

## 4. 同步与依赖

本地 Anaconda Prompt / CMD，由用户提交推送：

```bat
cd /d D:\ComputerScience\PROJECT\MimicKit
git status --short
git add args/smp_spinkick_g1_args.txt data/envs/smp_spinkick_g1_env.yaml data/envs/view_motion_spinkick_g1_env.yaml data/agents/smp_spinkick_g1_agent.yaml tools/diffusion_model/config/tinymdm_spinkick_g1.yaml tools/diffusion_model/train_tinymdm.py tools/g1_spinkick/workflow.py tools/g1_spinkick/evaluate.py tools/g1_spinkick/reference_budget.json tests/test_g1_spinkick.py docs/G1_SPINKICK_WINDOWS.md plan.md
git diff --cached --stat
git commit -m "Add G1 SMP spinkick prior and Newton RL workflow"
git push origin main
```

远端 Anaconda Prompt / CMD：

```bat
cd /d C:\Project\MimicKit
git pull --ff-only origin main
conda activate mimickit-newton
set MPLBACKEND=Agg
set PYTHONUTF8=1
python tools/g1_spinkick/workflow.py --stage check
```

预检应显示 G1 29DoF、2100 特征、约 1.283s、修整后超限接近 0，最后 `G1 ASSETS OK`。
模型/网格/motion/prior/output 被 Git 忽略，拉取不下载它们。
缺文件时，从已下载的 MimicKit 原数据包保留目录复制完整 `data/assets/g1/` 和
`data/motions/g1/g1_spinkick.pkl`；本地这些文件已存在。原数据包入口见 README.md 的 Data 下载链接。
例如把本机 G1 目录拷到 U 盘后，远端可执行（将 E: 替换为实际盘符）：

```bat
xcopy E:\MimicKitG1\assets\g1 data\assets\g1\ /E /I
if not exist data\motions\g1 mkdir data\motions\g1
copy E:\MimicKitG1\motions\g1_spinkick.pkl data\motions\g1\g1_spinkick.pkl
```

维持现有 Conda 环境，无需重新安装整套依赖。

## 5. 先播放原动作与修整动作

原始参考（运动学播放，不是 RL）：

```bat
python mimickit/run.py --mode test --num_envs 1 --engine_config data/engines/newton_engine.yaml --env_config output/g1_spinkick_assets/view_original.yaml --visualize true --test_episodes 1
```

修整副本：

```bat
python mimickit/run.py --mode test --num_envs 1 --engine_config data/engines/newton_engine.yaml --env_config data/envs/view_motion_spinkick_g1_env.yaml --visualize true --test_episodes 1
```

检查左脚是否严重扭曲、穿地，整体回旋踢是否仍可辨认；kinematic 播放不证明真能在物理中完成。

## 6. diffusion prior：短测 → 完整默认预算 → 独立采样

```bat
python tools/g1_spinkick/workflow.py --stage prior-smoke
type output\a5000_g1_spinkick_prior_smoke_r01\runtime.txt
python tools/g1_spinkick/workflow.py --stage prior
type output\a5000_g1_spinkick_prior_50k_r01\runtime.txt
python tools/g1_spinkick/workflow.py --stage sample
explorer output\a5000_g1_spinkick_prior_50k_r01_samples\samples
```

每个命令结束再执行下一行；短测退出码应为 0。正式 prior 使用全新随机初始化，不加载短测模型。
50k 正常结束为 Iteration49999、EXIT_CODE=0；权重和配置均保存在 prior 输出目录。
后续 RL/采样入口还会核对权重里的 EMA.step 是否达到配置预算，阻止把未训完的 prior 当成完整结果。
独立采样固定 seed0，产生 16 个 GIF/运动文件；每个片段只有 10 帧/0.3 秒，不是完整回旋踢。
训练中 sample 文件会被最新导出覆盖，独立 sample 阶段另存目录。
Loss 下降、GIF 看似正常都不替代物理 RL 验证。

若明确选择更长 prior，从头 400k 可用：

```bat
python tools/g1_spinkick/workflow.py --stage prior --prior_iterations 400000 --out_dir output/a5000_g1_spinkick_prior_400k_r01
```

后续 RL 训练命令需加 `--prior_dir output/a5000_g1_spinkick_prior_400k_r01`；
evaluate 会优先从策略输出目录的 agent_config.yaml 自动推断 prior 路径，也可显式传入。
它不是在 50k 上追加 350k。默认文档其余命令统一使用 50k。

## 7. RL：单次更新 → 短程学习检查

```bat
python tools/g1_spinkick/workflow.py --stage rl-smoke
type output\a5000_g1_spinkick_rl_smoke_r01\runtime.txt
python tools/g1_spinkick/workflow.py --stage pilot
```

RL 短测 256 samples、Iteration0、EXIT_CODE=0 仅验证仿真、SDS、更新和保存。
pilot 完成后先查看真实动作：

```bat
python mimickit/run.py --arg_file args/smp_spinkick_g1_args.txt --mode test --num_envs 4 --visualize true --agent_config output/a5000_g1_spinkick_2048_pilot_r01/agent_config.yaml --model_file output/a5000_g1_spinkick_2048_pilot_r01/model.pt --test_episodes 8
python tools/g1_spinkick/evaluate.py --model_file output/a5000_g1_spinkick_2048_pilot_r01/model.pt --init random --out_dir output/a5000_g1_spinkick_2048_pilot_r01/evaluation_random_seed123
python tools/g1_spinkick/workflow.py --stage feedback --run_dir output/a5000_g1_spinkick_2048_pilot_r01
```

第一次 G1 实验建议在 pilot 后反馈动作及评测，以免重复长时间训练无效策略。
四环境可视运行仍采用参考随机相位初始化，不能当作从静态站立开始的部署验证。

## 8. 完整 RL

pilot 学习趋势及动作合理时执行：

```bat
python tools/g1_spinkick/workflow.py --stage full
```

一条命令占用训练窗口，自动保存 `console.txt`、`runtime.txt`、`run_manifest.json`、配置及模型。
不要在这个窗口追加输入；不休眠的锁屏/关闭屏幕可继续训练。
新窗口查看进度与可选 GPU 负载：

```bat
cd /d C:\Project\MimicKit
conda activate mimickit-newton
python -c "from pathlib import Path; p=Path('output/a5000_g1_spinkick_2048_full_r01/log.txt'); print('\n'.join(p.read_text().splitlines()[-3:]) if p.exists() else 'Waiting for first saved log')"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5
```

监控窗口 Ctrl+C 只停止监控。无需提供 GPU CSV。
训练每 100 迭代及最终保存 `model.pt` 和 `int_models/model_XXXXXXXXXX.pt`。
结束标准为 Iteration5801、Samples380239872、EXIT_CODE=0。
checkpoint 不含完整优化器/RNG/计数，不是精确断点恢复。
所有阶段拒绝覆盖非空目录；重试需用新的 `--out_dir` 并同步修改后续路径。

**重要日志口径：**此任务 `task_reward_weight=0`、`smp_reward_weight=1`，环境 task reward 本身为 0。
框架 Train_Return/Test_Return 累计的是 task reward，因此显示 0 是预期行为；
主要看 Smp_Reward_Mean、Sds_Loss_Mean、回合长度和实际动作。
默认定期 test 的 `test_episodes:32` 在 2048 环境下实际至少每环境一回合，
即至少 2048 回合，测试样本不计入训练 samples。

## 9. 完整模型可视运行和精确 100 回合评测

```bat
python mimickit/run.py --arg_file args/smp_spinkick_g1_args.txt --mode test --num_envs 4 --visualize true --agent_config output/a5000_g1_spinkick_2048_full_r01/agent_config.yaml --model_file output/a5000_g1_spinkick_2048_full_r01/model.pt --test_episodes 8
python tools/g1_spinkick/evaluate.py --model_file output/a5000_g1_spinkick_2048_full_r01/model.pt --init starts --episodes 100 --out_dir output/a5000_g1_spinkick_2048_full_r01/evaluation_starts_seed123
python tools/g1_spinkick/evaluate.py --model_file output/a5000_g1_spinkick_2048_full_r01/model.pt --init random --episodes 100 --out_dir output/a5000_g1_spinkick_2048_full_r01/evaluation_random_seed123
```

每组 100 环境各统计首回合，恰好 100 回合。starts 是原参考第 0 帧及原速度；
random 是随机参考相位，均不是静态站立初始化。
输出 `summary.txt` / `evaluation.json` 和 `episodes.csv`；使用 checkpoint 保存的 SDS normalizer，
在 10Hz 抽样窗口上重复噪声评分，不更新任何网络/normalizer。
汇总包含平均回合秒数、超时比例、脚抬起高度、关节/转向速度、SMP reward，以及参考/静态候选评分。
超时比例是存活率，**不是踢腿成功率**；仍需目测是否连续完成回旋踢，避免将僵立或抖动算成功。
不同 checkpoint 的 SDS normalizer 可能不同，raw SDS 和真实动作也应一起比较。

## 10. 反馈文件和命令

prior 阶段：

```bat
python tools/g1_spinkick/workflow.py --stage feedback --run_dir output/a5000_g1_spinkick_prior_50k_r01
notepad output\a5000_g1_spinkick_prior_50k_r01\feedback.txt
```

pilot 阶段或完整 RL 阶段（以下为完整 RL）：

```bat
python tools/g1_spinkick/workflow.py --stage feedback --run_dir output/a5000_g1_spinkick_2048_full_r01
notepad output\a5000_g1_spinkick_2048_full_r01\feedback.txt
notepad output\a5000_g1_spinkick_2048_full_r01\evaluation_starts_seed123\summary.txt
notepad output\a5000_g1_spinkick_2048_full_r01\evaluation_random_seed123\summary.txt
```

请发 prior 的 feedback.txt、pilot/full 的 feedback.txt、对应评测 summary.txt，
并说明是否有完整踢腿与转身、频繁摔倒/抖动/僵立；可附短视频。
若出错，先提供该阶段 feedback.txt 或 console.txt 尾部。模型与完整 CSV 留在远端，不必上传。

## 实现与验证边界

训练配置未使用旧 Humanoid、Getup 或 DeepMimic 权重；模拟器继续使用现有位置 PD / 30Hz 控制 / 240Hz 仿真。
新增 diffusion CLI 可选 seed，默认不改变旧调用；修正 Loss 日志按实际累计更新数平均，
不再让短测 loss 被默认 2000 的输出间隔除小。该修正不改变优化损失或梯度。
本地只验证极小规模流程，不能保证 G1 在 A5000 上的学习效果或真机可用性。

本地验证记录：G1 数据/预算/派生文件保护/未完成 prior 拦截检查通过，真实 Newton 的动态 FK 与 prior 特征一致性检查通过；
2 次 diffusion 更新、GIF/动作生成、保存/加载、一次 PPO 更新、8 环境精确首回合评测及反馈汇总通过。
8 环境评测使用未学会动作的短测权重，结果摔倒属于预期，不能当成正式学习结果。
