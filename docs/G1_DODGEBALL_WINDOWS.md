# G1 SMP 躲避球：Windows A5000 完整流程（2026-10-31）

目标：保留 Newton1.0.0 / Warp1.12.0 当前环境，单卡 A5000、2048 环境，
先训练 G1 walk/run 冻结 prior，再从头训练 G1 躲避球。长训练由用户在远端执行。
本轮不加载已成功的 G1 回旋踢策略或 prior，不升级物理引擎，不修改旧任务参数。
Newton 增加可选约束/接触容量配置，旧任务默认仍为450/150；本任务使用1024/256。

## 规模与边界

| 阶段 | 预算 | 最后迭代 | A5000 时间估算 |
| --- | --- | --- | --- |
| prior-smoke | batch512，2 更新 | 1 | 数分钟含首次编译/导出 |
| 完整 prior | **200,000 更新、batch512、lr2e-4、统计20,000窗口** | 199999 | **1.5–2.5小时** |
| rl-smoke | 8环境、256样本、1更新，保留GSI | 0 | 数分钟 |
| pilot | 2048×32×400 = **26,214,400样本** | 399 | **0.7–1.5小时** |
| 完整 RL | 2048×32×20402 = **1,337,065,472样本** | 20401 | **35–50小时** |

RL 精确对齐 `data/logs/smp_dodgeball_humanoid_log.txt` 最后记录：
原4096环境×32步×10201更新，改2048环境后更新数翻倍，不减少样本。
这是公开日志最后记录的规模，不推测未记录的最终预算。
prior200k 对齐公开 multi_clip 配置和已下载 LaFAN/locomotion 权重的 EMA.step；
不是论文正文整体400k–800k区间的逐项复现。原始校验摘要与 SHA256 见
`tools/g1_dodgeball/reference_budget.json`，自动复制到每轮输出。

时间基于上一轮 G1 回旋踢 A5000 的 prior50k=21分29秒、RL3.802亿=9小时56分53秒外推，
躲避球额外有球体、GSI和不同回合长度，尚无 G1 躲避球长跑实测。
所有短测、pilot、full 都独立从头开始；full 不接着 pilot，也不把 pilot 样本计入完整预算。

当前 G1 数据：walk120Hz125帧约1.033秒、run30Hz25帧约0.8秒，均为循环片段，等权采样。
没有公开可直接加载的兼容 G1 LaFAN SMP prior。
**这是完整更新/采样预算的 G1 walk/run 基线，数据多样性不等同于 Humanoid LaFAN1。**
短片段会被反复采样；200k×512不代表约1亿独立动作。
同一 prior 可供后续相同特征定义的 G1 Location/Steering 使用，策略网络仍需各自训练。

G129DoF、31身体节点、prior2100维；策略额外接收球的位置/速度。
与回旋踢不同，`global_obs=False`，不能复用其 prior。
只允许两脚触地，膝盖、手、躯干触地会触发原任务的摔倒条件。
球体保持原资产0.1米半径、density200（质量约0.838kg），不按 G1 身高缩放。
1球、8–10米、水平发射速度12–15m/s、每1–4秒重发，瞄准 `torso_link`。
保留当前仓库 Newton 躲避球的命中检测实现，没有在本轮改动任务奖励/终止算法。
任务奖励与 SMP 各0.5；GSI开启，SDS统计在1亿样本后冻结，沿用公开 task agent。
本地未训练prior的GSI探针出现nefc>450溢出，故单独提供
`data/engines/newton_g1_dodgeball_engine.yaml`，增加缓冲容量；这不改变质量、PD或时间步。
容量增大会增加显存需求，2048环境的最终吞吐/显存需以A5000实测为准。

## 1. 同步：本地提交，远端拉取

本地 Anaconda Prompt / CMD：

```bat
cd /d D:\ComputerScience\PROJECT\MimicKit
git status --short
git add args/smp_dodgeball_g1_args.txt data/datasets/dataset_g1_locomotion.yaml data/envs/smp_locomotion_g1_prior_env.yaml data/envs/smp_dodgeball_g1_env.yaml data/envs/view_motion_locomotion_g1_env.yaml data/agents/smp_dodgeball_g1_agent.yaml data/engines/newton_g1_dodgeball_engine.yaml mimickit/engines/newton_engine.py tools/diffusion_model/config/tinymdm_locomotion_g1.yaml tools/g1_dodgeball tests/test_g1_dodgeball.py docs/G1_DODGEBALL_WINDOWS.md plan.md
git diff --cached --stat
git commit -m "Add G1 locomotion prior and Newton SMP dodgeball workflow"
git push origin main
```

远端训练机 Anaconda Prompt / CMD：

```bat
cd /d C:\Project\MimicKit
git pull --ff-only origin main
conda activate mimickit-newton
set PYTHONUTF8=1
set MPLBACKEND=Agg
python tools/g1_dodgeball/workflow.py --stage check
```

若 G1资产/motion/球体缺失，从原数据包复制并保留路径：
`data/assets/g1/`、`data/assets/objects/dodgeball.xml`、
`data/motions/g1/g1_walk.pkl`、`data/motions/g1/g1_run.pkl`。
G1 回旋踢使用过的机器人资产无需重下，但走跑 motion 必须存在。
Git 不传 ignored 的资产或训练权重；不需要远端重装当前 Conda 环境。

预检应为29DoF、2100维、两段动作、限位超差接近0，末尾 `G1 LOCOMOTION / DODGEBALL ASSETS OK`。
预检保留原motion/XML，生成 `output/g1_locomotion_assets/` 下派生 motion 与 `preparation.json`。
walk 无改动；run 左/右踝横滚各8帧被投影到合法角度，最大改动约0.04914/0.06561rad。
这不是 IK 重定向或物理可行证明；生成文件若被用户另行编辑，工具拒绝覆盖。

## 2. 播放参考动作

```bat
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/newton_engine.yaml --env_config data/envs/view_motion_locomotion_g1_env.yaml --visualize true --test_episodes 8
```

这是运动学播放，非策略执行。观察走跑是否仍正常、脚部是否明显扭曲。

## 3. 先训练 locomotion prior

在同一训练窗口逐行执行，每条结束再输入下一条：

```bat
python tools/g1_dodgeball/workflow.py --stage prior-smoke
type output\a5000_g1_locomotion_prior_smoke_r01\runtime.txt
```

EXIT_CODE=0 且 OUTPUT_VALIDATION=PASSED 后：

```bat
python tools/g1_dodgeball/workflow.py --stage prior
type output\a5000_g1_locomotion_prior_200k_r01\runtime.txt
python tools/g1_dodgeball/workflow.py --stage sample
explorer output\a5000_g1_locomotion_prior_200k_r01_samples\samples
python tools/g1_dodgeball/workflow.py --stage feedback --run_dir output/a5000_g1_locomotion_prior_200k_r01
```

正式 prior 从随机权重开始，不加载2更新短测；完成为199999、EXIT_CODE=0。
GIF为16段10帧窗口（约0.3秒），不是完整走跑序列。
RL入口会核对 prior 的 EMA.step 达到其配置预算，以及骨架、特征顺序、频率和数据哈希；
拒绝错用回旋踢 prior、未训练完 prior 或在 prior 后悄悄修改过的参考。

## 4. RL 短测、pilot

```bat
python tools/g1_dodgeball/workflow.py --stage rl-smoke
type output\a5000_g1_dodgeball_rl_smoke_r01\runtime.txt
```

短测使用已经完成200k的正式prior，保留公开GSI，验证其生成初始化、球体环境、SDS和PPO保存/加载通路。
不能拿prior-smoke或2000更新诊断权重做正式GSI训练；本地这些探针产生异常初始状态和NaN。
200k prior的真实GSI物理稳定性仍需在远端这一步验证，不能用本地导出成功替代。
256样本只是通路验证；可能在球发射前摔倒，不能代替独立球体发射测试。
EXIT_CODE=0 且 OUTPUT_VALIDATION=PASSED 后：

```bat
python tools/g1_dodgeball/workflow.py --stage pilot
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --mode test --num_envs 4 --visualize true --agent_config output/a5000_g1_dodgeball_2048_pilot_r01/agent_config.yaml --model_file output/a5000_g1_dodgeball_2048_pilot_r01/model.pt --test_episodes 8
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_pilot_r01/model.pt --init gsi --episodes 100 --out_dir output/a5000_g1_dodgeball_2048_pilot_r01/evaluation_gsi_seed123
python tools/g1_dodgeball/workflow.py --stage feedback --run_dir output/a5000_g1_dodgeball_2048_pilot_r01
```

先反馈 pilot：是否能走动/侧移、是否真的发球、会否立即摔倒/僵立。
400更新尚不能证明最终学习失败；结合回合长度、发球次数、task回报、SMP/SDS再决定是否扩大。
若通路报错、出现非有限值、参考明显不正常，先排查，不直接排队完整训练。

## 5. 完整 RL

pilot 趋势合理后：

```bat
python tools/g1_dodgeball/workflow.py --stage full
```

约35–50小时，训练窗口被占用；电脑不休眠时可锁屏或关闭屏幕。
自动保存输出目录 `output/a5000_g1_dodgeball_2048_full_r01/`：
console.txt、runtime.txt、run_manifest.json、实际配置、motion_audit.json、log.txt、
model.pt 和每200迭代的 int_models/model_XXXXXXXXXX.pt。
正常结束为Iteration20401、Samples1337065472、EXIT_CODE=0、OUTPUT_VALIDATION=PASSED。
workflow结束时检查全部日志数值和约束溢出，发现问题会返回非零状态并保存validation.json，
避免把Python进程正常退出误认为训练有效。该检查在阶段结束后执行，长训练仍需查看中途日志。
本轮 task reward 非零，Train_Return/Test_Return 不再像纯回旋踢那样恒为0；
它们仍是未加权环境task reward累计，不是0.5task+0.5SMP的PPO优化奖励总和。
Episode_Length单位为30Hz控制步，600步对应20秒。
默认训练内test的32回合在2048环境下至少每环境一回合，通常多于32；不计入训练采样预算。

另开 Anaconda Prompt 看进度，可选GPU监控：

```bat
cd /d C:\Project\MimicKit
conda activate mimickit-newton
python -c "from pathlib import Path; p=Path('output/a5000_g1_dodgeball_2048_full_r01/log.txt'); print('\n'.join(p.read_text().splitlines()[-3:]) if p.exists() else 'Waiting for first log')"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5
```

监控Ctrl+C只停止监控，无需反馈GPU CSV。
中途可反馈当前日志摘要，不必为反馈停止训练；同卡可视/评测会争用GPU，尽量等训练结束。
所有阶段拒绝覆盖非空输出目录。重试用新 `--out_dir`，同步修改后续路径；
重训 prior 改路径时，RL需显式 `--prior_dir 新prior目录`。
当前checkpoint没有完整优化器/RNG计数，不宣称精确断点续训。

## 6. 最终可视、精确100回合评测

```bat
type output\a5000_g1_dodgeball_2048_full_r01\runtime.txt
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --mode test --num_envs 4 --visualize true --agent_config output/a5000_g1_dodgeball_2048_full_r01/agent_config.yaml --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --test_episodes 8
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --init gsi --episodes 100 --out_dir output/a5000_g1_dodgeball_2048_full_r01/evaluation_gsi_seed123
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --init reference --episodes 100 --out_dir output/a5000_g1_dodgeball_2048_full_r01/evaluation_reference_seed123
```

每100环境只统计首回合，恰好100回合，包括早期失败。
gsi沿用训练的生成初始化；reference关闭GSI，使用原walk/run随机相位及原速度，
不是从任意静态站立或随机倒地启动。可另用 `--init starts` 检查原动作第0帧。
提供20秒达到时限比例、task回报、发球次数、命中检测/非脚触地的回合数及冻结SDS评分。
命中与摔倒类别可能重叠，发射不等于实际遭遇，达到时限不是经人工验证的每球躲避率。
命中判断来自当前task的接触/速度变化启发式，不代表有精确命中真值。
保留episodes.csv做个例分析；不更新网络或normalizer。

## 7. 反馈清单

prior：

```bat
python tools/g1_dodgeball/workflow.py --stage feedback --run_dir output/a5000_g1_locomotion_prior_200k_r01
notepad output\a5000_g1_locomotion_prior_200k_r01\feedback.txt
```

pilot：

```bat
python tools/g1_dodgeball/workflow.py --stage feedback --run_dir output/a5000_g1_dodgeball_2048_pilot_r01
notepad output\a5000_g1_dodgeball_2048_pilot_r01\feedback.txt
notepad output\a5000_g1_dodgeball_2048_pilot_r01\evaluation_gsi_seed123\summary.txt
```

最终：

```bat
python tools/g1_dodgeball/workflow.py --stage feedback --run_dir output/a5000_g1_dodgeball_2048_full_r01
notepad output\a5000_g1_dodgeball_2048_full_r01\feedback.txt
notepad output\a5000_g1_dodgeball_2048_full_r01\evaluation_gsi_seed123\summary.txt
notepad output\a5000_g1_dodgeball_2048_full_r01\evaluation_reference_seed123\summary.txt
```

请提供对应阶段 feedback.txt、评测summary.txt，以及走跑/躲闪/摔倒/发球是否正常的描述或短视频。
不必上传model.pt、完整episodes.csv或GPU CSV；报错提供console.txt尾部。

## 来源与本地验证

- 公开task配置：https://github.com/xbpeng/MimicKit/blob/main/data/agents/smp_task_humanoid_agent.yaml
- 公开躲避球：https://github.com/xbpeng/MimicKit/blob/main/data/envs/smp_dodgeball_humanoid_env.yaml
- 公开multi_clip：https://github.com/xbpeng/MimicKit/blob/main/tools/diffusion_model/config/tinymdm_multi_clip.yaml
- 精确预算/哈希：tools/g1_dodgeball/reference_budget.json。

本地验证范围：数据与预算检查、Newton特征/FK/球发射、以真实参考窗口解码GSI并重置、
2更新prior导出、2000更新prior诊断，以及关闭GSI后的1次PPO更新/保存加载/极小评测。
2/2000更新prior的GSI物理探针失败已保留，不能把此处写作“真实训练prior的GSI短测已通过”。
本地2000更新权重不是远端正式200k prior；长训练和200k prior的GSI稳定性仍待远端反馈。
