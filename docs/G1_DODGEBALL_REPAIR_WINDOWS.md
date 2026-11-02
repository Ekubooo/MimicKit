# G1躲避球：位置目标/限位压力短程对照（2026-11-03）

旧full模型已经完成13.37亿样本，但球出现后双膝频繁顶限位、动作偏离先验；不直接追加完整训练。
本轮复用原200k G1 walk/run prior，两个RL模型均从头初始化，不加载旧策略或其normalizer/优化器。

## 实验内容

| 组别 | 实际发送给PD的位置目标 | actor的物理越界目标惩罚 | 12个腿关节贴限位的每步代价 |
| --- | --- | --- | --- |
| A：targets | 截断到模型物理限位 | 权重1，角度尺度0.25rad | 0 |
| B：pressure | 同A | 同A | 权重0.25，边缘宽度0.03rad |

两组之间唯一参数差异是最后一项。每步代价为12腿关节的边缘接近程度平方均值，
每关节截到[0,1]，反复停留每步都会继续扣除；不把一次瞬时经过限位等同于机械故障。
它也可能惩罚原跑步数据里合法靠近踝限位的时刻，所以先做A/B，不默认B一定更好。

任务/SMP权重仍各0.5，GSI、球速、回合长度、PD、XML与prior均保持。
不改变原扩展动作空间或PPO所记录动作的概率定义：只在发送PD目标时截断，
另对未截断actor均值的物理越界量加可微惩罚，避免仅靠截断后仍持续输出极端请求。
物理限位由求解器软约束实现，限制目标不保证真实关节永不瞬间超限。
只通过新env/agent启用这些改动；旧Humanoid、回旋踢与原G1躲避球配置仍沿用原行为。

每组pilot：2048×32×400 = **26,214,400样本**，最终Iteration399，与原pilot匹配。
单卡A5000每组估计45–75分钟；两组加评测预留2–3小时，尚未获得修复版远端实测。
smoke每组8环境、256样本、1次PPO更新。工具只提供smoke/pilot，不自动启动38小时级完整训练。
本轮是控制修复实验，不宣称已经完成论文规模复现。

## 1. 同步

本地Anaconda Prompt / CMD，提交本次改动（包括此前未提交的诊断记录）：

```bat
cd /d D:\ComputerScience\PROJECT\MimicKit
git add mimickit/envs/char_env.py mimickit/learning/agent_builder.py mimickit/learning/smp_g1_dodgeball_agent.py data/envs/smp_dodgeball_g1_target_limits_env.yaml data/agents/smp_dodgeball_g1_target_limits_agent.yaml data/agents/smp_dodgeball_g1_limit_pressure_agent.yaml tools/g1_dodgeball/repair.py tools/g1_dodgeball/evaluate.py tools/g1_dodgeball/joint_diagnostics.py tests/test_g1_dodgeball_repair.py docs/G1_DODGEBALL_REPAIR_WINDOWS.md docs/G1_DODGEBALL_DIAGNOSTICS_WINDOWS.md plan.md
git diff --cached --stat
git commit -m "Add matched G1 dodgeball target-limit repair experiments"
git push origin main
```

远端：

```bat
cd /d C:\Project\MimicKit
git pull --ff-only origin main
conda activate mimickit-newton
python tools/g1_dodgeball/repair.py --stage check
```

必须保留 `output/a5000_g1_locomotion_prior_200k_r01/`，不需要重训diffusion。
预检确认兼容的prior已完成至少200k更新、资产哈希一致。
原full/pilot、原prior、旧诊断输出均不覆盖。所有新阶段拒绝非空目录；重试使用新--out_dir并同步后续路径。

## 2. 两组短测

每条结束后再输入下一条：

```bat
python tools/g1_dodgeball/repair.py --stage smoke --variant targets
type output\a5000_g1_dodgeball_targets_8_smoke_r01\runtime.txt
python tools/g1_dodgeball/repair.py --stage smoke --variant pressure
type output\a5000_g1_dodgeball_pressure_8_smoke_r01\runtime.txt
```

两组都需要 EXIT_CODE=0、OUTPUT_VALIDATION=PASSED。未通过时先反馈对应console.txt，不扩大预算。
smoke使用远端真实200k prior，保留真实GSI；本地的参考GSI接口测试不能替代这一步。

## 3. 从头进行两组pilot

```bat
python tools/g1_dodgeball/repair.py --stage pilot --variant targets
python tools/g1_dodgeball/repair.py --stage pilot --variant pressure
```

这是两个独立模型，B不加载A。先等A结束，再运行B，避免同卡争用。
每组最后Iteration399、Samples26214400；退出码与validation均通过。
输出会记录repair_spec.json、实际env/agent/engine配置、模型、日志、prior哈希和代码版本。
日志新增Physical_Target_Loss、G1_Limit_Pressure_Mean、G1_Limit_Pressure_Weighted_Mean。
targets组仍记录压力指标但扣分权重0，pressure组从PPO合成奖励中扣除0.25×压力。
Train_Return/Test_Return依然只统计原环境task reward，不包含压力扣分、SMP或actor正则。

## 4. 查看两组动作

```bat
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --env_config output/a5000_g1_dodgeball_targets_2048_pilot_r01/env_config.yaml --agent_config output/a5000_g1_dodgeball_targets_2048_pilot_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_g1_dodgeball_targets_2048_pilot_r01/model.pt --test_episodes 8
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --env_config output/a5000_g1_dodgeball_pressure_2048_pilot_r01/env_config.yaml --agent_config output/a5000_g1_dodgeball_pressure_2048_pilot_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_g1_dodgeball_pressure_2048_pilot_r01/model.pt --test_episodes 8
```

必须使用输出目录的env_config和agent_config；只换model_file会错用原控制配置。
使用各自保存的SMPG1Dodgeball类加载，新buffer和配置与旧SMP不同，不把旧full权重直接当修复版。
关注首次来球后膝是否仍僵直、能否主动弯腿、躲避后是否恢复；400更新尚不保证学会完整躲闪。

## 5. 两组相同参考初始化、16首回合带球诊断

```bat
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_targets_2048_pilot_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles on --joint_diagnostics --out_dir output/a5000_g1_dodgeball_targets_2048_pilot_r01/diagnostic_on_seed123
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_pressure_2048_pilot_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles on --joint_diagnostics --out_dir output/a5000_g1_dodgeball_pressure_2048_pilot_r01/diagnostic_on_seed123
```

评测自动使用保存的修复agent/env，并关闭GSI来比较相同参考初始化；不修改网络或normalizer。
target_outside_%是**实际发送**PD目标越界比例，修复版应为0；requested_outside_%是截断前请求越界比例，
用于检查网络是否仍依赖截断。near_limit_%仍按实际关节位置统计，不要求立即为0。
实际跟踪误差对比的是发送后的目标；请与requested_outside_%一起看，避免仅因截断就误判策略学好了。
根部低姿态/倾斜标记是诊断量，不冒充摔倒真值。SMP不同checkpoint有不同归一化，
同时比较raw SDS、参考控制窗口和动作，不按一个SMP数值挑选模型。

需要原pilot的等预算带球对照，可额外执行：

```bat
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_pilot_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles on --joint_diagnostics --out_dir output/a5000_g1_dodgeball_2048_pilot_r01/diagnostic_matched_seed123
```

## 6. 反馈清单

```bat
python tools/g1_dodgeball/repair.py --stage feedback --variant targets
python tools/g1_dodgeball/repair.py --stage feedback --variant pressure
notepad output\a5000_g1_dodgeball_targets_2048_pilot_r01\feedback.txt
notepad output\a5000_g1_dodgeball_pressure_2048_pilot_r01\feedback.txt
notepad output\a5000_g1_dodgeball_targets_2048_pilot_r01\diagnostic_on_seed123\diagnostics.txt
notepad output\a5000_g1_dodgeball_pressure_2048_pilot_r01\diagnostic_on_seed123\diagnostics.txt
```

发这4份短文本，加上无拖拽时动作观察。每个诊断目录另保留summary.txt、episodes.csv、joint_trace.csv和state_trace.csv，暂不上传大CSV。
如做原pilot对照，也发其diagnostics.txt。GPU CSV不需要。

先验证“请求/实际目标越界减少、实际顶限位减少，且运动和任务学习没有明显恶化”，
再决定延长哪组以及是否另做风格权重对照；本轮没有改风格权重或重新训练prior。

## 本地验证范围

检查物理越界损失的梯度方向、压力代价边界、A/B参数隔离；
真实Newton验证发送目标确实截断，并通过一次PPO更新、冻结prior统计、保存和重载检查。
本地使用已有2000更新诊断prior时，以真实参考窗口替代生成器来验证GSI接口，
不声称本地验证过真实200k prior的修复策略学习质量。远端smoke/pilot结果仍待用户反馈。
