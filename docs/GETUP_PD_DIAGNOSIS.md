# Getup PD、初始化和探索诊断（2026-11-14）

## 已完成的上一轮

`cef74b4`，远端 `output/humanoid_getup_standing_pair_r01`：两组各
10,092,544 samples，15:19:47～15:39:11。task_only 502.77s、frozen_smp 506.19s，
均 EXIT0；三种初始化各 100 回合，两组共 600 回合 success/end-stable 均 0。
因此不能把失败只归因于 SMP；扩大采样之前先检查动作和控制。

## 控制证据

远端仅执行无优化器更新的 10s 物理探针，每组 8 环境；agent/normalizer 指纹前后
一致。每步断言环境裁剪后的 PD 目标与 Newton command buffer 一致。
保存路径：`output/getup_pd_probe_20261114_r02/summary.json`（无初始扰动）与
`output/getup_pd_probe_20261114_r05/summary.json`（0.03rad 初始扰动，组间匹配起点）。
r03/r04 的临时探针参数构造失败，没有物理结果，不纳入统计。

| 无初始扰动的 10s 控制 | 连续站稳至少 5s | 平均最长站稳 |
|---|---:|---:|
| 固定中立 PD | 8/8 | 约 9.97s |
| 固定动作范围中心 | 0/8 | 0s |
| 随机策略确定性输出 | 0/8 | 0s |
| 仅将随机策略输出中心移到中立姿态 | 8/8 | 10s |
| 中立 PD + 原噪声 std=0.05 | 0/8 | 0.079s |
| 中立 PD + std=0.0125 | 0/8 | 1.779s |
| 中立 PD + std=0.003125 | 8/8 | 9.992s |
| 中立 PD + std=0.00125 | 8/8 | 9.975s |
| task_only / frozen_smp 已训练策略 | 各 0/8 | 0s |

这不是总体成功率估计；确定性无扰动组的 8 个槽位不是独立随机起点。
原 0.03rad 起点扰动下，匹配对照的固定中立目标仅 4/8 成功，说明“站姿附近”
也有需要主动平衡恢复的状态，不能假定每个起点都能靠恒定目标保持。

动作链：网络归一化均值 → action normalizer 反归一化 → 环境裁剪 → Newton PD。
它没有自动叠加当前关节姿态。随机网络接近零输出对应范围中心；膝关节中心
1.39625rad（约 80°），中立目标是 0。膝关节的原探索 std 约 5.6°，从中心探索
到中立约相差 14.3 个标准差。固定中心目标约 0.667s 头高跌破 0.6m，随机策略
约 0.767s。整个动作空间的 std=0.05 对应约 3.30°～9.60°，不是 0.05rad。

因果解释：初始策略主动破坏支撑姿态，而原噪声又破坏正确目标附近的站立。
20s 无摔倒提前结束回合因此主要采到倒地状态。它解释早期站立样本缺失，
但尚不能断言是长训练失败的唯一根因，也不能据此保证起身成功。

## 本次最小修改

- 仅 SMPGetup 增加可选 `actor_init_pose: standing`，把新策略输出 bias 初始化为
  中立 PD 目标对应的归一化值。新诊断配置末层权重初始为 0，确保初始均值精确。
- 保留绝对 PD 动作空间、边界、动作归一化和旧 checkpoint schema；加载旧模型
  会覆盖新建模型的 bias。旧配置、G1、其他任务默认行为不变。
- 新 `standing_neutral` 配置使用探索 std=0.003125（原值的 1/16）、起点扰动
  std=0.003rad（原值的 1/10），保留旧对照。只用于站稳诊断，不直接用于完整起身。
- 低噪声配套 actor learning rate=3.90625e-7（原 1e-4 的 1/256），critic 不变。
  这是按高斯均值更新相对标准差的敏感度做的局部校准，不是论文给出的学习率。
  本地实测低噪声仍用旧 actor 学习率时，一次更新 Clip_Frac=0.8、Imp_Ratio≈0.2，
  更新后 8 回合站立评测 0/8。该结果是降低噪声后的新增优化风险，不能倒推原配置
  一定存在同样问题；故必须连同更新幅度一起验证。
- `standing_diagnostic.py --neutral_init` 选择新配置。原入口默认仍用旧配置。
- prior 仍冻结，奖励公式/权重不再改变；不改引擎增益、关节限位或接触参数。

本地 WarpSim：21 项测试通过，包括原配置默认行为、归一化后目标恢复、旧状态
加载覆盖初始化、非法控制模式拒绝，以及真实 Newton 的既有重置/站稳测试。
新策略未训练前，在 0.003rad 匹配初始扰动下，确定性策略与中立 PD 加
std=0.003125 噪声均 8/8 站稳近 10s。此物理测试不使用 prior 奖励；本地加载的是
极短训练的 fixture prior，不能代表真实先验评分或 RL 学习效果。

同 seed、8 环境、256 samples 的一次更新对照：仅保留旧 actor 学习率时更新后
站立评测为 0/8；校准为 3.90625e-7 后 EXIT0，Clip_Frac 从 0.8 降至约 0.245，
Imp_Ratio 从约 0.2 回到 1.01。随后独立 20s 评测 8/8 success、8/8 end-stable，
平均最长连续站稳 19.9667s、平均任务回报 600/600、数值失败 0。
本地证据：`output/getup_neutral_smoke_20261114_r02` 与
`output/getup_neutral_smoke_eval_20261114_r02/evaluation.json`。
这只证明小扰动下初始化与一次更新可以保留站稳，不证明持续学习、真实 prior 下
的奖励平衡、抗扰恢复或完整倒地起身已经成功。

## 论文边界与下一步

[SMP §6.3](https://arxiv.org/html/2512.03028v3#S6.SS3) 使用高斯策略输出绝对 PD
目标，冻结 diffusion；[D.5](https://arxiv.org/html/2512.03028v3#A4.SS5) 使用
90% 参考帧 / 10% 增广倒地初始化。本地动作形式与该描述一致，并非发现论文
要求残差动作而实现遗漏。原版 [CharEnv](https://github.com/xbpeng/MimicKit/blob/main/mimickit/envs/char_env.py)
也用动作范围中心；不能将通用实现说成普遍错误。
[Peng & van de Panne 2017](https://xbpeng.github.io/projects/ActionSpace/index.html)
研究表明动作参数化/局部反馈影响学习和稳健性，但不直接给出本任务的初始化或噪声值。
本次 bias、噪声和站稳课程属于根据实测做的工程适配，不宣称论文原版配方。

本轮不启动额外训练。用户 push 后建议只做更小的两组诊断：每组 4096×32×16 =
2,097,152 samples（总计 4,194,304），核对更新是否破坏初始站立能力；保存间隔 5。
预计含评测约 8～15min，以实际启动后吞吐修正。成功后再扩大起点分布、参考阶段，
最终回到完整倒地分布和独立起身验收；低探索幅度不能未经验证直接用于复杂起身。

```bat
python tools/getup/standing_diagnostic.py --neutral_init --num_envs 4096 --max_samples 2000000 --out_dir output/humanoid_getup_standing_neutral_pair_r01
```

复查工具：`tools/getup/probe_pd.py --out_dir 新目录 --model_files 模型路径`；
可选 `--init_noise 0.03`，不更新任何模型参数。严格以独立倒地起身评测判定最终目标。
