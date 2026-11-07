# G1躲避球：论文奖励与受限PD目标

2026-11-07更新：2048环境完整训练及16组独立评测均正常完成。持续顶膝下限的主要行为在本次采样中消失；动作质量仍需视觉验收。

## 目的和边界

旧策略能通过大量20秒timeout，但会反复顶膝限位；增强先验的2001更新模型减少贴限位，
却64/64带球回合命中。本轮直接改变造成持续向限位外请求目标的控制条件，并对齐论文奖励。
这是组合修复实验，不是单因素因果研究，不宣称保证消除实际关节超限或获得自然躲避。
从随机RL权重和归一化状态训练，不加载旧完整模型；复用原真实200k frozen walk/run prior。

## 精确配置

- 论文依据：[arXiv v3 D.4 Eq.27–28](https://arxiv.org/html/2512.03028v3#A4.SS4)。
- task = `1 - exp(-0.3 * (distance(root, ball) - 1.5))`；不截掉负奖励，不含旧水平速度奖励。
- 总奖励 `0.6 * task + 1.2 * SMP`，SDS scale = 4。
- 所有PD目标截在 `[physical_lower + 0.03, physical_upper - 0.03]` rad；
  左右膝目标下限进一步设为 `+0.03 rad`（约1.72度屈膝）。G1物理膝下限仍为-0.087267 rad。
- 0.03 rad为本轮工程选择，并非论文参数。它限制的是目标，不是直接修改实际角度。
- 沿用既有SMPG1Dodgeball的actor目标正则：权重1、角尺度0.25 rad，约束策略均值到同一目标范围。
  环境对所有采样动作做最终截断；保留原零中心动作归一化，避免突然把随机策略均值搬到关节中位。
- 原按接近限位计费的pressure奖励权重为0；仍记录其诊断值。
- 保留G1 XML、真实物理限位、PD增益、力矩上限、30Hz控制/240Hz物理、GSI、先验与球参数。
  球仍为12–15m/s、8–10m、1–4s间隔；因此只对齐论文奖励，不称为完整论文复现。

当关节静止在物理下限时，新目标必在其内侧，PD位置误差项指向离开下限；上限同理。
这去掉持续向限位外请求目标的机制，但速度阻尼、接触和外力仍会影响实际力矩/角度。
训练与评测必须使用保存的env/agent配置。默认任务及旧checkpoint配置保持原行为。

## 规模选择与耗时

| stage | 环境 | 更新 | 最后Iteration | samples | A5000估计 |
| --- | --- | --- | --- | --- | --- |
| smoke | 8 | 3 | 2 | 768 | 数十秒至数分钟，仅检查通路 |
| expanded | 2048 | 8001 | 8000 | 524353536 | 11–16小时 |
| full | 2048 | 20402 | 20401 | 1337065472 | 28–40小时 |

推荐full，匹配原完整策略采样预算并保存每200次更新的中间checkpoint。
时间依据最近2001更新实测2小时41分18秒线性外推（full约27.4小时），
以及旧full实测约37小时56分，考虑新增actor正则、回合长度/测试开销变化取上述区间。
这不是保证，也不承诺完整规模必然成功；不再用几百更新的技能效果否决长训。

## 本地与远端

本地主分支提交由助手执行，push由用户执行。远端拉取后核对提交、清洁工作区、环境与资产。
先运行CPU contracts和真实prior/GSI smoke，验证无NaN/容量溢出、日志列完整、模型可保存。
smoke失败不启动长训。正式训练仍用已验证的隐藏WMI独立启动方式，日志重定向，保留UU会话。
不要用普通Start-Process之后退出UU。用户反馈完成后再检查，不持续监控。

远端CMD中的训练命令（使用所选的一项，不连续启动两项）：

```bat
cd /d C:\Project\MimicKit
conda activate mimickit-newton
python -B -m unittest tests.test_g1_limit_training.LimitContracts tests.test_g1_dodgeball_repair.RepairContracts tests.test_g1_style_balance
python -B tools/g1_dodgeball/limit_training.py --stage smoke --out_dir output/g1_paper_targets_smoke_20261105_r01
python -B tools/g1_dodgeball/limit_training.py --stage full --out_dir output/g1_paper_targets_full_20261105_r01
```

若选择expanded，将最后一行stage改为expanded，out_dir改为
`output/g1_paper_targets_expanded_20261105_r01`。目录非空拒绝运行，重试使用新目录。

## 检查与评测

完成后检查runtime双退出码0、feedback状态budget_completed、validation passed、精确samples/最后迭代、
模型有限性与console。通过仅表示数值和预算完成。需要读取中间checkpoint，避免只验收最终模型。
反馈由助手读取，用户提供完成时间和视觉判断，不需传模型。可用CMD查看：

```bat
type output\g1_paper_targets_full_20261105_r01\runtime.txt
notepad output\g1_paper_targets_full_20261105_r01\feedback.json
```

标准评测命令示例（之后扩展seed123/456及新seed789/1011，reference/GSI、有球/无球）：

```bat
python -B tools/g1_dodgeball/evaluate.py --model_file output/g1_paper_targets_full_20261105_r01/model.pt --init reference --episodes 32 --seed 123 --projectiles on --joint_diagnostics --out_dir output/g1_paper_targets_full_eval_r01/reference_on_seed123
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --env_config output/g1_paper_targets_full_20261105_r01/env_config.yaml --agent_config output/g1_paper_targets_full_20261105_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/g1_paper_targets_full_20261105_r01/model.pt --test_episodes 8
```

验收同时检查原球参数下存活/命中/触地、实际膝反向与连续贴限位、目标越界率应为0、姿态与raw SDS，
以及用户观察连续来球后能否恢复自然动作。与旧full及等预算checkpoint比较，奖励公式不同，
task return不能横向直接判优劣；新目标范围也不自动保证实际关节始终在范围内。
暂不因孤立的短时接近限位而终止回合，也不放宽现有命中/触地标准来获得更高存活率。

## 本地验证记录

- 12项CPU contracts通过：论文奖励符号/数值、PD位置误差在两端指向内侧、正则梯度、
  原默认配置不变、预算、历史目标截断与日志验证。
- 1项实际Newton GPU集成测试通过（RTX5070，83.76秒）：8环境极端输入检查、3次更新/768样本、
  精确日志校验、模型重新加载、4首回合保存配置评测；发送目标越物理限位比例为0。
- 本地集成使用已有2k toy prior，并以参考状态替代GSI生成来检查接口；不冒充远端真实prior或技能验收。
- full/expanded dry-run均通过。新pressure加权指标名缩短为`G1_Pressure_Cost`，
  避免既有25字符宽度文本logger粘连列；历史日志不修改。
- 首次GPU检查因沙箱Warp缓存目录权限失败；指定仓库output下的缓存后通过。
  退出时wandb临时目录仍有清理权限警告，测试本体通过且退出0，未据此判断训练技能成功。

## 远端完整训练记录（2026-11-07核查）

- 提交：`66a00d5`；目录：`output/g1_paper_targets_full_20261105_r01`。
- 2048环境、32步rollout、20402次更新，最后Iteration=20401，samples=1337065472。
- 北京时间2026-11-05 19:15:57启动，11-07 10:56:44结束；实测142848.20秒，约39小时40分48秒。
- `EXIT_CODE=0`、`WORKFLOW_EXIT_CODE=0`，feedback=`budget_completed`，validation passed。
  104行训练记录和模型数值校验通过，console未发现NaN/容量溢出/异常退出。
- 训练内测试平均存活：更新2000为4.39秒，4000为16.95秒，8000为19.64秒，
  12000为19.85秒，20000为19.91秒，最后19.94秒。短训练不足以判断本轮最终技能。
- 最后一行Train_Episode_Length/回报为0，是更新20400重置累计器后最后一个更新尚无完成回合，
  不能解释为策略突然失效。最后一行SDS也仅覆盖短窗口，不与之前整段统计直接比较。
- 训练完成与数值通过不等于动作验收通过；独立评测使用保存的配置和冻结权重/归一化器。

## 独立评测结果（2026-11-07）

远端目录：`output/g1_paper_targets_eval_20261107_r01`。
北京时间15:30:29–15:42:50，共16组、512首回合，每组32环境，20秒时限。
全部退出0，manifest=completed，控制台异常检查通过；没有修改策略、归一化器、球参数或终止规则。
远端仍为提交66a00d5，工作区干净；Newton1.0.0 / Warp1.12.0 / MuJoCo3.5.0 /
MuJoCoWarp3.5.0.2 / Torch2.10.0+cu128，复用原200k prior。

| 模型/初始化/球 | seeds | 满时回合 | 平均存活(s) | 命中 | 非脚触地 | raw SDS均值 |
| --- | --- | --- | --- | --- | --- | --- |
| 旧full / reference / on | 123,456 | 62/64 | 19.776 | 2 | 0 | 2.0862 |
| 新full / reference / on | 123,456 | 64/64 | 20.000 | 0 | 0 | 0.5986 |
| 新full / reference / on（新增种子） | 789,1011 | 63/64 | 19.872 | 1 | 0 | 约0.6203 |
| 新full / GSI / on | 123,456 | 64/64 | 20.000 | 0 | 0 | 0.6019 |
| 新full / reference / off | 123,456 | 64/64 | 20.000 | 0 | 0 | 0.0499 |
| 新full / GSI / off | 123,456 | 64/64 | 20.000 | 0 | 0 | 0.0488 |
| 新8000 / reference / on | 123,456 | 64/64 | 20.000 | 0 | 0 | 0.6074 |
| 旧full / reference / off | 123,456 | 64/64 | 20.000 | 0 | 0 | 约0.1398 |

新full带球合计191/192满时，1次命中（reference seed1011），0次非脚触地；
无球128/128满时。reference表示walk/run随机相位，GSI表示先验生成初态。
每个slot只统计首回合，包括失败末帧；不是连续运行无限时稳健性证明。
相同seed不保证策略不同后的所有来球轨迹完全相同，不能把2次与0次失败解释为统计显著。
raw SDS是冻结先验下的动作分布诊断，不是视觉自然度分数；不同奖励公式的task return不直接比较。

### 膝关节与姿态

| 指标 | 旧full reference带球（123/456） | 新full全部10组 |
| --- | --- | --- |
| 首次发球后左膝贴下限采样比例 | 86.985% / 86.744% | 0% |
| 首次发球后右膝贴下限采样比例 | 53.303% / 53.859% | 0% |
| 膝角小于0 | 带球后左约93%、右约58% | 全程均未观测到 |
| 最长连续负膝角采样跨度 | 左7.37秒；右0.90秒 | 0 |
| 躯干倾斜>60度采样比例 | 发球后8.913% / 9.188% | 全程0% |
| 所有关节实际发送PD目标越物理限位 | 旧策略仍存在 | 全程0% |

新full共320回合，每侧膝64238个10Hz/终止采样点；最小左膝0.00814 rad，右膝0.08254 rad，
最小膝目标0.03 rad。实际角度可以低于目标下限，但本次未越过0，更未持续顶住-0.087267 rad物理下限。
10Hz加终止帧采样不是240Hz全部物理子步审计，本次未记录完整实际力矩，不能据此保证所有瞬时角度/力矩。

其他关节并未完全远离限位：新full reference seed123发球后腰pitch、左踝roll、腰roll、
右踝roll近限位比例约4.92%、4.62%、3.78%、1.60%；对应8000模型约6.39%、5.63%、5.58%、2.56%。
补查reference/GSI的123/456四组腰/踝连续近限位采样跨度，最大0.20秒，仍存在实际短时超限
（例如GSI456左踝roll最大超出物理边界约0.162 rad）。这是剩余动态限位问题，不应称为物理引擎限位已彻底修复。

### 当前结论与下一步

- 长训明显改善了短训没有学会的技能；主要的持续负膝角/顶膝下限行为在当前评测范围内已消除，
  同时保持了高带球存活率，没有用早摔缩短统计窗口来降低限位比例。
- 本轮同时改变奖励、目标范围和actor正则，不能单独归因于其中一项，也没有修改Newton求解器。
- 更新8000模型已取得相近任务表现；完整模型腰/踝近限位比例略低。当前没有仅凭日志立即自动加训的依据。
- 下一步先用本文件demo命令视觉验收新full：连续来球后的姿态、步态、避让与恢复。
  必须同时加载该模型保存的env_config和agent_config。视觉通过前，状态仍为“定量显著改善，待动作验收”，
  不标为G1完整躲避球已全面完成。
- 远端保留训练runtime/feedback/log/model及每组evaluation.json、diagnostics.json、episodes.csv和轨迹；
  汇总为run_manifest.json、analysis.json、other_joint_analysis.json。本地仅记录结论，不复制大轨迹或原始反馈。
