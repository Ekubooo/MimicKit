# G1 SMP Dodgeball：动作先验权重对照

2026-11-05。目标是改善G1真实动作质量与躲避能力；本轮只检验奖励平衡假设，不预先认定限位问题已修复。

## 设置与理由

控制探针见 `G1_CONTROL_PROBE.md`：膝关节能离开下限，直接截断/释放旧策略却增加非脚触地失败。
旧策略raw SDS长期偏高，因此选择从随机RL权重重新学习，避免把旧策略对限位的依赖带入修复。
复用真实200k G1 walk/run prior，不重训diffusion。

只改变训练目标的两项参数：

| 参数 | 原配置 | 本轮 |
| --- | --- | --- |
| task_reward_weight | 0.5 | 0.5 |
| smp_reward_weight | 0.5 | 1.5 |
| sds_loss_scale | 6 | 4 |

SMP权重提高3倍，降低指数衰减强度以减轻远离先验时风格奖励接近0的问题。
这是组合假设实验，不能分别归因于权重或scale；数值是本轮实验选择，不宣称论文复现或最优配置。
物理版本、PD增益、目标范围、机器人、球速、终止、GSI、观测、网络、优化器与数据保持原配置。
所有更改在独立runner生成的 `agent_runtime.yaml` 中，原默认配置不变。

训练seed123，2048环境×32步×2001更新=131,137,536 samples，最后Iteration=2000。
与旧full的 `int_models/model_0000002000.pt` 精确对齐2001更新，不拿新短训与旧20402更新直接判优劣。
旧训练过程不是新代码下重新训练的对照；远端已核实原seed/环境数，物理依赖和训练配置仍需按manifest比较。
预计训练3.5–4小时，依据原400更新40–46分钟及full吞吐外推；评测另约5–10分钟。

## 启动

本机在main提交，push由用户执行；远端拉取后核对提交。
在远端CMD激活mimickit-newton后：

```bat
cd /d C:\Project\MimicKit
python -B tools/g1_dodgeball/style_balance.py --stage smoke --out_dir output/g1_style_balance_smoke_20261105_r01
python -B tools/g1_dodgeball/style_balance.py --stage trial --out_dir output/g1_style_balance_2048_20261105_r01
```

smoke为8环境×32步×3更新，只检查训练接口。通过后才启动trial，两个阶段各自从随机RL权重开始。
目录非空拒绝运行，重试需换目录。runner记录配置、prior哈希、提交、预算和子进程起止时间。
检查所有日志列、有限值、最后更新数/samples及模型张量；列粘连或缺列直接报错，不输出虚假通过。
训练进程EXIT_CODE=0与WORKFLOW_EXIT_CODE=0均需成立；进程错误或最终数值审计失败不扩大预算。
本轮不自动接着full，训练期间不持续监控。意外断电/终端强杀可能没有完成记录，不能按模型存在判断完成。

## 完成后评测与验收

首先读取 `runtime.txt`、`feedback.json`、`validation.json` 和必要的 `console.txt`。
收到用户完成反馈后，运行以下远端PowerShell循环；不与训练同时运行：

```powershell
Set-Location C:\Project\MimicKit
$evalPython = 'C:\Users\ClassRoom\miniconda3\envs\mimickit-newton\python.exe'
$evalModels = @{
    candidate = 'output/g1_style_balance_2048_20261105_r01/model.pt'
    baseline2000 = 'output/a5000_g1_dodgeball_2048_full_r01/int_models/model_0000002000.pt'
}
foreach ($label in @('baseline2000', 'candidate')) {
    foreach ($seed in @(123, 456)) {
        foreach ($balls in @('on', 'off')) {
            & $evalPython -B tools/g1_dodgeball/evaluate.py --model_file $evalModels[$label] --init reference --episodes 32 --seed $seed --projectiles $balls --joint_diagnostics --out_dir "output/g1_style_balance_eval_20261105_r01/${label}_${balls}_seed${seed}"
            if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $label $seed $balls" }
        }
    }
}
```

总计256首回合，使用同种子、初始化、回合数及发球开关比较。
重点看raw SDS、非脚触地、膝限位占比、姿态倾斜、timeout和用户视觉；由于scale和normalizer不同，不横向比较SMP奖励均值来宣布风格改善。
预先设置继续扩大训练的门槛：两个seed带球raw SDS均比等预算旧模型降低至少25%，
timeout不下降超过2/32回合、非脚触地失败不增加超过2/32回合；
无球平衡和限位/倾斜指标不得明显恶化，且用户视觉确认动作没有靠僵硬姿态替代自然躲避。
这些是筛选门槛，不等于技能已经通过；未达标先分析，不自动追加预算。

新模型可视化（默认GSI初始化，单独注明与reference量化评测的区别）：

```bat
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --agent_config output/g1_style_balance_2048_20261105_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/g1_style_balance_2048_20261105_r01/model.pt --test_episodes 8
```

主要反馈由助手远端读取：训练runtime/feedback/validation，各组summary/diagnostics；
用户只需报告何时结束及后续视觉验收，不需要传模型或全部CSV。

## 2026-11-05 实测结果：未通过扩大预算门槛

训练提交 `240e0dc246ca7c6815e761626290e035058d29a0`，目录如上。
13:58:02–16:39:20（北京时间），耗时9677.70秒，即2小时41分18秒。
EXIT_CODE与WORKFLOW_EXIT_CODE均为0，validation通过；末次Iteration=2000，
Samples=131137536，模型张量有限。训练console未检出Traceback、NaN或约束/接触容量溢出。
末次训练Sds_Loss_Mean=0.52277356；该训练均值不替代独立技能评测。

评测目录 `C:\Project\MimicKit\output\g1_style_balance_eval_20261105_r01`。
18:13:49–18:19:09，run_manifest状态completed，8组均退出0，256首回合完整。
各组console和launcher日志均未检出上述异常。两策略评测env/engine配置完全相同，
agent配置差异仅smp_reward_weight与sds_loss_scale；prior哈希相同。
评测依赖为Newton1.0.0/Warp1.12.0/MuJoCo3.5.0/MuJoCo-Warp3.5.0.2/Torch2.10.0+cu128。

| 策略 | 球 | seed | 平均秒数 | timeout/32 | 命中回合 | 非脚触地回合 | raw SDS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 旧模型iter2000 | on | 123 | 10.822 | 5 | 25 | 2 | 2.209125 |
| 旧模型iter2000 | on | 456 | 8.458 | 3 | 26 | 3 | 2.124955 |
| 新模型 | on | 123 | 3.625 | 0 | 32 | 0 | 0.204586 |
| 新模型 | on | 456 | 3.361 | 0 | 32 | 0 | 0.212341 |
| 旧模型iter2000 | off | 123 | 20.000 | 32 | 0 | 0 | 1.875006 |
| 旧模型iter2000 | off | 456 | 20.000 | 32 | 0 | 0 | 1.888197 |
| 新模型 | off | 123 | 20.000 | 32 | 0 | 0 | 0.129834 |
| 新模型 | off | 456 | 20.000 | 32 | 0 | 0 | 0.130677 |

命中/非脚触地标签理论上可重叠，timeout只代表任务检测器下存活，不是自然躲避认证。
带球raw SDS下降90.74%/90.01%，但timeout分别减少5/32与3/32，均超过允许的2/32。
新策略64回合全部命中，其中seed123的27回合、seed456的29回合仅发球一次即结束；
发球次数不等于已验证的接球/躲球次数，不能据此计算躲球成功率。

| 策略/场景 | seed | 左膝近限位比例 | 右膝近限位比例 | 躯干倾斜超过60度比例 |
| --- | --- | --- | --- | --- |
| 旧/带球首次发球后 | 123 | 62.82% | 53.77% | 9.24% |
| 旧/带球首次发球后 | 456 | 63.48% | 54.09% | 10.47% |
| 新/带球首次发球后 | 123 | 5.61% | 0.00% | 0.00% |
| 新/带球首次发球后 | 456 | 6.12% | 0.00% | 0.00% |
| 旧/无球全程 | 123 | 37.53% | 49.05% | 0.00% |
| 旧/无球全程 | 456 | 37.39% | 49.39% | 0.00% |
| 新/无球全程 | 123 | 13.39% | 0.03% | 0.00% |
| 新/无球全程 | 456 | 13.40% | 0.02% | 0.00% |

以上为10Hz及额外终止采样的比例。带球新模型更早终止，首次发球后仅392/327个样本，
旧模型为2695/1958个，不能把带球姿态比例下降解释为长期鲁棒性提升。
无球均为20秒、6432个样本，仍显示raw SDS与膝近限位比例下降，
因此改善并非仅由带球早终止解释；自然动作质量仍待视觉确认。

结论：此组合调整在当前预算下减少限位依赖、提高先验相似度，却损害有效避球。
它没有跑通G1躲避球，不自动追加full。结合此前控制探针，更支持策略/训练目标层面的
问题参与其中，尚不能排除运动数据覆盖、奖励设计或控制参数的影响。
这是单个训练seed、两个评测seed和历史checkpoint对照，不能分别归因于两项参数，
也不能断言更长训练一定无效。后续优先核查来球前后动作响应与任务奖励信号，
再选单变量或渐进任务训练；本次未实施新训练设置。

模型SHA256：

- baseline iter2000：`1d2c0f88925ddb6d3cfd8355ac98318132fc60d31efa4d820cae21e05a97ea19`
- candidate：`50d7f0e011d38b6b745e817da3ee58576bdef644d807153e078c4577108ea380`
- prior：`b14e2857e3dad2af734f80e2d7fde2bba6f80de29182e2e12b5cd01be76efe3d`
