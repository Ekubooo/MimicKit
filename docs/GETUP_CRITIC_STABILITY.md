# 近站立课程 Critic 数值失稳诊断与修复（2026-11-15）

## 原失败

`42bfb22` 的 `output/humanoid_getup_near_curriculum_r01` 于 16:24:33 启动。
三组基线：干净站姿 100/100、扰动站姿 100/100、97%～100% 参考帧 16/100。
第一段第 0 次更新有限；第 1 次 Critic Loss 为 NaN，第 2 次起优势/Actor 也为 NaN。
16:27:25 人工仅终止该训练进程，runner 失败退出，第二段未执行。原站立模型哈希未变。
原 runner 只在一段结束后审查保存日志，不能及时阻止中途数值污染，这是本次一并修复的问题。

## 复现与因果证据

`tools/getup/probe_optimizer.py`：最多 5 次更新（4096 环境），逐 minibatch 记录
有限值、输入/目标/预测幅度、梯度范数和参数幅度。仅更新新的诊断实例，不覆盖源模型。
训练超时或中断后的原现场保持不变。诊断输出 `probe.json` 和约 105MB 的
`critic_replay.pt` 包含某次 rollout、更新前模型、SGD 动量及采样排列。

第一轮两次更新未再现 NaN，说明不能保证每次在同一个 minibatch 失稳。
第二轮复现了第 1 次更新的溢出，证据在远端
`output/getup_optimizer_probe_20261115_r02/probe.json`：

- 归一化观测有限，最大绝对值 10；仓库原本已有 ±10 裁剪。
- minibatch 目标值有限，最大约 141。
- Critic 预测从约 8.1e4、2.0e8 放大至 1.13e22。
- 梯度范数约 2.59e7 → 1.93e13，参数最大幅度最终约 3.73e8，随后平方损失溢出。
- 诊断在非有限损失进入 optimizer step 前停止，原 source SHA256 保持一致。

因此已确认主要数值链路为 **Critic 在连续 SGD 更新中发散 → 价值/优势污染 → Actor
污染**，不是已有证据支持的 Newton 输出 NaN 或未裁剪的无限大网络输入。
从低变化站立分布转到有恢复状态的数据，输入饱和及梯度尺度改变是合理诱因；
但没有通过消融证明某一个观测维度或 normalizer 更新是唯一诱因。

`tools/getup/replay_critic.py` 在同一失败捕获数据、初始模型、动量与相同循环采样
排列上对照（第二遍循环不重新洗牌，因此不宣称逐步复刻原现场）：

| 条件 | 更新结果 |
|---|---|
| 原 LR=1e-4、无梯度限制 | 第 127 个 minibatch loss=inf |
| 原 LR=1e-4、梯度范数上限 10 | 128 个 minibatch 均完成，末次 loss≈28.86 |
| LR=1e-5、无梯度限制 | 128 个 minibatch 均完成，末次 loss≈17.30 |

证据：`output/getup_critic_replay_20261115_r02.json`。这里选择只增加梯度限制，
保留既有学习率；末次 minibatch loss 不能用于宣称哪个策略效果更好。

## 最小修复

- 新 `data/agents/smp_getup_humanoid_near_agent.yaml`：Critic `grad_clip: 10.0`；
  actor/critic 均启用 `check_finite`。其他参数沿用校准站立配置。
- `MPOptimizer` 增加可选检查：backward 前 loss、step 前梯度、step 后参数。
  非有限值立即抛错，梯度裁剪的范数溢出也报错；不把 NaN 替换成零继续训练。
- 检查失败时不进入后续 PPO 更新或 checkpoint 输出，子进程非零退出，课程 runner
  保存失败状态并停止。它保留上次已完成保存的模型，不保证回滚内存里一次已溢出的 step。
- 有限值检查默认关闭，其他任务原配置行为不变；本课程预检查强制两优化器开启。
- 动作分布、归一化统计更新方式、课程起点、采样上限、奖励和冻结 prior 不变。
  本次未再次调小 Actor 学习率、重置 Critic、重置 normalizer 或改物理参数。

本地 29 项测试通过，覆盖 NaN loss、有限 loss/无限梯度、参数溢出、有限大梯度裁剪，
以及此前真实 Newton 重置、课程门槛和源权重校验。

## 远端有界验证与复跑

使用独立输出目录、旧已推送代码支持的 `grad_clip` 配置做诊断对照；不提前部署
未推送的训练源码。远端 `output/getup_optimizer_probe_20261115_clip_r03` 已完成
4096×32×5=655,360 samples：640 次 Critic、800 次 Actor 更新，全部有限；
最终模型有限，冻结 prior 的所有状态张量与源模型完全一致，源文件哈希未变。
新有限值保护代码另由本地测试验证，push 后才在远端实际启用。

该 5 更新模型以 seed=456 各评测 100 回合：干净站姿成功率 100%、末尾稳定率 97%；
97%～100% 参考帧恢复成功率和末尾稳定率均为 0%（原源模型该组末尾稳定率 16%）。
证据为该目录的 `eval_clean/evaluation.json`、`eval_reference/evaluation.json`。
这表明数值问题得到初步控制，但当前诊断模型的恢复能力没有改善，不能用有限 loss
代替任务效果验收。后续仍须按原门槛评测；未达标就停止，不进入第二段或完整训练。

这超过了原故障发生的更新位置，但不保证完整 32 次更新都稳定，更不证明起身成功。
复跑仍从原来的冻结 SMP 站立模型开始，不从诊断或失败模型开始；课程预算与晋级
条件不变。用户 push 后远端拉取，在新目录启动：

```bat
python tools/getup/near_standing_curriculum.py --num_envs 4096 --out_dir output/humanoid_getup_near_curriculum_r02
```

如发生新的有限值异常，程序应自行停止；保留异常信息后再分析，不自动加训。
