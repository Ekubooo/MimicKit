# 默认 Humanoid 近站立恢复小课程（2026-11-15）

## 起点和目标

上一轮 `58376be` 的 `output/humanoid_getup_standing_neutral_pair_r01` 已完成：
两组各 2,097,152 samples，总耗时约 9min。干净小扰动站姿均 100/100，参考动作
90%～100% 初始化 task-only 9/100、frozen-SMP 7/100，倒地均 0/100，无数值失败。
它证明训练保留了已初始化的站立能力，不证明从零学会平衡或起身。

本轮只用 frozen-SMP 组最终 `model.pt` 热启动，继续 Newton 1.7 / WarpSim /
现有 400k 冻结 prior。task/SMP=0.5/0.5、actor std=0.003125、actor LR=3.90625e-7、
critic LR=1e-4、actor/critic minibatch=4096/2048 均保留。不增加先验训练，
不修改 PD 增益、动作映射或物理限位。

目标：让站立附近受扰动状态恢复，并向参考动作末段扩展；完整倒地起身仍单独报告。

## 两段上限固定的课程

| 阶段 | 站姿/参考帧比例 | 站姿关节扰动 std | 站姿水平速度扰动 | 参考帧阶段 |
|---|---|---|---|---|
| 1 | 90% / 10% | 0.008rad | 每个水平分量 uniform[-0.03,0.03]m/s | 97%～100% |
| 2（仅通过门槛才运行） | 75% / 25% | 0.015rad | 每分量 uniform[-0.06,0.06]m/s | 93%～100% |

参考帧保留原动作速度，不降低速度或变成静态姿态。站姿支路在重置时获得水平速度
脉冲，竖直速度和关节速度为零；SMP 历史的旧帧是静止姿态，只有最新帧包含该脉冲，
避免把静态位置历史与十帧持续非零速度混在一起。姿态扰动后重新计算地面抬升。
此分布只影响 TRAIN；TEST 的 standing/reference/starts 明确选择各自初始化，
不把混合分布成功率当作起身成功率。训练中的定期 test 使用当前阶段的 reference。

每段默认 **4096×32×32=4,194,304 samples**，最多两段 **8,388,608 samples**。
CLI 不接受超出该单段上限的预算，2048 回退时仍保持采样上限。不存在自动加训或
第三段/完整起身启动。预计一段含基线及独立评测约 8～12min，两段约 16～24min，
参考上一轮实测，启动后再校准。

## 基线、评测和晋级

每段先用源 checkpoint、相同 seed=456 分别评测干净站姿、当前阶段扰动站姿、
当前阶段参考末段各 100 回合。基线出现失败状态或干净站姿结束稳定率低于 95%，
立即停止，不启动该段训练。基线不改模型/normalizer。

训练后重复上述三组，并增加倒地 starts 100 回合。只有同时满足以下工程门槛才
允许进入下一段：

- 无数值/越界失败；
- 干净站姿回合结束仍稳定 ≥95%；
- 扰动站姿回合结束仍稳定 ≥80%；
- 参考末段结束稳定率 ≥ max(25%, 该段训练前基线+10个百分点)，目标上限 95%。

这些阈值是实验推进门槛，不是统计显著性声明。倒地成功率不作为本轮晋级门槛，
但必须明确报告，不能用站立结果替代。未通过则保存 `stop_reasons` 并结束；
summary 的 `complete` 只表示这次有限实验结束，不表示学会完整起身。

## 热启动和证据

`train_getup.py --model_file` 使用既有 `run.py` 的严格权重加载：恢复 actor、critic、
固定动作 std、观测/SDS normalizer 和冻结 prior；优化器动量、采样计数、回合状态
重新开始。因此属于热启动，不能称为完整断点续训。
runner 核对源目录 `run_manifest.json` 的 prior 哈希与当前 prior 一致，并核对
保存的动作 std/type；源模型 SHA256 记录到阶段摘要和训练 manifest。
默认源为 `output/humanoid_getup_standing_neutral_pair_r01/frozen_smp/model.pt`。
第二段只从通过门槛的第一段最终模型热启动。

源 prior：`output/a5000_getup_prior_joint_400k_scratch_r01/model.pt`，SHA256
`9e718c27750e30032e557ff5e74b6b411593052618a6abdc5b6205a9a78cbbc0`。
Git 不同步这些权重。用户 push 后助手在远端拉取、核对资产并启动。

```bat
cd /d C:\Project\MimicKit
python tools/getup/near_standing_curriculum.py --num_envs 4096 --out_dir output/humanoid_getup_near_curriculum_r01
```

总状态位于该目录 `summary.json`，每段有 `before/`、`train/`、`after/`，各评测
完整配置和逐回合 CSV 均保留。旧训练目录及旧配置保持可用。本轮不提前远端部署
未推送的训练代码，不在本地执行课程训练。

## 检查

25 项测试已通过（含真实 CUDA/Newton）：混合分布、phase 范围、水平速度边界、
脉冲历史一致性、局部 reset 隔离、TEST 分布隔离、晋级条件和源 prior/std 防错。
本地单次热启动更新已正常完成（8 环境、256 samples、EXIT0）；冻结 prior 的
100 个状态张量前后完全一致，观测/SDS normalizer 计数从 256 累积到 512，确认
不是重置统计。更新后的独立干净站姿评测 8/8 success、8/8 end-stable，平均连续
站稳 19.9667s、回报 600/600。证据：`output/getup_near_warmstart_smoke_20261115_r01`
和 `output/getup_near_warmstart_clean_eval_20261115_r01/evaluation.json`。
本地 fixture prior 不用于声明真实先验下的恢复效果；仍需在远端真实 400k prior
和上一轮权重上检验课程是否有效。
