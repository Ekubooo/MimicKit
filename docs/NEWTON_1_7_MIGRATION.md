# Newton 1.7：G1 与默认 Humanoid 迁移

2026-11-09。本地提交由用户 push，之后助手远端拉取、验证，再启动训练。

## 版本与回退

| Conda 环境 | engine_name | Newton | Warp | MuJoCo | MuJoCo-Warp |
| --- | --- | --- | --- | --- | --- |
| mimickit-newton（备用） | newton | 1.0.0 | 1.12.0 | 3.5.0 | 3.5.0.2 |
| WarpSim（新版） | newton_1_7 | 1.7.0 | 1.18.0 | 3.15.0 | 3.15.0 |

两套环境均保留 Torch 2.10.0+cu128。选择后端必须同时选择对应解释器和 engine 配置；
入口检查 Newton minor 版本，错误搭配直接报错，不静默降级。
`constraints-warpsim.txt` 锁定新版四个物理包，不是独立安装列表。

升级分支 `codex/newton-warpsim` 在 `f801a68` 提交后，以 `c3e7aa1` 非快进合并至 main，
分支保留。原 `newton_engine.py` 未修改。后续 main 集成恢复两个后端的 G1 原色、
默认新版 Steering launcher、显式旧版回退和跨版本评测。

## 适配范围

- 独立新版后端处理 Newton 1.7 的关节目标布局、关节状态和控制 API。
- 默认 Humanoid 的球关节角速度、力矩和显式 PD 在局部关节坐标与父锚点坐标之间转换；
  reset、部分环境更新和 CUDA graph 路径保持一致。
- 新版接触传感器分别提供总接触与地面接触，保留 G1 的跌倒/球击检测所需语义。
- G1 原生 29DoF，资产质量、惯量、限位、力矩上限、PD 增益和 armature 经实际模型比对；
  保留原有受限 PD 目标、膝下界、actor 目标正则。未修改机器人 XML、奖励或冻结先验。
- 新版 GL 接收模型颜色并支持角色/任务标记颜色、视频接口、交互外力。
  G1 主角色默认保留 XML 的 0.2/0.7 黑灰原色，无额外开关，未接入 RTX。

这些是接口与运行验证，不保证新旧求解器轨迹或策略效果逐位相同。
旧权重可以严格加载；是否需要重训，应先对完整策略做同种子评测与视觉验收。

## 本地证据

使用 `D:\CondaEnvs\envs\WarpSim\python.exe`：

- `tests/test_newton_warpsim.py`：10 项通过，1 项 GL/视频专项未在此批开启。
  覆盖控制模式、旋转球关节、接触、部分 reset、外力、graph/eager、G1 与 Humanoid 资产。
- 后端选择 2 项、Steering 配方/预算/旧版回退 3 项、LaFAN 适配 8 项通过。
- `output/g1_steering_newton17_smoke_20261109_r01`：8 环境、真实 LaFAN GSI、3 更新/768 samples，
  62.83 秒，退出 0，manifest 为 `budget_completed`；模型有限、无 overflow，先验与 normalizer 逐位不变。
- 新 Steering smoke 权重及旧 Dodgeball smoke 权重分别在新版严格加载、实际策略步进、GL 出图通过。
  截图为 `output/g1_steering_newton17_gl_20261109.png` 和
  `output/g1_dodgeball_newton17_gl_20261109.png`，已检查黑灰颜色与 Steering 标记。
- 显式跨版本评测入口已用旧 Dodgeball smoke 权重跑完 8 个 GSI 首回合并生成关节诊断，
  输出 `output/g1_lafan_newton17_eval_20261109_r01`；保存后端为 `newton`，评测后端为 `newton_1_7`。
  该未训练充分的 smoke 策略 8/8 失败、均时约 0.492 秒，只验通路，不作为迁移质量结论。
- 合并后使用旧 `mimickit-newton` 解释器和原配置重新加载 Dodgeball smoke 权重，
  GL、原色断言及策略步进均通过，日志 `output/legacy_gl_regression_20261109.log`。
  本地受限执行时需将 `WARP_CACHE_PATH` 指向仓库可写目录，避免默认缓存的写入权限错误。

本机没有远端完整策略权重，上述 smoke 不代表完整策略迁移效果，也不代表训练质量。

## 远端克隆状态

远端 `C:\Users\ClassRoom\miniconda3\envs\WarpSim` 已由 `mimickit-newton` 克隆。
103 个已安装包的名称/版本一致；`pip check` 返回 0。尚未升级物理包。
Conda 日志显示 transaction 完成，但 PowerShell 对 Conda stderr 警告抛异常，未捕获原始 clone 退出码；
因此以实际文件、包清单和依赖检查确认克隆，不声称原命令退出 0。

运行检查发现新路径的 `warp.dll` 被 Windows 应用程序控制阻止（WinError 4551）。
CodeIntegrity 3033/3077 事件明确指出签名等级/策略要求未满足。
新旧 DLL 的大小和 SHA256 一致，未发现复制损坏；旧环境 Warp/CUDA 导入仍成功，A5000 可用。
未修改安全设置或绕过阻止。新环境目前不能标记为运行就绪；训练前必须解决并重新验证。
远端证据：`output/warpsim_setup_20261109/` 下的 `clone.log`、`cloned_versions.json`、
`code_integrity.json`、`mimickit-newton_import.txt`、`WarpSim_import.txt`。

## 用户 push 后的步骤

1. 拉取 main 并核对提交，保留旧环境、旧权重、旧配置。
2. 在 WarpSim 安装锁定的新版物理包；保留克隆的大包与 Torch，不整体升级依赖。
   先检查安装计划，完成后 `pip check`、版本/CUDA/import、真实仿真及 GL smoke 均需通过。
   若应用程序控制仍阻止加载，需要设备管理员处理该具体文件的授权，不能靠更换路径绕过。

```bat
cd /d C:\Project\MimicKit
conda activate WarpSim
python -m pip install --dry-run -c constraints-warpsim.txt newton warp-lang mujoco mujoco-warp
python -m pip install -c constraints-warpsim.txt newton warp-lang mujoco mujoco-warp
python -m pip check
```

以上针对已克隆且已有项目/GL 依赖的环境，不是从零安装配方。
检查计划不得意外替换 Torch/CUDA 大包；如有额外依赖问题，先核对再定向补齐。

3. 用远端完整 G1 Dodgeball 权重做显式新版评测：

```bat
python -u -B -m tools.g1_dodgeball.evaluate --model_file output/g1_lafan_full_20261107_r01/model.pt --engine newton_1_7 --episodes 512 --joint_diagnostics --out_dir output/g1_lafan_newton17_eval_remote_r01
```

默认 `--engine saved` 使用 checkpoint 保存的后端；显式切换只允许已验证的 G1 配置且其余物理参数一致。
评测输出分别记录 `saved_engine` 和 `evaluation_engine`，不改旧训练目录。

4. 新版 GL demo（待远端依赖和运行验证通过后使用）：

```bat
conda activate WarpSim
python mimickit/run.py --mode test --num_envs 1 --devices cuda:0 --visualize true --engine_config data/engines/newton_1_7_g1_dodgeball_engine.yaml --env_config output/g1_lafan_full_20261107_r01/env_config.yaml --agent_config output/g1_lafan_full_20261107_r01/agent_config.yaml --model_file output/g1_lafan_full_20261107_r01/model.pt
```

旧版备用：激活 `mimickit-newton`，engine_config 使用原训练目录中的 `engine_config.yaml`，同样显示黑灰原色。

5. 新版 Steering 先完成 smoke 与 4096 环境负载测试，再开始完整训练。
   预算仍为 **2,267,676,672 samples = 4096 × 32 × 17,301**；资源不足才退回 2048、34,602 更新。
   任务奖励、SMP/PPO 参数和原版冻结 LaFAN 先验保持不变。旧版吞吐外推约 50–65 小时，
   新版结束时间必须按 A5000 实际吞吐重估。具体启动入口见 [Steering 文档](G1_STEERING_TRAINING.md)。
