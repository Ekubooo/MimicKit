# G1 Steering：原版任务与冻结 LaFAN 先验

2026-11-09。本轮仅完成本地准备；用户 push 后再远端拉取、检查和启动。

## 配方与规模

- 原版 `smp_steering_humanoid_env.yaml` 的任务逻辑直接继承：目标速度 0.5–5.0 m/s，
  目标每 4–7 秒改变方向，朝向跟随目标方向，20 秒回合、原版早停；速度/朝向奖励 0.7/0.3，速度误差 scale=0.5。
- 原版 SMP/PPO 参数保持一致，包括 task/SMP=0.5/0.5、SDS scale=6、steps=[22,15,8]、10 帧历史及 GSI。
- 直接使用作者冻结 `data/models/smp_priors/smp_prior_lafan.pt`，SHA256：
  `93f14378cc9501e7ec2d52387c930320fcccc341580cbe69e17d5a61c667142f`。
  不训练新先验，不加载躲避球策略；Steering RL 权重和 normalizer 从头训练。
- 复用已完成躲避球的 G1→规范 Humanoid 观测、反向 GSI 适配，提取为共享 mixin；
  G1 仍为原生 29DoF、30Hz 控制/240Hz 仿真。本轮默认切换到独立 `newton_1_7` 后端；
  原 `newton` 后端仍可显式选择，详见 [新版迁移记录](NEWTON_1_7_MIGRATION.md)。
- 保留已验证的 G1 控制差异：PD 目标内缩 0.03rad、膝目标下界 0.03rad、actor 目标正则
  1.0/0.25rad、额外限位压力奖励为 0。通用 `SMPG1` agent 与旧 `SMPG1Dodgeball` 入口共享实现。
- 新版使用 `newton_1_7_g1_dodgeball_engine.yaml`，与旧版配置仅 `engine_name` 不同；
  文件名不表示 Steering 中有球。
- 构造阶段仍需要 `dataset_g1_locomotion.yaml` 的 G1 walk/run 元数据，launcher 自动准备；
  实际 RL reset 使用冻结 LaFAN 生成的 GSI buffer。这两段动作不替代 LaFAN 先验。

预算对齐作者发布 `data/logs/smp_steering_humanoid_log.txt` 末行，而非套用 Dodgeball 预算：

**4096 × 32 × 17,301 = 2,267,676,672 samples**，最后 Iteration=17300。
资源问题回退 2048 时使用 34,602 次更新，保持相同总样本数。逻辑错误不靠减环境数绕过。
按 A5000 上 G1 LaFAN Dodgeball 实测 33小时24分 / 10,201 更新外推，中值约 56.7 小时，
旧版预估 50–65 小时；这不是新版的实测时间。新版需先在 A5000 复测 4096 环境负载和吞吐，
远端启动后再给结束时间。

## 本地检查

- LaFAN 适配、原版 Steering 参数/完整与回退预算、PD 目标保护单元检查通过。
- `output/g1_steering_smoke_20261109_r01`：8 环境，真实冻结先验、默认完整 GSI buffer，
  3 次 RL 更新 / 768 samples，exit 0；预算、日志/模型有限性、无容量溢出、先验及其 normalizer
  逐位不变性检查通过。它只证明训练通路可运行，不代表 Steering 技能质量。
- 新 Steering smoke 模型与修改前保存的 Dodgeball LaFAN smoke 模型均严格重新加载成功，
  实际策略步进的观察/奖励有限；两个独立进程的 GL 出图通过，确认视觉网格保留 0.2/0.7
  黑灰颜色，Steering 红绿标记正常。截图位于 `output/g1_steering_gl_20261109.png`、
  `output/g1_dodgeball_gl_20261109.png`。本机没有远端完整权重，因此不冒充完整策略的本地效果验收。

## 远端启动（CMD / Anaconda Prompt）

用户 push 后，助手拉取并核对代码，在克隆的 WarpSim 环境安装锁定版本、完成迁移验证。
不需要新先验或新机器人资产。旧环境保留。当前远端准备状态见新版迁移记录。
输出目录必须为空；若启动日期改变或重试，换一个新目录并同步后续命令。

```bat
cd /d C:\Project\MimicKit
conda activate WarpSim
python -u -B -m tools.g1_steering.workflow --stage smoke --out_dir output/g1_steering_smoke_remote_20261109_r01
python -u -B -m tools.g1_steering.workflow --stage full --num_envs 4096 --out_dir output/g1_steering_full_20261109_r01
```

第二条在 smoke 完成并检查通过后执行。full 不自动追加预算或自动改环境数。
旧版备用：在 `mimickit-newton` 中给 launcher 加 `--engine newton`，并使用独立输出目录。
launcher 保存 `run_manifest.json`、`runtime.txt`、`console.txt`、`log.txt`、实际配置、
`model.pt` 和中间权重，并在结束时验证精确预算、有限性、容量溢出及冻结先验不变性。

训练结束后首先检查 manifest 的 `status=budget_completed`、`workflow_exit_code=0`、
日志最后 `Iteration=17300` / `Samples=2267676672`。标准评测命令（并行回合数可能上取整）：

```bat
python mimickit/run.py --mode test --num_envs 512 --test_episodes 512 --devices cuda:0 --visualize false --rand_seed 123 --engine_config output/g1_steering_full_20261109_r01/engine_config.yaml --env_config output/g1_steering_full_20261109_r01/env_config.yaml --agent_config output/g1_steering_full_20261109_r01/agent_config.yaml --model_file output/g1_steering_full_20261109_r01/model.pt --out_dir output/g1_steering_eval_20261109_r01
```

Steering 完成后可视运行：

```bat
python mimickit/run.py --mode test --num_envs 1 --devices cuda:0 --visualize true --engine_config output/g1_steering_full_20261109_r01/engine_config.yaml --env_config output/g1_steering_full_20261109_r01/env_config.yaml --agent_config output/g1_steering_full_20261109_r01/agent_config.yaml --model_file output/g1_steering_full_20261109_r01/model.pt
```

人工重点查看转向、速度跟随、连续行走及有无持续关节顶死。完整回合或程序正常退出不等于技能达标。
训练启动后不持续轮询，由用户反馈结束后再查日志；不把 demo 与 4096 环境训练同时运行来比较性能。

## 现有 Dodgeball：GL 恢复黑灰原色

两个 Newton 后端创建仓库 G1 角色时均取消统一蓝色覆盖，直接使用 XML 视觉网格的原始颜色；
无开关，不接入 RTX，不改机器人资产或模型权重。球与 Steering 标记仍保留任务颜色。
以下为旧环境备用命令；新版运行命令及跨版本评测见新版迁移记录。

```bat
cd /d C:\Project\MimicKit
conda activate mimickit-newton
python mimickit/run.py --mode test --num_envs 1 --devices cuda:0 --visualize true --engine_config output/g1_lafan_full_20261107_r01/engine_config.yaml --env_config output/g1_lafan_full_20261107_r01/env_config.yaml --agent_config output/g1_lafan_full_20261107_r01/agent_config.yaml --model_file output/g1_lafan_full_20261107_r01/model.pt
```
