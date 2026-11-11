# G1 Location：新版 Newton、原版冻结 LaFAN

2026-11-11。本地准备，用户 push 后再远端拉取启动；Steering 行为不变。

## 原版对齐与 G1 差异

- 直接继承 `TaskLocationEnv`，保留原版目标采样、目标观测、奖励和早停逻辑。
  20 秒回合，目标在距当前位置 1–10 米的随机方向，每 5–10 秒更新。
  位置奖励 `exp(-0.5 * squared_distance)`；位置/速度/朝向奖励权重为 1/0/0。
- SMP/PPO 参数与 `data/agents/smp_task_humanoid_agent.yaml` 一致：
  task/SMP=0.5/0.5，SDS scale=6，扩散步 [22,15,8]，10 帧历史，启用 GSI。
- 从随机 RL 权重开始，直接加载作者发布的冻结先验
  `data/models/smp_priors/smp_prior_lafan.pt`，SHA256 为
  `93f14378cc9501e7ec2d52387c930320fcccc341580cbe69e17d5a61c667142f`。
  不重训先验，不加载 Steering 或 Dodgeball RL 权重。
- 复用 G1→规范 Humanoid 观测适配和反向 GSI。构造阶段的 G1 walk/run 文件
  只提供原生参考元数据，不替代 LaFAN；实际训练 reset 使用 LaFAN 生成的 GSI buffer。
- G1 保留原生 29DoF 和已验证的控制保护：PD 目标内缩 0.03rad，膝目标下界 0.03rad，
  actor 目标正则 weight=1.0 / scale=0.25rad，额外限位压力奖励为 0。
  这些是机器人适配差异，不是原版人形的原始控制配置。
- 默认独立 `newton_1_7` 后端：Newton 1.7.0、Warp 1.18.0、MuJoCo/mujoco-warp 3.15.0，
  30Hz 控制 / 240Hz 仿真。复用 `newton_1_7_g1_dodgeball_engine.yaml`；文件名不表示任务含球。
  旧后端保留，显式 `--engine newton` 需使用旧 Conda 环境和独立输出目录。

## 完整预算

以作者发布的 `data/logs/smp_location_humanoid_log.txt` 最后一行为准：

**4096 × 32 × 17,401 = 2,280,783,872 samples**，最后 Iteration=17400。

默认 4096 环境；若远端确认资源不足，2048 环境需要 34,802 次更新，样本总量不变。
不复用其他 task 的预算，不自动追加训练或降并行数。

同一 A5000、新后端的 G1 Steering 本轮实测 17,301 次更新耗时 45小时11分30秒；
按更新数线性外推，Location 约 **45.5 小时**，初步预留 **40–55 小时**。
Location 吞吐尚未远端实测；启动后按实际时间和速度更新预计结束时间。

## 检查

- `python -B -m unittest tests.test_g1_location tests.test_g1_steering tests.test_g1_lafan -v`：
  13 项通过，覆盖原版配置/任务继承、发布日志预算、LaFAN 适配及 Steering 回归。
  测试进程 exit 0；沙箱退出时有 wandb 临时目录清理权限提示，不影响测试结果。
- WarpSim / RTX 5070 实际短训：
  `output/g1_location_newton17_smoke_20261111_r01`，8 环境、3 次更新、768 samples，
  65.41 秒，exit 0、manifest `budget_completed`；日志/模型数值正常，无容量溢出，
  冻结 LaFAN 权重和其 normalizer 逐位未变。
- smoke 权重严格重载成功，策略步进观察/奖励有限；headless GL 截图
  `output/g1_location_newton17_gl_20261111.png` 确认 G1 黑灰材质、红色目标立柱和指示线。
  强制移动目标后经过一次仿真步进，目标 marker 的实际 body 位置与任务目标一致。
  检查脚本为 `output/check_g1_location_newton17_gl.py`；无需更改渲染或物理后端。

短训只验证通路，不代表最终技能表现。

## 远端操作（用户 push 后）

输出目录必须为新目录；重试或日期变化时替换目录名，并同步评测/demo 路径。
远端保留旧 Conda 环境，使用现有 WarpSim。

```bat
cd /d C:\Project\MimicKit
conda activate WarpSim
set WARP_CACHE_PATH=C:\Project\MimicKit\output\warp_cache_newton17
python -u -B -m tools.g1_location.workflow --stage smoke --out_dir output/g1_location_newton17_smoke_remote_20261111_r01
```

确认 smoke 完成且 manifest 验证通过后，启动完整训练：

```bat
python -u -B -m tools.g1_location.workflow --stage full --num_envs 4096 --out_dir output/g1_location_newton17_full_20261111_r01
```

launcher 记录版本、配置、资产哈希、精确预算、日志和模型；结束检查日志/模型有限性、
容量溢出及冻结先验（含其 normalizer）逐位不变性。
成功标志：`run_manifest.json` 中 `status=budget_completed`、`workflow_exit_code=0`，
日志末行 Iteration=17400 / Samples=2280783872。启动后由用户监测并反馈结束。

完成后独立评测：

```bat
python mimickit/run.py --mode test --num_envs 512 --test_episodes 512 --devices cuda:0 --visualize false --rand_seed 123 --engine_config output/g1_location_newton17_full_20261111_r01/engine_config.yaml --env_config output/g1_location_newton17_full_20261111_r01/env_config.yaml --agent_config output/g1_location_newton17_full_20261111_r01/agent_config.yaml --model_file output/g1_location_newton17_full_20261111_r01/model.pt --out_dir output/g1_location_newton17_eval_20261113_r01
```

GL 可视化（保留 G1 黑灰原色、红色目标及指示线）：

```bat
python mimickit/run.py --mode test --num_envs 1 --devices cuda:0 --visualize true --engine_config output/g1_location_newton17_full_20261111_r01/engine_config.yaml --env_config output/g1_location_newton17_full_20261111_r01/env_config.yaml --agent_config output/g1_location_newton17_full_20261111_r01/agent_config.yaml --model_file output/g1_location_newton17_full_20261111_r01/model.pt
```

验收关注接近目标、到达后的稳定性、目标切换与连续运动，不能仅以回合时长判断任务达标。
