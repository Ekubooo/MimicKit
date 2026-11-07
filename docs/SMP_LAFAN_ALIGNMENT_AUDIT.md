# SMP Dodgeball：原版 LaFAN 对齐核查

2026-11-07，子代理核查及方案记录；本核查未改生产配置、未训练先验或RL。
同期主代理独立完成了[远端容量短训练](A5000_CAPACITY_20261107.md)。上游固定为
`xbpeng/MimicKit@fbe1a36e878f036d6bf5521defb8eb3b8ab4170d`。

## 已确认的结论

原版Humanoid可直接启动，是因为发布的LaFAN权重、114维动作表示及GSI解码已经匹配。
复用该冻结权重不需要重新取得LaFAN原始数据，也不需要为Dodgeball重新训练先验。
当前G1未具备这套表示适配，不能只替换模型路径；问题不是需要反复查数据或为每个任务训练先验。

本次实际CPU验证：同一发布权重加载到1140维窗口模型成功；加载到2100维窗口模型失败，
报`preprocess_conv`、`proj_in`等张量形状不一致。Humanoid为15个运动学节点、28DoF，
G1为31节点、29DoF，对应每帧114/210维；先验窗口均为10帧。

- 权重：`data/models/smp_priors/smp_prior_lafan.pt`。
- SHA256：`93f14378cc9501e7ec2d52387c930320fcccc341580cbe69e17d5a61c667142f`。
- 权重内EMA.step=200000；`obs_normalizer._mean/_std`形状为114。
- `dmodel.proj_in.weight`为256×114，`proj_out.weight`为114×256。

## 锁定的原版配置

| 项目 | 原版Dodgeball基准 |
| --- | --- |
| env | `data/envs/smp_dodgeball_humanoid_env.yaml` |
| agent | `data/agents/smp_task_humanoid_agent.yaml` |
| prior config / weight | `tinymdm_multi_clip.yaml` / `smp_prior_lafan.pt` |
| prior使用 | eval、requires_grad=False；保留权重内normalizer，RL期间不重训先验 |
| GSI / 窗口 | 开启 / 10帧；30Hz控制，默认Newton配置240Hz物理 |
| task / SMP权重 | 0.5 / 0.5 |
| SDS scale / diffusion steps | 6 / [22,15,8] |
| 任务奖励 | `0.9*(1-exp(-0.3*d)) + 0.1*exp(-sum(v_xy**2))` |
| 球 / 回合 | 1球，12–15m/s，8–10m，1–4秒重发，20秒回合；命中/非脚触地早停 |
| PPO | rollout32；actor/critic均2×1024，SGD lr=1e-4；actor/critic epochs=5/2 |
| 环境数 | 原版示例4096；2048为用户算力适配，4096能力由本轮远端测试决定 |
| RL预算参照 | 发布资产日志末行1,337,065,472 samples：4096×32×10201，或2048×32×20402 |

公开示例没有固定`max_samples`；上表RL预算是发布日志末行，不能冒充作者未公开的强制停止规则。
原`args/smp_dodgeball_humanoid_args.txt`指向不存在的`smp_dodgeball_humanoid_agent.yaml`，
应以README明确提供的`smp_task_humanoid_agent.yaml`为准。之前远端使用保存agent显式覆盖，所以能够运行。
Newton后端仍按用户既定要求使用；不因原README示例Isaac Gym而擅自切后端。

当前G1的精确差异：两段walk/run自训先验、210维/帧；最新`paper_targets_v1`还使用
task/SMP=0.6/1.2、SDS4和paper_v3任务奖励，并有目标截断及actor目标正则。
这些是迁移修复实验，不是原版LaFAN基线。目标截断是已验证的G1工程修复，不能为追求表面相同而静默撤销；
若保留，必须单列为机器人控制适配差异。

## 推荐执行路线：保留作者原始冻结权重

不再默认启动新的先验训练。下一步新增双向骨架表示适配，使现有作者LaFAN权重直接参与G1 RL。
该方向满足“复用同一原始先验”，但适配质量尚未验证，不能称为现成上游功能或保证技能成功。

1. **独立表示适配器**：拟新增`mimickit/envs/g1_lafan_adapter.py`。载入原Humanoid运动学骨架，
   将G1完整关节链的旋转合成为Humanoid规范骨架，而不是截取114列、补零或丢弃权重。
   校准静止姿态、轴方向、左右语义及身体尺度；由规范骨架FK产生一致的关键点位置。
   正向映射的语义节点如下（表内名称是链末端，不是仅取末端单关节角）：

   | Humanoid节点 | G1语义节点/链末端 |
   | --- | --- |
   | pelvis / torso / head | pelvis / torso_link（完整waist链）/ head_link |
   | left/right_upper_arm | 对应shoulder_yaw_link（完整shoulder链） |
   | left/right_lower_arm | 对应elbow_link |
   | left/right_hand | 对应wrist_yaw_link（完整wrist链） |
   | left/right_thigh | 对应hip_yaw_link（完整hip链） |
   | left/right_shin | 对应knee_link |
   | left/right_foot | 对应ankle_roll_link（完整ankle链） |

   两套骨架臂轴、关节自由度和比例不同，不能只按表复制四元数。
   Humanoid固定手节点、G1无主动颈关节及少一个踝自由度等不可一一对应项，需明确处理并记录损失。

2. **评分接口**：接入`AMPEnv.get_disc_obs_space/_compute_disc_obs_demo/_update_disc_obs`，
   让SMP收到10×114的规范表示，复用原`compute_disc_obs`坐标、时序和特征顺序；
   actor观察、29维动作、G1物理状态与诊断仍保持G1原生表示。
   `SMPAgent._build_prior_model`因此可严格加载原权重，原normalizer及SDS实现保持不变。

3. **GSI反向适配**：原先验生成Humanoid序列后，按原骨架解码，再做满足G1关节范围和脚底接触的序列重定向，
   由同一序列计算速度。接入`SMPEnv._disc_obs_to_motion_frames/_motion_frames_to_init_states`或对应适配分支。
   同时正确填充G1原生历史缓冲与映射后的评分特征；不能关闭GSI或仍用两段动作初始化冒充原版。

4. **单独启动配方**：在适配通过后新增明确的LaFAN运行配方，复用原版agent/任务奖励，
   显式声明G1控制修复及Newton差异。更新对应`evaluate.py`配置契约和prior资产检查；
   现有`workflow.py`强制两段数据及210维的检查只适用于旧流程，不应全局删除这些保护。
   不覆盖已完成模型；新目录从头RL，环境数按负载测试结论，完整预算按上表。

## 最小验证门槛

- 原Humanoid身份映射：同一窗口的114维特征和固定噪声下SDS与原实现一致；原权重严格加载，
  测试前后权重及prior normalizer不变。不得使用`strict=False`或随机补权重掩盖不兼容。
- G1正向映射：静止、单轴动作、左右对称、前后/侧向运动的符号与节点正确；关键点与规范FK一致，
  世界平移/航向变换符合原表示的不变性，30Hz时序及速度自洽。
- GSI序列：批量生成并重定向，检查有限性、关节范围、脚底穿地、时序速度及接触质量，记录映射失败率；
  不以大规模丢弃不合适样本来暗中缩窄原先验。若不可行，应报告具体失败再选择替代路线。
- Newton通路：小规模评分/GSI/RL更新/模型保存加载全部通过；不发生维度错配、NaN、容量溢出。
  这只证明兼容通路，不能提前宣称自然运动或抗扰能力通过。

已检索第三方`SUZ-tsinghua/smp`发布的G1 LaFAN-run先验，但其接口为59维/帧、50Hz及q01/q99归一化，
不是本仓库原版LaFAN权重或210维G1接口的直接替代，不能当作“已找到兼容作者权重”。
论文§8.2使用LaFAN非结构化跑步子集，并非整个LaFAN库；如果复用作者原权重，无需为此先重建原始数据清单。
只有日后明确改走新先验路线，才需要核实数据范围和重新训练预算。

## 主要依据

- [原版SMP说明](https://github.com/xbpeng/MimicKit/blob/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d/docs/README_SMP.md)
- [原版任务agent](https://github.com/xbpeng/MimicKit/blob/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d/data/agents/smp_task_humanoid_agent.yaml)
- [原版Dodgeball实现](https://github.com/xbpeng/MimicKit/blob/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d/mimickit/envs/task_dodgeball_env.py)
- [原版先验加载](https://github.com/xbpeng/MimicKit/blob/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d/mimickit/learning/smp_agent.py)
- [原版GSI解码](https://github.com/xbpeng/MimicKit/blob/fbe1a36e878f036d6bf5521defb8eb3b8ab4170d/mimickit/envs/smp_env.py)
- [论文§8.2](https://arxiv.org/html/2512.03028v3#S8.SS2)
- [第三方G1先验接口](https://github.com/SUZ-tsinghua/smp/blob/master/README.md)
