# SMP / Newton 实验与问题总记录

记录日期：2026-11-03，Asia/Shanghai。源码基准：`4a2265a3eb2e177bc9980b303f80f111ed681638`。

## 0. 本次决定与记录边界

- 用户要求暂停推进，先撤销全部未提交代码，再系统说明情况、讨论解决路线。
- 已将暂停前7个未提交文件还原至HEAD内容，包括logger、两套workflow、repair、测试、plan和repair文档。
- 已提交的G1 A/B实验实现仍属于HEAD；本次没有要求撤销历史提交，因此没有移除。
- 本次新增/更新内容仅为情况说明文档；没有改变训练、物理、奖励或评测代码，没有启动新训练或评测，没有提交或推送。
- 上一轮准备的日志解析修复也已撤销。它的原因、验证经历及当前仍存在的限制在第8节保留，不能误认为远端已修复。
- 文中启动命令均为历史流程与复查索引，**不是当前批准执行的训练计划**。暂停期间不自动执行下一轮实验。
- 本记录汇总当前可见对话、已有plan/流程文档、源码与用户反馈。缺失的原始文件明确标记；不补造未知数值、日期、结论或新版本性能。
- 暂停后的情况与待讨论事项另记于根目录 `SMP_PAUSE_NOTES.md`；原有plan暂不整理或更新。旧流程文档里的“下一步”“待远端运行”可能早于实际结果。

Newton1.0.0下的训练属于探索性尝试，本记录仅保留情况、结论、问题与流程参考，不作为需要恢复的正式实验归档。
按用户要求，不在仓库保留原始反馈副本、来源清单、哈希存档或旧plan快照。
后续将开展新的训练；后端版本、数据、控制、奖励及验收标准需要重新确定，历史命令和模型不自动作为后续方案。

## 1. 环境、授权和协作方式

| 项目 | 已知情况 |
| --- | --- |
| 本地仓库 | `D:/ComputerScience/PROJECT/MimicKit` |
| 长训练机器 | Windows 11，单卡RTX A5000，24GiB |
| 训练机仓库 | `C:/Project/MimicKit` |
| Conda | `mimickit-newton`；用户主要使用Anaconda Prompt / CMD |
| 本地小检查 | `D:/CondaEnvs/envs/mimickit-newton/python.exe`，RTX5070 12GB；不在本地做完整训练 |
| 依赖记录 | Torch2.10.0+cu128、Newton1.0.0、Warp1.12.0、MuJoCo3.5.0、MuJoCo-Warp3.5.0.2、diffusers0.36.0 |
| 动力学链路 | MimicKit → Newton → SolverMuJoCo → MuJoCo-Warp/Warp GPU计算 |
| 控制 | 位置PD，30Hz控制、240Hz仿真；每控制步8个仿真步 |
| 远端 | origin为Ekubooo/MimicKit，上游xbpeng/MimicKit |
| 源码同步 | 本地准备；用户自行commit/push；训练机pull，不代替用户推送 |
| 结果同步 | output、资产、下载权重多被Git忽略；pull不会获得训练模型或prior |

Newton、MuJoCo-Warp、Warp不是三套可互换的任务后端：当前solver经后两者执行GPU物理。
Warp是计算基础；更新它不等于自动改进奖励、动作数据或策略质量。

协作约定：每次提出新训练，必须给出完整CMD启动、查看、评测、可视化命令、反馈文件及打开命令、精确采样预算与时间依据。
每个实验独立输出目录，不覆盖模型；用户反馈优先短文本和动作观察，不必上传巨大GPU CSV或model.pt。
历史checkpoint不包含完整优化器/RNG/训练计数恢复状态；加载model继续训练不等于精确断点恢复，新增样本需单独计数。

### 1.1 Windows操作说明

```bat
cd /d C:\Project\MimicKit
conda activate mimickit-newton
```

CMD不能直接执行PowerShell语法。`powershell`是代码块语言标签时不应照抄为命令。
若确需PowerShell辅助，可在激活Conda的Prompt中调用 `powershell -NoProfile -Command "..."`；无需在PowerShell注册Conda。
一条训练命令占用当前窗口；等它结束，或在另一个Prompt看进度/监控。
Windows锁屏、单纯关闭屏幕通常不停止进程；必须保持系统不睡眠、不休眠，避免重启/注销。

可选GPU监控（独立窗口，有输出后持续刷新；Ctrl+C仅结束监控）：

```bat
nvidia-smi
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5
```

1024/2048是并行环境数，不是GPU数。samples/s指每秒收集的环境转移样本，包含多环境合计，
不等于仿真Hz、PPO更新速度或独立动作数量。环境翻倍不保证吞吐翻倍：仿真/SDS/PPO、同步、GSI及测试均占时间。
早期Location记录中1024约13.3k、2048约17.6k samples/s，约1.32倍；不能推广到G1所有任务。
log4b某时刻80℃、软件功率限制有效、未报告硬件热降频；不能据单次快照判定整轮散热或性能瓶颈。
GPU CSV可能包含训练前后长时间空闲，统计训练负载须按起止时间截取。

论文v3附录B写的是单卡A100/H100/RTX4090、1024并行环境；公开日志/部分args使用4096，
是不同的记录口径。不能据此保证“4090跑任意4096环境都完全没问题”；机器人、碰撞、SDS与容量都会影响显存和吞吐。
当前G1 A5000已验证的是2048环境长跑，未验证4096环境完整训练。
[论文实验设置](https://arxiv.org/html/2512.03028v3#A2)

## 2. 方法概念与任务范围

### 2.1 Diffusion → frozen prior → reward → RL

motion是姿态/速度参考数据；diffusion学习局部运动窗口分布。冻结的diffusion权重通过SDS计算SMP reward，
PPO再学习物理仿真中的动作控制。冻结不是额外训练或权重格式转换；采样GIF正常不等于物理策略可执行。
本地 `SMPAgent._build_prior_model()` 冻结参数，`_calc_smp_rewards()` 负责评分映射。

batch512表示每次diffusion更新采样512个动作窗口；10帧、30Hz覆盖约0.3秒局部运动。
400k×512是重复窗口采样量，不是2亿段独立原始动作；prior训练也不运行Newton。
两段仰卧/俯卧联合采样并不等于把两段逐帧硬拼接，更不等于运行时切换策略。

SMP reward在PPO采样后加入buffer。Train_Return/Test_Return主要累计环境task reward，
不是组合优化总回报。纯回旋踢task权重0且环境task reward为0，Return为0是预期，不说明训练未运行。
Episode_Length通常为控制步；30步为1秒。Wall_Time在这些训练日志中按小时记录，
不能把终端最后一行的小时值误读为秒或迭代耗时。

### 2.2 任务与demo

本次已实际训练的四个Humanoid SMP任务：单动作Spinkick、目标位置Location、方向/速度Steering、Dodgeball。
Steering默认是平地目标方向/速度控制，不会因名称就具备楼梯能力。
去除Dodgeball发球可以观察基础动作/平衡，但没有形成专门的随机外力平衡训练任务；球体冲击与拖拽也不同。
四任务之间不能按名字保证平衡强弱，需统一初始化、扰动、终止规则及指标再比较。

论文另讨论Target Speed、Object Carry、Stair Traversal、Getup、风格条件/组合、单动作模仿与G1实机行走；
不能据论文演示就认为本仓库已有所有完整预设。[SMP论文v3](https://arxiv.org/html/2512.03028v3)

仓库还含DeepMimic、AMP、ADD、ASE、AWR、LCP，以及vault、view_motion、DOF测试等入口；
对应Humanoid、SMPL、持剑盾Humanoid、G1、Go2、Pi Plus等骨架，配置存在不代表已适配当前Newton或已有对应SMP权重。
当前入口可查看仓库args目录，以实际源码为准；不额外保留目录快照。

view_motion是运动学播放，不使用RL模型、不验证重力下平衡：

```bat
python mimickit/run.py --arg_file args/view_motion_g1_args.txt --engine_config data/engines/newton_engine.yaml --mode test --num_envs 4 --visualize true --test_episodes 8
python mimickit/run.py --engine_config data/engines/newton_engine.yaml --env_config data/envs/view_motion_spinkick_g1_env.yaml --mode test --num_envs 4 --visualize true --test_episodes 8
python mimickit/run.py --engine_config data/engines/newton_engine.yaml --env_config data/envs/view_motion_locomotion_g1_env.yaml --mode test --num_envs 4 --visualize true --test_episodes 8
```

最后两条引用派生motion，需已有对应 `output/g1_*_assets/`；不能把缺派生文件当成模型失效。

### 2.3 风格混合、起身和多模型

风格混合是组合条件diffusion的预测，例如按上下身特征掩码组合风格，再作为reward训练策略；
它不等于平均RL checkpoint，也不直接实现“走路→摔倒→起身→继续走”的时间顺序管理。
当前仓库没有一套已接通、可直接调用的完整风格条件/混合预训练流程。
若要恢复任务，更自然的工程方案是独立起身模型加运行时状态机，按倒地/站稳状态切换，并保留真实物理状态。
切换要适配各模型观测、归一化、动作尺度及过渡；单模型多任务则需任务条件和额外联合训练。
这两条路线均未完成，G1没有现成起身策略。

## 3. 历史Humanoid训练记录

此前原物理后端策略迁移Newton后大范围失效；在Newton1.0.0下重新RL，主要复用公开冻结prior，
并非每次物理后端迁移都重新训练diffusion。
Spinkick使用公开 `smp_prior_spinkick.pt`；三任务默认LaFAN1 prior，另有locomotion权重，不能混淆默认与可选。
[上游SMP用法](https://github.com/xbpeng/MimicKit/blob/main/docs/README_SMP.md)

### 3.1 早期通路与逐步扩展

| 反馈 | 已知内容与意义 |
| --- | --- |
| ExitError | 8环境、256samples、一次PPO更新的通路测试；model存在不代表技能已学会。测试10回合后pyglet Win32 on_deactivate弱引用AssertionError |
| log2 | Location1024环境，500更新，16,384,000samples，Wall_Time0.343h；动作仍僵直/摔倒属于未收敛，不能判训练没执行 |
| log3 | Location2048环境，100更新，6,553,600samples，0.104h；吞吐17.56k samples/s，仍属短测 |
| log4a/b/c | Location2048，从头约1亿样本，1526更新，1.569h；用户已观察走得稳。100回合return207.11，512.99控制步≈17.10s |
| log6/8 | Dodgeball约3000迭代、Location约120迭代的中途反馈，不是完整结果 |

退出异常与学习失败必须分开：ExitError在测试完成后触发窗口清理异常；
已有 `docs/RESOURCE_LIFECYCLE.md` 记录显式close/finally清理及验证边界。
关闭窗口/中断测试不会自动完成评测汇总；test模式输出目录空也不能单独证明模型加载失败。
用户log10提供三合一配置并提到一次命令退出报错，但不能凭该文本还原未提供的全部异常堆栈。

### 3.2 四项完整训练

| 任务 | samples | 耗时 | seed123单环境100回合 | 用户观察 |
| --- | ---: | ---: | --- | --- |
| Spinkick | 400,031,744 | 5.844h | 平均9.823s/10s；task return0 | 效果很好，强拖拽仍可能摔倒 |
| Dodgeball | 1,337,065,472 | 21.367h | 平均20.00s/20s；return491.976 | 运行效果尚可 |
| Location从头完整 | 2,280,783,872 | 50.133h | 平均19.267s/20s；return441.854 | 平衡尚可，最终动作野、古怪 |
| Steering | 2,267,676,672 | 47.229h | 平均19.815s/20s；return519.012 | 目测良好 |

累计约125 GPU小时。上述旧Humanoid实测不能直接作为G1时间或稳定性保证。
Location迭代6000当时测试约19.46s，用户确认动作更像人；最终SMP约0.0005–0.0008、SDS约5.6，
任务成绩和先验匹配分化。归一化、奖励尺度/指数饱和、替代策略、动力学与优化过程均是候选原因，尚无唯一根因；已暂缓。

### 3.3 已有模型与可视运行命令（远端CMD）

以下只读载入用户已训练模型，不是原后端示例权重。output缺失时先找回原结果；不要用其他骨架权重替代。
各命令显式覆盖旧args可能指定的Isaac后端；使用保存agent配置以保持实际prior路径。

```bat
cd /d C:\Project\MimicKit
conda activate mimickit-newton
python mimickit/run.py --arg_file args/smp_humanoid_args.txt --engine_config data/engines/newton_engine.yaml --agent_config output/a5000_spinkick_full_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_spinkick_full_r01/model.pt --test_episodes 8
python mimickit/run.py --arg_file args/smp_dodgeball_humanoid_args.txt --engine_config data/engines/newton_engine.yaml --agent_config output/a5000_dodgeball_2048_full_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_dodgeball_2048_full_r01/model.pt --test_episodes 8
python mimickit/run.py --arg_file args/smp_location_humanoid_args.txt --engine_config data/engines/newton_engine.yaml --agent_config output/a5000_location_2048_full_scratch_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_location_2048_full_scratch_r01/model.pt --test_episodes 8
python mimickit/run.py --arg_file args/smp_location_humanoid_args.txt --engine_config data/engines/newton_engine.yaml --agent_config output/a5000_location_2048_full_scratch_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_location_2048_full_scratch_r01/int_models/model_0000006000.pt --test_episodes 8
python mimickit/run.py --arg_file args/smp_steering_humanoid_args.txt --engine_config data/engines/newton_engine.yaml --agent_config output/a5000_steering_2048_full_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_steering_2048_full_r01/model.pt --test_episodes 8
```

注意：当前旧Dodgeball args引用的默认 `smp_dodgeball_humanoid_agent.yaml` 不存在，以上已用保存agent覆盖；
不要直接照抄旧args而不检查解析后的路径。若当时保存env后来被改动，也应显式加载对应env_config。

旧任务100回合命令模板（将对应可视命令的参数改为如下模式，并用新输出文件名重定向）：

```bat
python mimickit/run.py --arg_file args/smp_steering_humanoid_args.txt --engine_config data/engines/newton_engine.yaml --agent_config output/a5000_steering_2048_full_r01/agent_config.yaml --mode test --num_envs 1 --visualize false --rand_seed 123 --model_file output/a5000_steering_2048_full_r01/model.pt --test_episodes 100 > output/a5000_steering_2048_full_r01/evaluation_seed123_record.txt 2>&1
```

框架多环境test可能向上取整、超出请求回合数；需要恰好100首回合时使用后文专用评测工具。

## 4. Humanoid Getup：diffusion与RL失败记录

2026-11-13 重新推进：用户授权先用默认 Humanoid / Newton 1.7 排查并短训，保留旧冻结 prior。
真实站稳轨迹在旧权重下得分低于旧策略贴地轨迹，已准备独立 0.5/0.5 权重实验，原配置保留。
计划 4096 环境、100,007,936 samples；当前待用户 push 后远端拉取启动，尚未启动短训。
诊断证据、复现边界和完整命令见 [Getup 新版恢复实验](docs/GETUP_NEWTON17_RECOVERY.md)。

### 4.1 数据、预算与结果

| 阶段 | 设置/结果 |
| --- | --- |
| 数据 | faceup约3.77s、facedown约3.03s，共6.8s；Humanoid骨架，等权联合采样；不混用SMPL/G1 |
| 50k prior | batch512、lr1e-4，2026-10-19 21:09–21:27，18min，Loss0.2749→0.04247，EXIT0；用户看生成尚可 |
| 400k prior | 从头训练，batch512、lr1e-4、统计10k窗口，2026-10-19 22:43至10-20 01:13，2h30m12s，Loss0.0228136 |
| 离线log14 | 参考相对SDS约1；静态faceup末帧244、facedown末帧94、默认站姿615；是诊断相对值，不是真实RL reward |
| RL规模 | 2048×32×15,259=1,000,013,824samples，项目预算；不是已核实的论文Getup精确样本数 |
| RL完成 | 2026-10-20 02:53–22:27，19h34m37s，EXIT0，最终15258；78行日志有限，起身/结束站稳/保持5s指标均0 |
| 用户观察 | 最终地面小幅抽搐；5000会循环大幅打挺但无法站起；5800仰卧不动、俯卧尝试后不动 |

400k prior输出 `output/a5000_getup_prior_joint_400k_scratch_r01/`；
RL输出 `output/a5000_getup_2048_full_r01/`。50k输出另保留。
动作很短、模型较小、局部窗口重复采样，prior训练快于长时物理RL不奇怪；仅用动作时长不能预测耗时，主要看更新次数、网络/batch与导出开销。

### 4.2 实现与结论边界

90%随机参考帧、10%增广倒地；GSI关闭，task/SMP0.2/0.8、SDS scale8。
task由高度/向上速度0.3/0.7组成；实现选择deficit模式，以max缺口对应抬身文字目标。
论文印刷min公式与文字方向存在疑点，未经作者确认，不宣称逐公式复现。
允许手/膝/身体触地及再倒地，20s超时；成功要求连续5s满足高度、直立、脚接触和低速度条件，成功条件不直接进入task奖励。

日志5000 task回报418.94，SMP0.00422；5800回报17.93，SMP0.32163。
分数提升未转化起身。可能有低姿态速度项替代解、静态站姿先验冲突、探索/奖励欠约束、控制或指标问题；
均未获唯一根因证实。关掉摔倒reset只是允许恢复尝试，不会自动学会恢复。
本路线与多模型切换已暂缓，不将失败模型用于G1重训。

历史启动、反馈详见 `docs/GETUP_PRIOR_WINDOWS.md`、`GETUP_PRIOR_400K_WINDOWS.md`、
`GETUP_PRIOR_CHECK_WINDOWS.md`、`GETUP_SHORT_MOTION_ANALYSIS.md`、`GETUP_RL_WINDOWS.md`。

归档命令（当前暂停，不执行训练）：

```bat
python tools/diffusion_model/train_tinymdm.py --mode train --cfg_path tools/diffusion_model/config/tinymdm_getup_400k.yaml --out_dir output/a5000_getup_prior_joint_400k_scratch_r02 --device cuda:0
python tools/getup/train_getup.py --stage full --check_only
python tools/getup/train_getup.py --stage smoke
python tools/getup/train_getup.py --stage full
python tools/getup/evaluate_getup.py --model_file output/a5000_getup_2048_full_r01/model.pt --init starts --episodes 4 --visualize --out_dir output/a5000_getup_2048_full_r01/view_starts_record_r02
python tools/getup/evaluate_getup.py --model_file output/a5000_getup_2048_full_r01/model.pt --init starts --episodes 100 --seed 123 --out_dir output/a5000_getup_2048_full_r01/evaluation_starts_record_r02
python tools/getup/evaluate_getup.py --model_file output/a5000_getup_2048_full_r01/model.pt --init augmented --episodes 100 --seed 123 --out_dir output/a5000_getup_2048_full_r01/evaluation_augmented_record_r02
python tools/getup/collect_feedback.py --run_dir output/a5000_getup_2048_full_r01
python tools/getup/diagnose_getup.py --model_files output/a5000_getup_2048_full_r01/int_models/model_0000005000.pt output/a5000_getup_2048_full_r01/int_models/model_0000005800.pt --out_dir output/a5000_getup_2048_full_r01/diagnosis_5000_5800_record_r02
```

用notepad打开相应feedback.txt、评测summary.txt或诊断summary.txt即可反馈；失败提供console.txt尾部。
已有目录非空时换新名称，不能覆盖原证据。

## 5. Newton新版迁移、资产与真机方向

### 5.1 版本迁移

尚未升级。plan记载2026-10-19曾查询Newton1.7.0、Warp1.18.0及Newton对应MuJoCo依赖；
这是历史版本信息，本次没有把它重新认定为2026-11-03最新稳定组合。
MimicKit当前验证基准仍为Newton1.0.0，新版收益和旧策略兼容性未实测。

可将迁移独立于训练：隔离环境 → API/索引/状态/控制/接触/重置/Viewer适配 → 用现有模型只读回归 → 决定调参或重训。
保持同一SolverMuJoCo体系时，预计比PhysX跨Newton更容易维持行为，但不保证无退化。
若骨架与特征保持，prior可先复用；升级不强制重训全部diffusion。

| 部分 | 工程重点 | 当前结论 |
| --- | --- | --- |
| Newton | ModelBuilder/MJCF、关节索引、state/control、solver、Viewer/sensor、reset | 主要后端适配入口；旧模型推理可先做 |
| MuJoCo-Warp | solver参数/约束接触容量、导入差异、接触力、数值/性能回归 | 实际动力学可能变化；按Newton支持矩阵配套 |
| Warp DSL | 内核/API、数组互操作、图捕获、编译缓存与CUDA兼容 | 底层兼容工作；不自动修复任务目标和数据缺口 |

历史工程粗估为初版数天、完整回归1–2周，不含重训GPU时间；尚不是已验证工期。
物理版本迁移、G1新骨架适配、reward设计应分开处理，避免同时改变后无法归因。

### 5.2 G1、H1-1与真机

本地已有G1 29DoF MJCF、网格与部分motion；另有DeepMimic/ADD/LCP的G1权重，不是SMP prior，
不可凭文件后缀同为.pt就互用。H1-1本轮未建立经过核实的资产、观测、PD、训练或部署链路。
当前成功只属于G1仿真；不等于论文实机方案，也不等于可直接下发电机。

若以后上真机，需确认硬件型号/关节版本、方向/零位、限位、力矩/速度上限、PD与延迟、控制频率、
观测可获得性及模型归一化，加入随机化/状态估计，并通过仿真回放及sim2sim验证。
真实电机系统辨识是否必需取决于名义参数与实机偏差，不能简单断言每个任务都必须重新做完整电机测试。
若差异明显再标定与重训，当前并未实施真机验证。
宇树官方提供Train→Play→Sim2Sim→Sim2Real流程；其示例G1控制维度与本仓库29DoF全身方案也需核对，不能直接替换。
[Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)

## 6. G1数据、prior与完整规模的实际含义

### 6.1 机器人与动作

G1为29个驱动DoF、31身体节点、36网格引用，XML质量35.212142kg；
Humanoid为28DoF。G1 prior窗口2100=10×210维，Humanoid1140=10×114维，权重不能直接通用。
相同2100维也不足以兼容：回旋踢global_obs=True，locomotion=False。

| 数据 | 原始信息 | 适配处理 |
| --- | --- | --- |
| G1 spinkick | 60Hz、78帧、约1.283s | 左踝roll全部78帧超限，最大0.46814rad；投影至-0.2618，损失该关节变化，未做完整物理重定向 |
| G1 walk | 120Hz、125帧、1.033s、WRAP | 未需限位投影 |
| G1 run | 30Hz、25帧、0.8s、WRAP | 左右踝roll各8帧投影，最大改动0.049138/0.065609rad |

保留原motion/XML，派生文件在 `output/g1_spinkick_assets/` 与 `output/g1_locomotion_assets/`。
投影保证采样角度范围，不证明接触、速度连续性、动量、平衡或电机可执行性。
locomotion仅两段合计1.833s、等权采样；并非完整G1 LaFAN1重定向，也没有证实包含足够站立、侧移、转身、制动或恢复动作。
这属于覆盖风险，尚不是已证实根因。论文的躲避能力也由locomotion prior配合task涌现，
不能说必须提供躲球参考动作才能训练；“缺少躲球数据”本身不足以解释当前失败。
[论文多任务实验](https://arxiv.org/html/2512.03028v3#S8.SS2)

### 6.2 对齐依据

| 公开权重/日志 | 核实计数 |
| --- | --- |
| 公开Spinkick prior | EMA50,000更新、normalizer100,000帧=10,000窗口 |
| 公开LaFAN/locomotion prior | EMA200,000更新、normalizer200,000帧=20,000窗口 |
| 公开Spinkick RL最后日志 | 380,239,872samples；4096×32×2901 = 2048×32×5802 |
| 公开Dodgeball RL最后日志 | 1,337,065,472samples；4096×32×10201 = 2048×32×20402 |

证据在两套 `tools/g1_*/reference_budget.json`，不是只读默认YAML推测实际训练步数。
论文正文diffusion区间400k–800k与发布checkpoint计数不同；G1当前按公开权重更新数与日志样本数对齐。
同样样本量、网络和更新数，不等于同数据、同机器人、同reward、同探索分布或同论文效果。

locomotion prior SHA256：`b14e2857e3dad2af734f80e2d7fde2bba6f80de29182e2e12b5cd01be76efe3d`。
XML SHA256：`c44d4dfcc07f179404a60d66f3c8ccc55210a8c64d8fffbc6c380515c6135c5b`。
dataset原始文件本地LF与远端CRLF哈希不同；各机器prior与运行时一致的情况下，不能把换行差异当成动作数据变化。

## 7. G1训练与评测登记

### 7.1 回旋踢：已获得可用仿真基线

| 阶段 | 规模/耗时/结果 |
| --- | --- |
| prior | 50k，batch512、lr1e-4；21m29s，正常完成 |
| pilot | 2048×32×400=26,214,400samples；34m44s；用户见旋转/踢腿尝试但快摔、跳不起来 |
| full | 380,239,872samples，5802更新，最后5801；9h56m53s，2026-10-30 08:35:36结束，EXIT0 |
| 评测 | reference starts/random各100首回合，均10s、0失败；不能直接叫100%踢腿成功率 |
| 用户动作验收 | 基本都完成回旋踢，效果不错；尚非静态站立启动、任意扰动或真机验证 |

prior：`output/a5000_g1_spinkick_prior_50k_r01/`；
策略：`output/a5000_g1_spinkick_2048_full_r01/`。

### 7.2 Dodgeball：prior、pilot、full

| 阶段 | 规模/耗时/结果 |
| --- | --- |
| locomotion prior | 200k、batch512、lr2e-4、统计20k窗口；10-31 00:43–02:19，1h36m7s，Loss0.012836，EXIT0 |
| 原pilot | 26,214,400samples、400更新，40m30.5s；无报告NaN/容量溢出；GSI100回合平均3.423s，全部失败，87命中/13非脚接触 |
| 原full | 1,337,065,472samples、20402更新；10-31 14:13至11-02 04:09，37h56m2s，EXIT0、验证通过 |
| GSI100评测 | 平均19.466s，95%达到20s，5命中、0非脚接触，task480，SMP0.1033，raw SDS2.0633 |
| reference100 | 平均19.665s，98%达到20s，2命中、0非脚接触，task482.4，SMP0.09735，raw SDS2.089 |
| 参考窗口控制 | SMP0.99535，raw SDS0.001188，策略轨迹明显偏离先验 |

full后期训练SMP约0.02、SDS约2.34。最后20401的SMP0.322来自前次输出/测试重置后的首段rollout，
不表示后期动作自然度突然恢复。SDS normalizer约100,007,936计数是跨过1亿阈值的一批超量，符合冻结逻辑。

prior路径 `output/a5000_g1_locomotion_prior_200k_r01/`；
原pilot/full分别 `output/a5000_g1_dodgeball_2048_pilot_r01/`、`output/a5000_g1_dodgeball_2048_full_r01/`。
full策略SHA256：`e8051b6bd8346dcf95391c8cdc6ae7a7ec16313020585c9b98daff53bf9d8573`。

用户确认：开始动作正常，第一次躲避后腿部像卡住、动作怪异、容易倒；**无拖拽/外力也会出现**。
因此高task return/timeout不能代替动作质量验收，G1 Dodgeball目前不算通过。

配置保持20s、1球、半径0.1m、约0.838kg、8–10m距离、12–15m/s、1–4s重发，目标torso_link；
仅两脚允许触地，task/SMP0.5/0.5、SDS scale6、GSI开启。
Newton单独容量1024/256，旧任务默认450/150；不是提升电机强度。
2048环境有G1和球两类物体，构建4096个obj不表示4096环境。

### 7.3 原full关闭/开启发球诊断（11月3日，reference、seed123、各16首回合）

| 指标 | 不发球全程 | 发球组全程/首球发射后 |
| --- | --- | --- |
| 20s存活 | 16/16 | 16/16 |
| SMP / raw SDS | 0.689789 / 0.138229 | 全程0.096517 / 2.06791 |
| 左膝截断前目标越界 | 8.3% | 首球后85.7% |
| 左膝实际near_limit | 0% | 首球后86.5% |
| 右膝目标越界/实际near_limit | 0% / 0% | 首球后57.4% / 53.1% |
| 左/右膝跟踪误差 | 0.139 / 0.121rad | 首球后0.530 / 0.561rad |

两膝XML下限-0.087267rad，上限2.8798rad；原full实际最低左-0.224、右-0.233rad。
左膝首球后速度0.776rad/s，发球前约2.810；右膝仍明显运动，不能说所有关节冻结。
首球后约8.98%样本root倾斜>60°，头最低0.332m，非脚接触检测仍0。
这支持“策略应对球时持续施压限位”的机制，但不证明引擎停止、每次都倒地或传感器必坏。

diagnostic的near_limit包含实际超限；10Hz加终止帧采样，比例按样本加权；
分段依据首球**发射**，不是首次躲避或首次命中。
原始这两份文本在随后同名文件上传时已被覆盖；本节依据先前已读反馈及HEAD计划保存的摘要。

### 7.4 A/B修复pilot：完成，但技能未通过

两组独立从随机RL权重训练，复用同一200k prior，不加载原full。
共同改动：保留扩展PPO动作表示，实际PD目标截断物理角度限位；actor均值增加物理越界损失，尺度0.25rad、权重1。
targets不加姿态代价；pressure另对12腿关节near_limit加每步代价，margin0.03rad、权重0.25。
不是按时间检测机械“卡死”，而是每步位置邻近限位的代价。

| 项目 | targets | pressure |
| --- | --- | --- |
| 本次文件 | feedback (1).txt / diagnostics (1).txt | feedback.txt / diagnostics.txt |
| 运行时间 | 11-03 02:07:46–02:53:58，46m12s | 11-03 15:09:30–15:55:13，45m44s |
| 规模 | 2048×32×400=26,214,400 | 相同，约full预算1.96% |
| 进程/原验证 | EXIT0 / PASSED | EXIT0 / PASSED |
| reference带球16回合均时 | 1.729s | 1.623s |
| 失败 | 全16次非脚触地，无命中 | 相同 |
| 全程左/右膝near_limit | 29.1% / 47.6% | 1.5% / 13.8% |
| 实际发送目标越界 | 0% | 0% |
| 左/右膝请求越界 | 24.7% / 28.4% | 10.9% / 5.1% |
| SMP / raw SDS | 0.120077 / 1.435473 | 0.140554 / 1.441286 |
| 最后训练压力均值 | 0.0218，权重0 | 0.00876，加权0.00219 |

targets模型SHA256：`05f09f9443c490c7377ff88817f407620a2229812168b92b1cfd4a722e38578a`；
pressure：`34b9789c48f70c888387b699b0ed27a29adccec0b9802cde7428ab8ae2177835`。

用户仅看数回合且不知道对应组：仍出现关节限位卡住，是否减少不确定。保留这一不确定性，不擅自归到某一组。
所测限位压力下降是A/B结果；没有平衡/躲避或风格改善证据。大部分轨迹在发球前，
不能与原full长时首球后百分比直接比较，也不能把GSI原pilot与reference修复pilot当作等条件效果比较。
两组都小预算、单种子；不能用它们判定所有限位约束一定失败或原full只需加训即可修好。

### 7.5 其余G1任务

G1 Steering和Location尚未训练，没有对应可用SMP策略或完整实测。
G1四任务目标只完成回旋踢仿真基线；Dodgeball仍有动作问题，当前暂停复盘。

## 8. 当前工具缺陷与证据限制

- A/B feedback显示 `rows=0 incomplete_rows=3`，repair metrics为空。原因：Logger固定最小宽25但无列分隔符，
  `G1_Limit_Pressure_Weighted_Mean` 超长并粘连后续Exp_Prob表头。
- 不等于没有训练：manifest、模型、runtime和console尾部均证明400更新执行完成。
- 原validator逐列zip；粘连表头可能漏掉最后一列检查。因此PASSED只代表原工具没有报告数值/容量问题，不能表述为所有列已全面审计，更不能当成技能成功。
- 上一轮曾实现兼容旧表头与新分隔符、缺列拒绝，并通过6项检查；**本次按用户要求全部撤销**。
  当前HEAD仍有上述缺陷，不能要求用户反复生成同一feedback就期待自然修复。
- 真实G1先前结果的feedback/summary曾同名覆盖；本记录保留已知情况与结论，不将原始反馈副本纳入仓库。
- 本地小规模集成只证明接口可运行；部分GSI接口测试替换成真实参考窗口，不能冒充正式200k prior学习/生成质量验证。
- 本地未训练/2000更新toy prior曾产生异常GSI高度、约束溢出和NaN；正式远端200k和完整RL没有报告同类数值失败。
  不能据toy prior失败认定正式prior损坏，也不能用正式训练进程成功证明生成分布完全正确。

## 9. 系统分析：怎样减少盲试

### 9.1 先纠正“对齐论文”的范围

| Dodgeball设置 | 当前G1公开参数路线 | 论文v3附录D.4 |
| --- | --- | --- |
| task/SMP权重 | 0.5 / 0.5 | 0.6 / 1.2 |
| SDS scale | 6 | 4 |
| 发球速度 | 12–15m/s | 20–25m/s |
| task形式 | 0.9距离奖励 + 0.1低平移速度奖励 | 带1.5m偏移的距离指数奖励 |

此差异已于2026-11-03重新核对。当前源码、公开配置与论文文字不能混写。
目前对齐“公开日志最后采样数、公开权重更新数”，不等于完整论文设置。
把论文参数改来不保证G1成功：机器人、motion分布和控制也不同，且论文更快球速可能加大难度。
[论文附录D.4](https://arxiv.org/html/2512.03028v3#A4.SS4)

### 9.2 已证实、合理推测、未证实

| 层面 | 当前证据 | 能得出的结论 | 不能得出的结论 |
| --- | --- | --- | --- |
| 引擎/通路 | G1回旋踢成功，多轮RL完成、模型可加载 | 引擎不是所有G1任务都不能运行 | 所有导入、力矩、接触定义都已正确 |
| 控制/限位 | 原full来球后膝目标越界、实际贴边和跟踪误差大 | 有持续限位施压行为 | 越界PD目标本身必然是bug；唯一根因已知 |
| 动作质量 | 高timeout却怪异，SDS显著远离参考 | task成绩与自然度分化 | 非脚接触sensor必坏、视觉每次均是真倒地 |
| 数据 | 两段共1.833s、走跑、踝投影 | 数据覆盖与通用LaFAN路线明显不同 | 数据短必失败、200k更新不足或必须单纯训更久 |
| A/B | pressure限位压力低，仍全部早倒 | 指标被抑制但技能未通过 | pressure已解决问题；400更新足以预测full结果 |
| GPU | 训练退出正常，有实测长跑 | 不是没执行训练 | GPU负载决定动作怪异，换4090一定修复 |

### 9.3 为什么“硬截断目标”不是已经确定的直接修复

PD力矩近似由 `Kp*(q_target-q)-Kd*qdot` 决定，目标可以被用来请求控制力矩，而非要求关节真的到达该角度。
目标超出几何限位不能自动归为索引错误。硬截断会改变力矩范围：在下限附近不能再通过更低目标请求相同伸展力矩。
因此它可能减少顶限位，也可能减弱平衡控制能力。此前A/B与原方案相比同时改变截断及actor正则，
又从头只训400更新；不足以判定哪项导致早倒。只有A和B之间才主要差在压力代价。

更值得一次明确审计的是动作定义与机器人控制：29DoF顺序、符号/零位、归一化尺度，导入后的实际Kp/Kd、
effort限制、motor gear/ctrlrange/actuatorfrcrange、armature，以及软限位约束参数。
XML中有这些参数不等于它们经过导入后必然按预期生效；目前没有足够证据宣布这里已发现某个确定错误。

### 9.4 为什么高任务回报可能仍很怪

当前task直接奖励远离球及低平移速度，没有直接约束直立、关节余量、恢复步态或连续动作自然度。
当SMP偏离参考且贡献偏弱时，策略可能找到task可接受的低姿态/僵硬替代解。
这是源码与现象支持的解释，不是已证明策略有意“钻漏洞”。原full不发球较自然、发球偏离，说明来球处理是关键条件。
SMP归一化与指数映射也影响有效奖励，名义0.5/0.5不保证两部分实际贡献相等。

### 9.5 更有目标的三条路线（待讨论，本次不执行）

| 目标 | 推荐优先方向 | 代价/边界 |
| --- | --- | --- |
| 研究SMP在G1的迁移能力 | 先完成控制/评测契约核对，明确采用公开版还是论文版；再补经过物理检查的G1动作覆盖和对应prior | 保留研究可比性；主要投入在资产/数据，而不只是增加步数 |
| 尽快得到稳健G1平衡/躲避 | 以已验证G1基础平衡/走跑控制为基础，再增加来球反应；动作可采用名义姿态附近残差PD，配套力矩/速度/平滑与关节余量约束 | 属于新的机器人控制工程路线，不再冒充原SMP任务逐参数复现 |
| 优先复现论文角色任务 | 保留Humanoid及原prior作为对照，把G1看作额外新实验 | 已有三项较好Humanoid基线；不能替代G1需求 |

宇树官方G1示例使用名义关节角加缩放动作、PD、位置限位/动作变化等奖励，并有sim2sim流程；
它是路线参考，使用12DoF资产，不是可直接加载到当前29DoF全身模型的成品。
[官方G1配置](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py)

可以用现成G1 DeepMimic/行走demo作只读控制与motion参照，但更换算法或初始化方式需独立登记，
也不能保证这些旧权重在当前后端直接成功。本次不下载新框架、不换算法、不重训prior、不修改奖励。

### 9.6 建议的一次性审计交付物

这是后续范围建议，不是已经执行的检查或新的训练命令：

1. 一份配置差异表：G1/原Humanoid、公开仓库/论文、原full/A/B，包含数据、动作、PD、reward、GSI、normalizer与评测规则。
2. 一份控制链核对结果：XML→导入后的29DoF顺序、目标、实际角度/速度/力矩/限位约束，外加位置差分与reported速度一致性。
3. 一份任务验收定义：命中/碰撞、非脚接触、姿态、真实侧移、动作恢复、风格与能耗，不能只看20秒timeout。
4. 对已有checkpoint进行同条件轨迹复核，明确首次发球、接近、接触、异常姿态的时间关系；不是再重复几次“看起来如何”。
5. 用上述证据选择一条路线，写清主要改变、预测、预算、停止条件；只在可归因且通过前置检查时训练。

没有足够证据承诺“一项参数改动必定直接修复”。更好的做法是先使问题和目标可判定，避免每次只按一个表面指标改奖励。

## 10. G1完整工作流与命令归档

下列命令为已经使用的工作流，**当前暂停，不启动训练**。大部分r01目录已有结果，会拒绝覆盖。
未来若经用户决定复做，应统一新建r02目录，并同步prior_dir、model_file、eval与反馈路径。

### 10.1 提交、拉取和环境核对

本次只记录；未自动commit/push。历史同步顺序：本地检查diff→指定文件add→commit→push；远端检查工作区→pull。

```bat
cd /d D:\ComputerScience\PROJECT\MimicKit
git status --short
git diff --check
git diff --stat
```

```bat
cd /d C:\Project\MimicKit
git status --short
git pull --ff-only origin main
conda activate mimickit-newton
git rev-parse HEAD
python -c "import importlib.metadata as m,torch; print({p:m.version(p) for p in ['torch','newton','warp-lang','mujoco','mujoco-warp','diffusers']}); print(torch.cuda.get_device_name(0))"
```

### 10.2 G1回旋踢

```bat
python tools/g1_spinkick/workflow.py --stage check
python tools/g1_spinkick/workflow.py --stage prior-smoke
python tools/g1_spinkick/workflow.py --stage prior
python tools/g1_spinkick/workflow.py --stage sample
python tools/g1_spinkick/workflow.py --stage rl-smoke
python tools/g1_spinkick/workflow.py --stage pilot
python tools/g1_spinkick/workflow.py --stage full
```

smoke只做一次PPO更新，不用于技能质量；pilot/full均从头，不是依次续训。完整RL5802更新。

已训练full可视化与精确评测：

```bat
python mimickit/run.py --arg_file args/smp_spinkick_g1_args.txt --agent_config output/a5000_g1_spinkick_2048_full_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_g1_spinkick_2048_full_r01/model.pt --test_episodes 8
python tools/g1_spinkick/evaluate.py --model_file output/a5000_g1_spinkick_2048_full_r01/model.pt --init starts --episodes 100 --seed 123 --out_dir output/a5000_g1_spinkick_2048_full_r01/evaluation_starts_record_r02
python tools/g1_spinkick/evaluate.py --model_file output/a5000_g1_spinkick_2048_full_r01/model.pt --init random --episodes 100 --seed 123 --out_dir output/a5000_g1_spinkick_2048_full_r01/evaluation_random_record_r02
python tools/g1_spinkick/workflow.py --stage feedback --run_dir output/a5000_g1_spinkick_2048_full_r01
notepad output\a5000_g1_spinkick_2048_full_r01\feedback.txt
```

starts使用原动作首帧及速度；random为随机参考相位；均非任意静态站立。
反馈包括prior/full的feedback和两组summary.txt及动作是否完整，不把timeout冒充踢腿成功。

### 10.3 G1 Dodgeball原路线

```bat
python tools/g1_dodgeball/workflow.py --stage check
python tools/g1_dodgeball/workflow.py --stage prior-smoke
python tools/g1_dodgeball/workflow.py --stage prior
python tools/g1_dodgeball/workflow.py --stage sample
python tools/g1_dodgeball/workflow.py --stage rl-smoke
python tools/g1_dodgeball/workflow.py --stage pilot
python tools/g1_dodgeball/workflow.py --stage full
```

真实prior校验包含EMA预算、骨架/特征/频率、数据/资产哈希；不得用toy prior做正式GSI。
workflow自动记录console、runtime、manifest、配置与validation；当前日志解析限制另见第8节。

full可视化与精确评测：

```bat
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --agent_config output/a5000_g1_dodgeball_2048_full_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --test_episodes 8
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --init gsi --episodes 100 --seed 123 --out_dir output/a5000_g1_dodgeball_2048_full_r01/evaluation_gsi_record_r02
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --init reference --episodes 100 --seed 123 --out_dir output/a5000_g1_dodgeball_2048_full_r01/evaluation_reference_record_r02
python tools/g1_dodgeball/workflow.py --stage feedback --run_dir output/a5000_g1_dodgeball_2048_full_r01
notepad output\a5000_g1_dodgeball_2048_full_r01\feedback.txt
```

只读发球开/关诊断：

```bat
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles on --joint_diagnostics --out_dir output/a5000_g1_dodgeball_2048_full_r01/diagnostic_on_record_r02
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles off --joint_diagnostics --out_dir output/a5000_g1_dodgeball_2048_full_r01/diagnostic_off_record_r02
```

off保留远处球体及观测维数，仅把发射时间推迟，不等于删除球体或重新训练无球任务。
工具冻结网络/normalizer，只取每环境第一回合；输出summary、episodes、diagnostics以及state/joint轨迹。

### 10.4 A/B已提交实现与命令

```bat
python tools/g1_dodgeball/repair.py --stage check
python tools/g1_dodgeball/repair.py --stage smoke --variant targets
python tools/g1_dodgeball/repair.py --stage smoke --variant pressure
python tools/g1_dodgeball/repair.py --stage pilot --variant targets
python tools/g1_dodgeball/repair.py --stage pilot --variant pressure
```

没有自动full阶段。两组各400更新、26,214,400samples；命令顺序第一组targets，第二组pressure。

```bat
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --env_config output/a5000_g1_dodgeball_targets_2048_pilot_r01/env_config.yaml --agent_config output/a5000_g1_dodgeball_targets_2048_pilot_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_g1_dodgeball_targets_2048_pilot_r01/model.pt --test_episodes 8
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --env_config output/a5000_g1_dodgeball_pressure_2048_pilot_r01/env_config.yaml --agent_config output/a5000_g1_dodgeball_pressure_2048_pilot_r01/agent_config.yaml --mode test --num_envs 4 --visualize true --model_file output/a5000_g1_dodgeball_pressure_2048_pilot_r01/model.pt --test_episodes 8
```

必须同时加载各自env与agent，不能只换model_file错用原SMP类/控制设置。

已有带球评测命令：

```bat
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_targets_2048_pilot_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles on --joint_diagnostics --out_dir output/a5000_g1_dodgeball_targets_2048_pilot_r01/diagnostic_on_record_r02
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_pressure_2048_pilot_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles on --joint_diagnostics --out_dir output/a5000_g1_dodgeball_pressure_2048_pilot_r01/diagnostic_on_record_r02
python tools/g1_dodgeball/repair.py --stage feedback --variant targets
python tools/g1_dodgeball/repair.py --stage feedback --variant pressure
```

feedback工具目前有第8节缺陷，不能将空repair metrics当成真实0。
每组反馈文件默认 `output/a5000_g1_dodgeball_{targets或pressure}_2048_pilot_r01/feedback.txt`；
评测反馈同目录诊断子目录的diagnostics.txt，用notepad打开，必要时保留summary.txt/原log.txt补证。

### 10.5 上一轮提出、尚未收到结果的三组评测（已暂停）

此前建议targets-off、pressure-off、原400更新pilot相同reference-on，目的为基础平衡与等预算对照。
用户本轮要求停止，**不要求现在执行，也不把这些结果写成已完成**。历史命令保留：

```bat
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_targets_2048_pilot_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles off --joint_diagnostics --out_dir output/a5000_g1_dodgeball_targets_2048_pilot_r01/diagnostic_off_seed123
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_pressure_2048_pilot_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles off --joint_diagnostics --out_dir output/a5000_g1_dodgeball_pressure_2048_pilot_r01/diagnostic_off_seed123
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_pilot_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles on --joint_diagnostics --out_dir output/a5000_g1_dodgeball_2048_pilot_r01/diagnostic_matched_seed123
```

## 11. 后续训练的反馈流程参考

对话约定的通用协作流程已单独整理在根目录 `TRAINING_WORKFLOW.md`，涵盖检测、准备、启动、查看、评测、GIF与反馈名单。
本节保留历史实验所需文件的参考说明，不代替该通用工作流。

每次区分三类完成：进程完成、预算完成、技能验收；EXIT0/有限指标只解决前两者的一部分。
以下是后续新训练可采用的运行目录与反馈内容参考，不要求把Newton1.0.0旧尝试的原始记录补存进仓库：

| 文件/内容 | 用途 |
| --- | --- |
| runtime.txt / run_time.txt | 起止时间、耗时、退出码；不要用model修改时间推算整轮训练 |
| run_manifest.json、commit、dependencies | 实际代码/版本、seed、环境数、命令、samples、prior SHA |
| engine/env/agent/diffusion_config | 实际运行参数；不能只用后来变动过的默认配置解释旧模型 |
| motion_audit、reference_budget | 数据修整、资产哈希、预算对齐依据 |
| log.txt、console.txt、validation | 完整趋势、错误/溢出与数值审计；反馈汇总可能有解析缺陷 |
| model.pt、int_models | 最终/中间策略，留远端，不覆盖；中期可能优于最终 |
| summary/evaluation、episodes | 初始化、种子、真实回合数、失败原因、SMP/raw SDS/参考控制 |
| diagnostics、state/joint_trace | 请求/发送目标、实际状态、限位、发球与终止时间关系 |
| 动作观察或短视频 | 哪个模型、是否拖拽、何时异常；“像卡住”不能代替关节实际运动证据 |

反馈优先stage的feedback.txt、独立summary或diagnostics.txt及简短动作说明；
错误先给console尾部；GPU CSV、巨量trace及权重按具体需要再提供，不作为每轮必传。
test不会自动产生训练模型或完整训练日志；需要记录时用显式重定向/专用评测工具。
同卡长训练期间避免同时可视化或大规模评测，监控窗口与训练窗口分开。
不同checkpoint的SDS normalizer可能不同，需同时比较raw SDS、参考窗口分数、任务/姿态与视觉，不能只按SMP单值挑模型。

### 11.1 可重复使用的记录字段

| 字段 | 本次填入 |
| --- | --- |
| 状态 | 暂停 / 待审计 / 实验中 / 预算完成 / 技能通过 / 技能失败 |
| 目的与假设 | 主要改变、预期可观察现象、反证与停止条件 |
| 基线 | commit、后端组合、机器人/数据、prior、策略初始化 |
| 预算 | num_envs×rollout×updates、总samples；新增或累计须写清 |
| 时间 | 开始、结束、实测耗时与评测/导出时间 |
| 验收 | 固定初始化/种子/回合数，任务、姿态、风格、控制与失败类型 |
| 证据 | 模型、实际配置、原始日志、汇总、视频路径 |
| 结论 | 已证实/支持但未证实/无法判断；下一步是否改变路线 |

## 12. 当前状态清单

- [x] 撤销暂停前全部未提交源码/测试及文档改动，恢复HEAD。
- [x] 保存当前对话实验记录、流程、运行命令、G1问题及结论边界。
- [x] 按用户要求移除原始反馈副本与附属存档，仅保留情况说明。
- [x] 保留Humanoid四任务、Getup失败、G1成功/失败实验的区分。
- [x] 明确公开配置/日志规模与论文设置不完全一致。
- [ ] 讨论并选择G1后续目标：SMP迁移研究、完整论文设置对照，或稳健机器人控制优先。
- [ ] 控制链与任务评测的一次性系统审计；本次只读源码复核不等于已完成原生动力学验收。
- [ ] A/B-off及原pilot等条件评测（此前建议已暂停，未收到结果）。
- [ ] G1 Dodgeball动作验收通过。
- [ ] G1 Steering / Location训练。
- [ ] Getup改进与运行时切换。
- [ ] 新版Newton迁移与旧模型回归。
- [ ] 真机验证。

## 13. 资料索引

- 根目录 `TRAINING_WORKFLOW.md`：对话约定的通用工作流，适用于后续不同任务与后端。
- 根目录 `SMP_PAUSE_NOTES.md`：暂停后的情况、最新反馈及未决事项，不是新实施计划。
- 根目录 `plan.md`：保留本轮整理前的原有计划，尚未按最新情况重整。
- `docs/G1_SPINKICK_WINDOWS.md`：G1回旋踢完整工作流。
- `docs/G1_DODGEBALL_WINDOWS.md`：原躲避球工作流，时间/状态可能早于实测。
- `docs/G1_DODGEBALL_DIAGNOSTICS_WINDOWS.md`：full发球开/关诊断及解释。
- `docs/G1_DODGEBALL_REPAIR_WINDOWS.md`：HEAD中的A/B定义与历史命令，当前不是成功修复方案。
- `docs/GETUP_*WINDOWS.md`、`docs/GETUP_SHORT_MOTION_ANALYSIS.md`：起身数据、prior、评分、RL与失败分析。
- `docs/RESOURCE_LIFECYCLE.md`：早期退出清理异常与验证范围。
- [MimicKit SMP上游说明](https://github.com/xbpeng/MimicKit/blob/main/docs/README_SMP.md)：LaFAN任务、GSI与参数入口。
- [SMP论文v3](https://arxiv.org/html/2512.03028v3)：特定版本的任务、风格组合与实验设置；不以主张取代本仓库实测。
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)：机器人控制及sim2sim路线参考，未接入本仓库。

现存历史log编号映射：ExitError为早期退出/单更新；log2–4为Location逐步短测；log5为Humanoid回旋踢；
log6–7为Humanoid躲避球；log8–10为Location全量/配置与自然度讨论；log11为Steering；
log12–14为起身50k/400k/离线评分；historical/feedback.txt及getup/complete_log.txt为起身RL失败反馈。
以上log编号仅用于解释历史对话，不表示仓库保留相应原始反馈。原图/GIF也未重新生成或存档。
