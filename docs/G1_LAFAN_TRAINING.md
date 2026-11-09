# G1 Dodgeball：复用作者冻结LaFAN先验

实现日期2026-11-07；2026-11-09更新：远端完整RL已正常完成，512首回合评测完成，用户反馈“视觉效果尚可”。
这不是重新训练先验，也不是将两段G1动作先验更名为LaFAN。

## 对齐内容与工程差异

- 直接严格加载作者`data/models/smp_priors/smp_prior_lafan.pt`，SHA256为
  `93f14378cc9501e7ec2d52387c930320fcccc341580cbe69e17d5a61c667142f`。
  原网络、normalizer、EMA均不改，100个状态张量训练前后逐位一致。
- `smp_dodgeball_g1_lafan_agent.yaml`继承原版任务agent所有参数值：task/SMP=0.5/0.5，
  SDS scale=6，steps=[22,15,8]，10帧，GSI启用；PPO网络、学习率、rollout及更新次数不改。
- Dodgeball使用仓库原版legacy奖励、发球参数、20秒回合和早停规则，不沿用paper_targets_v1奖励。
- G1物理/actor继续29DoF、Newton 30Hz控制/240Hz物理。保留已验证的PD目标限位：
  物理边界内缩0.03rad，膝目标下界0.03rad；actor目标正则1.0/0.25rad，额外限位压力奖励为0。
  agent类、动作中心、G1刚体/驱动参数及目标保护属于明确的机器人适配差异，不宣称与Humanoid动力学相同。

`g1_lafan_adapter.py`将完整G1关节链合成规范Humanoid骨架，再用原`compute_disc_obs`构造114维/帧。
手臂T姿态、肘关节零点与左右符号、髋部静态旋转有显式坐标校准；不是取前114列或补零。
关键点严格按`head,right_hand,left_hand,right_foot,left_foot`排列，由规范骨架FK重建。
根位置/速度按中立姿态身高比缩放（1.11309），修正根高度保持实际脚底离地量；高度修正速度由同一历史差分获得。

反向GSI先用原骨架解码，再逐链执行有限位的旋转IK（阻尼最小二乘），按实际已解父链求子链。
根高度保持源序列脚底离地量并至少留2mm间隙；整段速度由最终G1序列以30Hz重新计算。
G1原生历史与映射后的评分历史同步初始化；GSI buffer及刷新数量/频率保持原版默认。
没有丢弃生成窗口、关闭GSI或替换成walk/run来隐藏兼容问题。

不能无损表示的部分：G1没有主动颈部，规范neck固定；Humanoid没有主动腕部，GSI中G1手腕取中立值，
规范手节点跟随前臂，SMP不直接约束独立腕部旋转；G1踝只有两个自由度，做可达朝向投影。
当前适配是运动学投影，不是动力学可行性证明或接触时脚底零滑动保证。

环境构造阶段仍加载G1 walk/run元数据，供父类在先验尚未构建时完成观察空间/reset；
launcher会准备这些旧派生文件。**它们不用于训练先验，也不作为正常RL初始化来源**：
agent在第一次训练reset前填满作者LaFAN生成的GSI buffer，后续所有训练reset均使用该buffer。
评测显式选择`reference/starts`仍表示旧动作对照；新配方默认`gsi`。

## 本地验证

RTX5070，本地Newton1.0.0；不能冒充远端新配方容量或技能效果验收。

- 单元测试覆盖肘/膝正负方向、限位内往返、原特征顺序、航向/平移不变性、脚底间隙、速度一致性、
  坏输入拒绝及原版参数/完整预算。
- `output/g1_lafan_preflight_20261107_r02/preflight.json`：真实作者先验生成1024段×10帧，未丢弃窗口。
  所有输出有限且关节满足限位；脚底最小高度0.00199985m；0.515%关节帧投影到保护边界。
  局部关节朝向误差均值0.02778rad（1.59°），P99=0.34105rad；自由度速度P99=6.544rad/s，
  最大14.239rad/s；最大根速度4.49m/s。映射窗口SDS均值0.08969。
  包含neck/踝等不可完整表示的误差，不能将均值解读为所有关节的误差上限。
- `output/g1_lafan_smoke_20261107_r02`：8环境、3次真实RL更新、768 samples；完整默认4096 GSI buffer及
  第0次1024窗口刷新；exit 0，精确预算与有限性验证通过，没有约束容量溢出。
  保存策略中的先验100个张量与原发布文件逐位相等。
- `output/g1_lafan_reload_20261107_r01`：保存模型严格加载，8个GSI初始化首回合、SDS及关节诊断完成。
  此策略仅训练3次更新，8/8摔倒、均时0.492s；这验证加载/运行通路，不代表技能成功。
- 初次smoke因沙箱内Warp缓存目录权限失败；改用工作区`output/warp_cache_lafan`后运行成功。
  第二次smoke退出时wandb临时目录清理有沙箱权限警告；退出码0、模型/日志有效。
  reload将TMP/TEMP也放到工作区后正常结束。这些是本地工具环境问题，不修改远端依赖。

## 已授权完整规模

4096环境 × 32 rollout × **10,201更新** = **1,337,065,472 samples**，最后Iteration=10200。
从随机RL权重/normalizer开始，复用作者冻结先验。该预算取自作者发布Dodgeball日志末行。
资源异常回退2048时改为20,402更新，保持相同samples，不在旧目录覆盖或将权重加载冒充完整断点恢复。

本轮已在用户push后远端拉取`466bb2c`启动；本机未运行完整训练。
A5000的4096容量及新适配链路启动检查通过，独立后台进程运行，UU会话保留。

远端CMD（`C:\Project\MimicKit`、已激活`mimickit-newton`），新目录示例：

```bat
python -u -B -m tools.g1_dodgeball.lafan --stage full --num_envs 4096 --out_dir output/g1_lafan_full_20261107_r01
```

目录存在且非空会拒绝覆盖。保存`run_manifest.json`、`runtime.txt`、`console.txt`、`log.txt`、
env/agent/engine配置、`model.pt`和中间模型；结束后检查精确预算、有限性、溢出、先验不变性。
若实际启动日期变化，使用当日新目录并记录真实路径。

可复现检查与结束后评测（各输出目录须为空）：

```bat
python -B -m unittest discover -s tests -p test_g1_lafan.py -v
python -u -B -m tools.g1_dodgeball.lafan --stage preflight --out_dir output/g1_lafan_preflight_remote_r01
python -u -B -m tools.g1_dodgeball.lafan --stage smoke --out_dir output/g1_lafan_smoke_remote_r01
python -u -B -m tools.g1_dodgeball.evaluate --model_file output/g1_lafan_full_20261107_r01/model.pt --out_dir output/g1_lafan_eval_r01 --episodes 512 --init gsi --joint_diagnostics
```

评测自动识别新配方并使用原始LaFAN资产，不接受旧`--prior_dir`覆盖。其reference SDS对照来自
另外生成的LaFAN窗口经G1适配后评分，不再用两段walk/run作为先验质量对照。

可视运行：

```bat
python mimickit/run.py --mode test --num_envs 1 --devices cuda:0 --visualize true --engine_config output/g1_lafan_full_20261107_r01/engine_config.yaml --env_config output/g1_lafan_full_20261107_r01/env_config.yaml --agent_config output/g1_lafan_full_20261107_r01/agent_config.yaml --model_file output/g1_lafan_full_20261107_r01/model.pt
```

## 2026-11-09：完整训练及最终评测

远端`output/g1_lafan_full_20261107_r01`，代码`466bb2c`，4096环境。
北京时间2026-11-07 17:07:46至2026-11-09 02:32:00，共120253.58秒（33小时24分14秒）。
训练及外层launcher均exit 0，`run_manifest.json.status=budget_completed`。
精确完成1,337,065,472 samples / 10,201更新，最后Iteration=10200。
日志/模型有限性、约束容量、作者冻结先验及normalizer逐位不变性均通过launcher结束验证。

最终模型SHA256：`1e3264c79221439c9cd7e849eb83dcf15fb3e0353d4370bd0478e1d9156966f9`。
最终训练内测试平均595.031控制步，即19.834秒；Test_Return=488.1433。
训练内并行测试实际累计4097回合，不能将该平均时长直接当成首回合存活率。

独立最终评测：`output/g1_lafan_eval_20261109_r01`，seed123，512环境，各取第一个完整回合，
LaFAN GSI初始化、发球开启、无学习，exit 0；模型重新加载及SDS诊断通过。

| 指标 | 结果 |
| --- | ---: |
| 满20秒回合 | 503/512（98.2422%） |
| 平均回合时长 | 19.7888秒 |
| 失败回合 | 9 |
| 命中检测回合 | 7 |
| 非脚触地检测回合 | 2 |
| 采样窗口SMP奖励均值 | 0.47898 |
| raw SDS均值 | 1.07233 |
| 另外生成并适配的LaFAN窗口raw SDS | 0.09722 |

命中/非脚触地是当前任务检测器结果，满时不等于所有来球都完成主动躲避。
不同先验、SDS normalizer与初始化下的SMP/raw SDS不直接与旧两段动作先验的值比较。

关节诊断为10Hz及额外终止帧，每侧膝101830个采样点：

- 左膝角最小0.11478rad，右膝最小0.00479rad；均无负角度、无贴物理限位采样。
- 所有关节实际发送PD目标越物理边界比例均为0。
- 腰pitch、右踝roll、腰roll近限位比例分别约10.383%、10.004%、5.215%；左踝pitch约3.393%。
- 对腰pitch/roll和左右踝逐回合检查连续同侧近限位采样：最长0.4秒（腰pitch、右踝roll）；
  腰roll最长0.2秒，左踝pitch/roll最长0.1秒。详细结果为`near_limit_spans.json`。
- 仍有实际短时超限，例如左踝pitch最小约-1.0757rad（物理下界-0.8727），腰pitch最大约0.6875rad
  （上界0.52）。目标保护不等于物理状态绝不超限，也不证明Newton限位已被彻底修复。

本次未发现旧策略那种持续负膝角顶死。腰/踝残余问题属于采样中短时近限位/超限，未见长时间持续顶住；
10Hz采样不覆盖全部240Hz子步或实际力矩。用户反馈“视觉效果尚可”。当前结论是LaFAN对齐链路和完整
G1 Dodgeball训练已跑通、现有评测表现可用；未新增鼠标外力抗扰测试，不据此声称外力恢复已达到Humanoid水平。
本次没有自动追加训练或改动生产代码。
