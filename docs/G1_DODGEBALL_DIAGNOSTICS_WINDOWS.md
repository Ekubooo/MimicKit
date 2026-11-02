# G1 躲避球：完整训练后腿部僵直的只读排查（2026-11-02）

完整r01在37小时56分2秒后正常完成1,337,065,472样本，无非有限值、无容量溢出。
GSI/参考初始化各100首回合，达到20秒比例95%/98%，平均19.47/19.67秒；
但用户在不施加外力时仍观察到第一次躲避后腿部卡住、动作怪异和摔倒，动作质量未验收通过。
两组非脚触地检测均为0，与目测存在差异，需要独立几何与关节轨迹，不能用存活率盖过实际动作。

训练后期SMP约0.02，独立确定性轨迹约0.10，真实参考约0.995；
task得分高与风格明显偏离同时存在。SDS统计在100,007,936样本冻结，符合配置，
不是normalizer停止工作的证据。末尾20401只有一次32步rollout，不能拿其SMP0.322代表全部后期行为。
prior仅walk/run两段、缺乏侧闪/停步/恢复等覆盖，可能限制当前任务的风格约束。

已查明的接口事实：G1位置动作边界使用以0为中心、按关节最大绝对限位再扩展1.4倍的范围；
CharEnv只按这个动作范围截断，再直接发送PD目标，不再次截断到XML物理关节限位。
因此Action_Bound_Loss小不代表目标未越过物理限位，策略可能持续顶住限位。
这是待轨迹核实的机制，尚不能据此断言本次已经顶死关节，更不能直接改XML或对旧策略强行限幅。

## 1. 同步诊断工具，不重新训练

本地CMD：

```bat
cd /d D:\ComputerScience\PROJECT\MimicKit
git add tools/g1_dodgeball/evaluate.py tools/g1_dodgeball/joint_diagnostics.py tests/test_g1_dodgeball.py docs/G1_DODGEBALL_DIAGNOSTICS_WINDOWS.md plan.md
git diff --cached --stat
git commit -m "Add read-only G1 dodgeball joint and geometry diagnostics"
git push origin main
```

远端CMD：

```bat
cd /d C:\Project\MimicKit
git pull --ff-only origin main
conda activate mimickit-newton
```

## 2. 同一模型、同一种子：发球与不发球各16首回合

```bat
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles on --joint_diagnostics --out_dir output/a5000_g1_dodgeball_2048_full_r01/diagnostic_on_seed123_r01
python tools/g1_dodgeball/evaluate.py --model_file output/a5000_g1_dodgeball_2048_full_r01/model.pt --init reference --episodes 16 --seed 123 --projectiles off --joint_diagnostics --out_dir output/a5000_g1_dodgeball_2048_full_r01/diagnostic_off_seed123_r01
```

每条结束再输入下一条。两组仅作诊断，不是训练，通常预留数分钟，以本机编译/运行时间为准。
使用真实walk/run随机相位初始化，关闭GSI以排除生成初始化的影响；策略权重不变。
off只把发射时间移到1e9秒，球仍保留在远处，观测维度不变；这不是重新训练无球策略。
网络、normalizer、奖励、PD和关节限位均不更新、不修改；原训练输出和旧评测保留。

10Hz记录真实角度、速度、送往PD的目标、物理限位、root/head高度、root up.z、非脚地面接触力，
并额外保留每个回合终止的瞬间，避免10Hz采样漏掉短暂接触冲量；比例按采样点而非回合加权。
分为发射第一球之前/之后；**第一球发射不等于第一次躲避或命中时刻**。
root z<0.45米、up.z<0.5仅是低姿态/倾斜提示，不自动当作摔倒率；贴近限位阈值为0.03rad。
扩展动作边界允许的目标超限也不单独证明故障；需结合实际角度停滞、速度和跟踪误差。

## 3. 反馈两份短文本

```bat
notepad output\a5000_g1_dodgeball_2048_full_r01\diagnostic_on_seed123_r01\diagnostics.txt
notepad output\a5000_g1_dodgeball_2048_full_r01\diagnostic_off_seed123_r01\diagnostics.txt
```

发这两份文本即可。保留同目录state_trace.csv、joint_trace.csv、diagnostics.json、episodes.csv和summary.txt，
暂不上传大CSV。若方便，可附不拖拽的短视频，显示初始正常到第一次动作异常的全过程。
失败时提供对应console.txt尾部；输出目录已存在时换r02，不覆盖。

## 4. 可选比较中期4000模型

```bat
python mimickit/run.py --arg_file args/smp_dodgeball_g1_args.txt --mode test --num_envs 4 --visualize true --agent_config output/a5000_g1_dodgeball_2048_full_r01/agent_config.yaml --model_file output/a5000_g1_dodgeball_2048_full_r01/int_models/model_0000004000.pt --test_episodes 8
```

如中期自然但最终僵直，可对中期模型也执行上面的带球诊断，model_file换为该checkpoint，
out_dir换为diagnostic_iter4000_on_seed123_r01。现有模型不含完整优化器/RNG恢复，暂不直接续训。

## 判读方向

- 目标持续越过物理限位、实际角度贴边、速度低且误差大：支持限位/PD持续施压导致的僵直。
- 目标本身几乎不变、实际位置能跟随：支持策略选择静态/僵硬姿态。
- 发球关闭仍明显异常：不只是在处理飞球观测时出错，需再对照初始化、风格与控制。
- 独立几何显示明显低姿态/倾斜而接触检测始终0：应验证接触/摔倒终止，不能把旧timeout当作平衡成功率。

上述均为定位线索，根因需远端真实checkpoint轨迹确认后再制定重训对照。
