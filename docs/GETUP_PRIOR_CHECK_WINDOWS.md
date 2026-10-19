# 起身 prior：进入 RL 前的离线 SDS / 站姿检查

目标：检查已完成的 400k prior 对参考起身、静态姿态和受控扰动的评分。
不会训练模型、修改 checkpoint 或启动 Newton。检查完成后再设计 Getup RL。
入口：tools/diffusion_model/check_getup_prior.py。

## 本次检查覆盖什么

- 每段参考动作分层抽取 256 个窗口，等权合并为 512 个，不跨动作边界。
- 参考动作在三个噪声等级 22/15/8 上评分，使用模型保存的归一化器和 EMA。
- 重复 8 组噪声。同一参考窗口和它的扰动版本使用相同噪声与批大小，减少随机差异。
- 检查仰卧/俯卧起点与末帧的静态保持姿态，以及原 Humanoid init_pose 的站立候选。
- 0.15 / 0.60 rad 持续关节偏移共用同一扰动方向；0.35 rad 逐帧关节抖动；
  参考轨迹整体升高 0.75m。关节扰动后重新计算身体位置与相关速度。
- 这些扰动是用于测试评分敏感性的控制样本，不保证关节限制、接触或动力学可行。
- 平移/偏航变换检查观测接口；原始参考插值的轻微非单位四元数可能放大倒地姿态下的
  heading 敏感性。报告额外给出单位四元数诊断，实际评分输入保持训练入口原样。

这是训练分布与候选姿态的诊断，不是独立测试集泛化证明。
静态案例是单个姿态重复多次，分数离散程度来自噪声，不代表覆盖了几百种站姿。

## 评分如何解读

- SDS 越小，表示对该 prior 越接近参考分布；不是“成功概率”。
- Relative = 每个噪声等级的 SDS 除以参考均值，再在等级上平均，分母下限 1e-4。
  因此参考整体通常为 1；分母下限触发时可以小于 1。
- 真正 RL 使用策略数据自适应更新 SDS normalizer，并进行指数奖励映射。
  此处不把参考归一化后的 Relative 值冒充真实 RL reward，也不据此预测 RL 回报。
- 受控大扰动平均比对应参考分数更差，是希望看到的现象，不是严格必然的数学保证。
- 站姿候选分数显著高于参考尾部，提示任务目标可能与 prior 冲突，需要结合姿态和后续
  物理验证判断。脚本的 p95 提示不是论文验收阈值，也不会自动判定需要重训。
- 静止躺姿得分高不等于起身策略失败；RL 的高度/恢复任务奖励负责推动角色离开躺姿。

## 1. 同步

开发电脑 PowerShell：

~~~powershell
Set-Location D:\ComputerScience\PROJECT\MimicKit
git add tools/diffusion_model/check_getup_prior.py tests/test_getup_prior_check.py docs/GETUP_PRIOR_CHECK_WINDOWS.md docs/GETUP_PRIOR_400K_WINDOWS.md docs/GETUP_SHORT_MOTION_ANALYSIS.md plan.md
git diff --cached --stat
git commit -m "Add offline Getup prior score and standing diagnostics"
git push origin main
git log -1 --oneline
~~~

训练机 Anaconda Prompt：

~~~bat
conda activate mimickit-newton
cd /d C:\Project\MimicKit
git status --short
git pull --ff-only origin main
git log -1 --oneline
set PYTHONUTF8=1
set MPLBACKEND=Agg
~~~

保持当前 main 分支；如拉取报冲突，先保留改动并反馈，不强制覆盖。
没有新增第三方依赖，不需要重新安装环境。

## 2. 启动 400k 检查

通常几十秒到数分钟；默认只评估几百个窗口，不是长训练。
模型仍读取原来的 400k 输出目录，报告写入独立目录。

~~~bat
python -c "from pathlib import Path; import torch; paths=['output/a5000_getup_prior_joint_400k_scratch_r01/model.pt','output/a5000_getup_prior_joint_400k_scratch_r01/diffusion_config.yaml','tools/diffusion_model/check_getup_prior.py']; missing=[p for p in paths if not Path(p).is_file()]; assert not missing,missing; assert torch.cuda.is_available(),'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print('Checks OK')"
if not exist output\a5000_getup_prior_check_400k_r01 mkdir output\a5000_getup_prior_check_400k_r01
powershell -NoProfile -Command "$taskOutDir='output/a5000_getup_prior_check_400k_r01'; ('START '+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) | Set-Content -Encoding UTF8 ($taskOutDir+'/run_time.txt'); python -u tools/diffusion_model/check_getup_prior.py --cfg_path output/a5000_getup_prior_joint_400k_scratch_r01/diffusion_config.yaml --model_file output/a5000_getup_prior_joint_400k_scratch_r01/model.pt --out_dir $taskOutDir --device cuda:0 --num_samples 256 --batch_size 256 --noise_repeats 8 --seed 123 --diffusion_steps 22 15 8 2>&1 | Tee-Object -FilePath ($taskOutDir+'/console.txt'); $taskExitCode=$LASTEXITCODE; ('END '+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+' EXIT_CODE='+$taskExitCode) | Add-Content -Encoding UTF8 ($taskOutDir+'/run_time.txt'); exit $taskExitCode"
~~~

运行会打印 Scoring 1/10 至 10/10，最后有 CHECK COMPLETE。
Checkpoint EMA step counter 对这轮完整权重应为 400000。
EXIT_CODE=0 只代表检查程序正常执行，不代表已通过物理起身或站稳验收。
没有 num_envs 参数，不启动 2048 个仿真环境，不需要 GPU CSV。
同一报告目录再次运行会覆盖报告，不会覆盖训练模型。

## 3. 查看与反馈

~~~bat
type output\a5000_getup_prior_check_400k_r01\summary.txt
type output\a5000_getup_prior_check_400k_r01\run_time.txt
start "" "output\a5000_getup_prior_check_400k_r01\score_distributions.png"
start "" "output\a5000_getup_prior_check_400k_r01\standing_candidates.png"
explorer "output\a5000_getup_prior_check_400k_r01"
~~~

反馈 summary.txt 全文、run_time.txt 和两张 PNG 即可，不上传模型。
如果需要进一步分析，完整数值和配置在同目录 report.json：

~~~bat
type output\a5000_getup_prior_check_400k_r01\report.json
~~~

如果报错或没有 summary：

~~~bat
powershell -NoProfile -Command "Get-Content 'output/a5000_getup_prior_check_400k_r01/console.txt' -Tail 100"
~~~

## 4. 可选 50k 对照

先运行并反馈 400k 检查即可。如需要调查过拟合或站姿冲突，可用相同参数另跑旧 50k。
两个模型各自使用自己的归一化器与参考标尺；不直接用跨模型 raw SDS 的绝对大小选模。
注意原 50k checkpoint 是另一轮初始化，不是 400k 训练的中间 checkpoint。

~~~bat
python -u tools/diffusion_model/check_getup_prior.py --cfg_path output/a5000_getup_prior_joint_50k_r01/diffusion_config.yaml --model_file output/a5000_getup_prior_joint_50k_r01/model.pt --out_dir output/a5000_getup_prior_check_50k_r01 --device cuda:0 --num_samples 256 --batch_size 256 --noise_repeats 8 --seed 123 --diffusion_steps 22 15 8
~~~

## 本地验证边界

5 项契约检查覆盖：训练特征完全一致、选定姿态的坐标变换不变性、静态候选零速度、
扰动后的骨架重算及输入不变、噪声配对与重复性。
本地还用仅训练 2 步的短测 checkpoint 跑通完整默认检查与图表/JSON 导出。
这些是程序验证；不能据此判断远端 400k 权重质量。本地没有该完整权重。
