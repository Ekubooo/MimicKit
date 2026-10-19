# A5000 联合起身 prior：论文区间下限 400k，从头训练

2026-10-20 状态：本轮已完成。log13 记录 2026-10-19 22:43:44 至 2026-10-20 01:13:56，2h30m12s，
EXIT_CODE=0，最终 Loss 0.0228136；用户目测大多数样本正常。
下一步评分/站姿检查见 GETUP_PRIOR_CHECK_WINDOWS.md。

本轮只进行 diffusion；收到结果后再继续 Getup RL。
配置为 tools/diffusion_model/config/tinymdm_getup_400k.yaml。
以已验证 50k 起身配置为基线，仅将更新预算扩展至 400,000：
batch 512、lr 1e-4、归一化统计窗口 10,000。针对两段短动作保留较小学习率。
旧 50k 配置与结果保留，模型和归一化统计重新初始化。

512 是每次更新采样的动作窗口数，每个窗口含 10 帧（30 Hz）；
不是并行仿真环境数。累计 204,800,000 次窗口采样，仍来自原来两段动作。
不运行 Newton，也不需要 RL 的 num_envs、max_samples 或 engine_config 参数。

论文 v3 第 6.2 节给出 400k–800k 次更新，本轮取区间下限 400k。
架构沿用两层、四头、10 帧、50 噪声等级与 EMA；具体 batch/LR 取公开代码小数据默认。
这对齐了论文的架构和迭代范围，但不能称为 Getup 数据、优化器及全部超参数的精确复现。
论文：https://arxiv.org/html/2512.03028v3#S6.SS2
用户最新要求优先对齐论文，本轮选择 400k，原先准备的 200k 不再作为主方案。除更新次数外，数据、骨架、特征与具体 Getup 设定
也影响是否对齐论文，增加更新次数本身不等于完整复现。

A5000 上一轮 50k 已实测 18 分钟。相同网络与 batch 下线性估算为 144 分钟，
本轮预留约 140–180 分钟；保存/导出、系统负载等可能带来差异。

## 1. 同步仓库

开发电脑 PowerShell：

~~~powershell
Set-Location D:\ComputerScience\PROJECT\MimicKit
git add tools/diffusion_model/config/tinymdm_getup_400k.yaml docs/GETUP_PRIOR_400K_WINDOWS.md docs/GETUP_SHORT_MOTION_ANALYSIS.md docs/GETUP_PRIOR_WINDOWS.md plan.md
git diff --cached --stat
git commit -m "Add 400k joint getup prior experiment"
git push origin main
git log -1 --oneline
~~~

训练电脑 Anaconda Prompt：

~~~bat
conda activate mimickit-newton
cd /d C:\Project\MimicKit
git status --short
git switch main
git pull --ff-only origin main
git log -1 --oneline
set MPLBACKEND=Agg
set PYTHONUTF8=1
~~~

两个机器的最新提交应一致。如工作区有改动或拉取失败，先处理或反馈，不强制覆盖。

## 2. 检查与存档

~~~bat
python -c "from pathlib import Path; import yaml,torch,diffusers,matplotlib,PIL; c=yaml.safe_load(Path('tools/diffusion_model/config/tinymdm_getup_400k.yaml').read_text(encoding='utf-8')); assert (c['num_iterations'],c['batch_size'],c['lr'],c['num_samples_stat'])==(400000,512,0.0001,10000); paths=[c['env_config'],c['motion_file'],'data/assets/humanoid/humanoid.xml','data/motions/humanoid/humanoid_getup_faceup.pkl','data/motions/humanoid/humanoid_getup_facedown.pkl']; missing=[p for p in paths if not Path(p).is_file()]; assert not missing,missing; assert torch.cuda.is_available(),'CUDA unavailable'; assert not Path('output/a5000_getup_prior_joint_400k_scratch_r01/model.pt').exists(),'Output already has a model; use a new run directory'; print('GPU:',torch.cuda.get_device_name(0)); print('Iterations:',c['num_iterations'],'Batch:',c['batch_size'],'LR:',c['lr']); print('Checks OK')"
if not exist output\a5000_getup_prior_joint_400k_scratch_r01 mkdir output\a5000_getup_prior_joint_400k_scratch_r01
copy /Y data\datasets\dataset_humanoid_getup.yaml output\a5000_getup_prior_joint_400k_scratch_r01\dataset.yaml
copy /Y tools\diffusion_model\config\tinymdm_getup_400k.yaml output\a5000_getup_prior_joint_400k_scratch_r01\source_config.yaml
python -m pip freeze > output\a5000_getup_prior_joint_400k_scratch_r01\dependencies.txt
git rev-parse HEAD > output\a5000_getup_prior_joint_400k_scratch_r01\source_commit.txt
git diff > output\a5000_getup_prior_joint_400k_scratch_r01\source_diff.patch
~~~

检查应显示 A5000、Iterations: 400000 Batch: 512 LR: 0.0001、Checks OK。
任何检查失败都先处理，不启动。再次独立实验时统一更换 r01 为 r02 等新目录。

## 3. 从头训练

下面整行在激活 Conda 的 Anaconda Prompt 中粘贴。
没有 model_file，不读取旧权重；400k 是完整新预算，不是在 50k 上补 350k。
没有改变原版训练入口的随机种子逻辑，因此两次初始化随机性不同；独立 test 入口固定采样种子，可用于同种子样本对照。

~~~bat
powershell -NoProfile -Command "$taskOutDir='output/a5000_getup_prior_joint_400k_scratch_r01'; if (Test-Path ($taskOutDir+'/model.pt')) { throw 'Output already has a model; use a new run directory' }; ('START '+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) | Set-Content -Encoding UTF8 ($taskOutDir+'/run_time.txt'); python -u tools/diffusion_model/train_tinymdm.py --mode train --cfg_path tools/diffusion_model/config/tinymdm_getup_400k.yaml --out_dir $taskOutDir --device cuda:0 2>&1 | Tee-Object -FilePath ($taskOutDir+'/console.txt'); $taskExitCode=$LASTEXITCODE; ('END '+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+' EXIT_CODE='+$taskExitCode) | Add-Content -Encoding UTF8 ($taskOutDir+'/run_time.txt'); exit $taskExitCode"
~~~

约每 2000 次更新输出一次，保存前导出 16 个 GIF。
最终 Iteration 为 399999，run_time.txt 应包含 END 和 EXIT_CODE=0。
中间 model.pt、samples 被覆盖保存，未启用历史 checkpoint 留存。

## 4. 进度与查看

另开 Prompt：

~~~bat
cd /d C:\Project\MimicKit
powershell -NoProfile -Command "Get-Content 'output/a5000_getup_prior_joint_400k_scratch_r01/console.txt' -Tail 35"
~~~

可选 GPU 监视（单独窗口，Ctrl+C 结束监视）：

~~~bat
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5
~~~

完成后：

~~~bat
type output\a5000_getup_prior_joint_400k_scratch_r01\log.txt
type output\a5000_getup_prior_joint_400k_scratch_r01\run_time.txt
dir /T:W output\a5000_getup_prior_joint_400k_scratch_r01\model.pt
explorer "output\a5000_getup_prior_joint_400k_scratch_r01\samples"
start "" "output\a5000_getup_prior_joint_400k_scratch_r01\samples\anim_000.gif"
~~~

## 5. 最终模型独立采样

~~~bat
conda activate mimickit-newton
cd /d C:\Project\MimicKit
set MPLBACKEND=Agg
set PYTHONUTF8=1
if not exist output\a5000_getup_prior_joint_400k_scratch_r01_samples mkdir output\a5000_getup_prior_joint_400k_scratch_r01_samples
python -u tools/diffusion_model/train_tinymdm.py --mode test --cfg_path output/a5000_getup_prior_joint_400k_scratch_r01/diffusion_config.yaml --model_file output/a5000_getup_prior_joint_400k_scratch_r01/model.pt --out_dir output/a5000_getup_prior_joint_400k_scratch_r01_samples --device cuda:0 > output\a5000_getup_prior_joint_400k_scratch_r01_samples\sampling_console.txt 2>&1
echo Sampling exit code: %ERRORLEVEL%
type output\a5000_getup_prior_joint_400k_scratch_r01_samples\sampling_console.txt
explorer "output\a5000_getup_prior_joint_400k_scratch_r01_samples\samples"
~~~

重定向期间没有实时输出，等待返回。生成的仍是 10 帧局部片段；增加训练预算不改变片段长度。

## 6. 反馈

提供以下命令的输出、独立采样日志和多个样本的观察即可，无需模型或 GPU CSV。

~~~bat
type output\a5000_getup_prior_joint_400k_scratch_r01\log.txt
type output\a5000_getup_prior_joint_400k_scratch_r01\run_time.txt
dir /T:W output\a5000_getup_prior_joint_400k_scratch_r01\model.pt
type output\a5000_getup_prior_joint_400k_scratch_r01\diffusion_config.yaml
type output\a5000_getup_prior_joint_400k_scratch_r01\env_config.yaml
type output\a5000_getup_prior_joint_400k_scratch_r01\dataset.yaml
type output\a5000_getup_prior_joint_400k_scratch_r01_samples\sampling_console.txt
~~~

如报错：

~~~bat
powershell -NoProfile -Command "Get-Content 'output/a5000_getup_prior_joint_400k_scratch_r01/console.txt' -Tail 100"
~~~

收到结果后，再实现并验证 Getup RL 环境、奖励及初始化，不自动启动 RL。
RL 暂定 1B samples、2048 环境；完整短动作适配分析和时间估算见 GETUP_SHORT_MOTION_ANALYSIS.md。
