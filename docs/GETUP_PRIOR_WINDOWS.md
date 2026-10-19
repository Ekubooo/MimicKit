# Windows / A5000 联合起身 prior 首轮

目标：仰卧、俯卧等概率联合 prior。暂不进行策略切换。
本步骤只训练离线 diffusion，不运行 Newton，也不训练物理控制策略。

## 配置与预算

- 数据：data/datasets/dataset_humanoid_getup.yaml，两段动作不硬拼接。
- 特征：data/envs/smp_getup_prior_env.yaml，仅供离线 prior，不是 Getup RL 环境。
- 训练：tools/diffusion_model/config/tinymdm_getup.yaml。
- 50,000 次更新，batch 512，lr 1e-4；网络和优化参数沿用单动作 prior。
- 30 Hz、10 帧，每帧 114 维，输入 1140 维，可训练参数约 2.72M。
- 50k 是首轮预算，不代表保证收敛或达到论文规模。
- A5000 尚无本任务实测；先预留 1–4 小时，再根据日志修正。

## 远端准备

全部在 Anaconda Prompt / CMD 中逐行执行。无需 PowerShell 注册 Conda。
若尚未同步配置，以下三条 Python 命令直接生成等价内容，会覆盖三个专用配置。

~~~bat
conda activate mimickit-newton
cd /d C:\Project\MimicKit
set MPLBACKEND=Agg
set PYTHONUTF8=1
python -c "import torch,diffusers,matplotlib,PIL,yaml; from pathlib import Path; paths=['data/assets/humanoid/humanoid.xml','data/motions/humanoid/humanoid_getup_faceup.pkl','data/motions/humanoid/humanoid_getup_facedown.pkl']; missing=[p for p in paths if not Path(p).is_file()]; assert not missing,missing; assert torch.cuda.is_available(),'CUDA unavailable'; print('GPU:',torch.cuda.get_device_name(0)); print('torch:',torch.__version__,'diffusers:',diffusers.__version__); print('Assets OK')"
python -c "from pathlib import Path; import yaml; d={'motions':[{'file':'data/motions/humanoid/humanoid_getup_'+name+'.pkl','weight':1.0} for name in ('faceup','facedown')]}; Path('data/datasets/dataset_humanoid_getup.yaml').write_text(yaml.safe_dump(d,sort_keys=False),encoding='utf-8')"
python -c "from pathlib import Path; import yaml; b=yaml.safe_load(Path('data/envs/smp_humanoid_env.yaml').read_text(encoding='utf-8')); keys=('char_file','global_obs','root_height_obs','enable_tar_obs','disc_dof_vel_obs','num_disc_obs_steps','key_bodies'); e={k:b[k] for k in keys}; e['global_obs']=False; Path('data/envs/smp_getup_prior_env.yaml').write_text(yaml.safe_dump(e,sort_keys=False),encoding='utf-8')"
python -c "from pathlib import Path; import yaml; c=yaml.safe_load(Path('tools/diffusion_model/config/tinymdm_single_clip.yaml').read_text(encoding='utf-8')); c.update(env_config='data/envs/smp_getup_prior_env.yaml',motion_file='data/datasets/dataset_humanoid_getup.yaml'); Path('tools/diffusion_model/config/tinymdm_getup.yaml').write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8'); print('Iterations:',c['num_iterations'],'Batch:',c['batch_size'],'LR:',c['lr'])"
~~~

检查应显示 A5000、Assets OK、Iterations: 50000、Batch: 512、LR: 0.0001。
若报错，先处理错误，不启动训练；不需要升级依赖。

## 启动

首轮用 r01。后续独立实验换新目录，避免覆盖。

~~~bat
if not exist output\a5000_getup_prior_joint_50k_r01 mkdir output\a5000_getup_prior_joint_50k_r01
copy /Y data\datasets\dataset_humanoid_getup.yaml output\a5000_getup_prior_joint_50k_r01\dataset.yaml
python -m pip freeze > output\a5000_getup_prior_joint_50k_r01\dependencies.txt
git rev-parse HEAD > output\a5000_getup_prior_joint_50k_r01\source_commit.txt
powershell -NoProfile -Command "$taskOutDir='output/a5000_getup_prior_joint_50k_r01'; ('START '+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) | Set-Content -Encoding UTF8 ($taskOutDir+'/run_time.txt'); python -u tools/diffusion_model/train_tinymdm.py --mode train --cfg_path tools/diffusion_model/config/tinymdm_getup.yaml --out_dir $taskOutDir --device cuda:0 2>&1 | Tee-Object -FilePath ($taskOutDir+'/console.txt'); $taskExitCode=$LASTEXITCODE; ('END '+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+' EXIT_CODE='+$taskExitCode) | Add-Content -Encoding UTF8 ($taskOutDir+'/run_time.txt'); exit $taskExitCode"
~~~

最后一行整行粘贴，结束时自动记录时间和退出码。不要添加 RL 的 num_envs/max_samples/engine_config 参数。
约每 2000 更新及最后记录日志，记录前还会导出 16 段动画，两次输出之间不刷屏正常。
最终 Iteration 为 49999；成功结束还应有 END 和 EXIT_CODE=0。
model.pt 和 samples 在各保存点覆盖更新，不保留全部历史 checkpoint。
EMA 预热阈值为 5000，早期动画不能代表最终质量。

## 查看和独立采样

训练中可另开 Prompt 查看进度：

~~~bat
cd /d C:\Project\MimicKit
powershell -NoProfile -Command "Get-Content 'output/a5000_getup_prior_joint_50k_r01/console.txt' -Tail 35"
~~~

完成后：

~~~bat
type output\a5000_getup_prior_joint_50k_r01\log.txt
type output\a5000_getup_prior_joint_50k_r01\run_time.txt
dir /T:W output\a5000_getup_prior_joint_50k_r01\model.pt
start "" "output\a5000_getup_prior_joint_50k_r01\samples\anim_000.gif"
explorer "output\a5000_getup_prior_joint_50k_r01\samples"
~~~

独立重新加载模型并生成 16 段样本：

~~~bat
conda activate mimickit-newton
cd /d C:\Project\MimicKit
set MPLBACKEND=Agg
set PYTHONUTF8=1
if not exist output\a5000_getup_prior_joint_50k_r01_samples mkdir output\a5000_getup_prior_joint_50k_r01_samples
python -u tools/diffusion_model/train_tinymdm.py --mode test --cfg_path output/a5000_getup_prior_joint_50k_r01/diffusion_config.yaml --model_file output/a5000_getup_prior_joint_50k_r01/model.pt --out_dir output/a5000_getup_prior_joint_50k_r01_samples --device cuda:0 > output\a5000_getup_prior_joint_50k_r01_samples\sampling_console.txt 2>&1
echo Sampling exit code: %ERRORLEVEL%
type output\a5000_getup_prior_joint_50k_r01_samples\sampling_console.txt
start "" "output\a5000_getup_prior_joint_50k_r01_samples\samples\anim_000.gif"
~~~

采样因重定向不实时显示输出，等待返回。GIF 是慢放的 10 帧局部片段，不是完整起身或 RL 效果。
检查多个样本；loss 不能单独证明模型质量或后续起身成功。

## 反馈

无需上传模型或长 GPU CSV。提供：

1. 训练目录 log.txt 全文、run_time.txt、上面的 model.pt 文件时间/大小。
2. 同目录 diffusion_config.yaml、env_config.yaml、dataset.yaml。
3. 多个 GIF 的观察或代表性 GIF；独立采样的 sampling_console.txt。
4. 如报错，console.txt 末尾约 100 行；依赖问题再提供 dependencies.txt。

~~~bat
type output\a5000_getup_prior_joint_50k_r01\diffusion_config.yaml
type output\a5000_getup_prior_joint_50k_r01\env_config.yaml
type output\a5000_getup_prior_joint_50k_r01\dataset.yaml
powershell -NoProfile -Command "Get-Content 'output/a5000_getup_prior_joint_50k_r01/console.txt' -Tail 100"
~~~

GPU 监视可选，另开窗口，Ctrl+C 只结束监视：

~~~bat
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv -l 5
~~~

## 本地流程检查

2026-10-19，RTX 5070：原网络、batch=4、统计样本=32、仅 2 次更新；
通过动作读取、前反向、EMA、保存、重新加载、motion/GIF 导出。
该结果不代表收敛、A5000 性能或起身策略效果。
