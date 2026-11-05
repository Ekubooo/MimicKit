# UU CLI 远程开发与训练

更新日期：2026-11-05。训练准备与验收遵循 [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)。

## 1. 连接实验室设备

本机通过原生 UU CLI 连接，无需安装 MCP：

```powershell
& 'C:\SoftWare\UU netdesk\GameViewer\bin\uuyc-cli.exe' device ls
& 'C:\SoftWare\UU netdesk\GameViewer\bin\uuyc-cli.exe' term -d aeawvpf46abdkdoi new
```

目标主机：`DESKTOP-EQ9L7HF`，设备 ID：`aeawvpf46abdkdoi`，GPU：NVIDIA RTX A5000（约 24 GB 显存）。连接后默认进入远端 PowerShell。

已验证连续发送命令、读取主机名和完整 GPU 输出，以及通过 CMD 激活 Conda、运行 Python。
2026-11-05已验证远端 `git pull --ff-only origin main`、真实G1模型/先验加载及192回合控制测试完整完成。
长训练尚未通过此通道完成；不把测试通过等同于新策略训练成功。

连接注意事项：

- 沙箱内曾误报 `Main application is not running`，获准在沙箱外调用 CLI 后成功；遇到该报错需检查本地通信隔离。
- 远端锁屏时可能要求账户密码，由用户在 UU 客户端完成验证或解锁后重试。

## 2. 正式训练环境

MimicKit 正式训练使用 **`mimickit-newton`**。在远端 PowerShell 中启动已激活环境的 CMD：

```powershell
cmd /d /k C:\Users\ClassRoom\miniconda3\condabin\conda.bat activate mimickit-newton
```

随后在 CMD 中核对环境和解释器：

```bat
echo %CONDA_DEFAULT_ENV%
python -B -c "import sys; print(sys.executable)"
```

已核实解释器为 `C:\Users\ClassRoom\miniconda3\envs\mimickit-newton\python.exe`。
2026-11-05实际版本：Newton 1.0.0、Warp 1.12.0、MuJoCo 3.5.0、MuJoCo-Warp 3.5.0.2、Torch 2.10.0+cu128。
无需为此执行 `conda init`。

项目目录已核实为 `C:\Project\MimicKit`。

## 3. 本地开发、远端训练

1. 本地修改代码与配置，完成必要检查。
2. 按现有约定由用户提交、推送；远端检查工作区后拉取，核对两端目标提交一致。
3. 确认远端数据、资产、prior 和 checkpoint 齐全；Git 不会同步被忽略的文件。
4. 在 `mimickit-newton` 中按当轮确定的配置和预算启动训练，记录提交、参数、输出目录、起止时间和退出码。
5. 训练保存独立日志，通过 UU CLI 读取日志、检查进程和 GPU，分析后继续本地迭代。监控使用另一个终端。

两端 Git SSH 认证独立于 UU。若密码是私钥口令，由用户分别在两端通过 `ssh-agent` 解锁，并确认 Git 能访问对应 agent；解锁失效后需重新处理。口令不写入聊天、脚本或仓库。

## 4. 会话与日志限制

- **会话退出：**此前测试均已用 `exit` 结束。嵌套 CMD 中第一次 `exit` 返回 PowerShell，再次 `exit` 才结束远端终端。
- **连接保留：**用户要求无必要时保留UU会话，避免频繁退出/重连。UU提供 `term -d DEVICE_ID ls --json` 和 `term -d DEVICE_ID attach NAME`。
- **独立进程：**2026-11-05实测：在UU内直接 `Start-Process` 启动的Python测试会随远端终端 `exit` 被结束，不能据此宣称后台独立运行。使用 `Invoke-CimMethod -ClassName Win32_Process -MethodName Create` 经WMI启动，`Win32_ProcessStartup.ShowWindow=0` 隐藏窗口；已通过退出UU后延迟写入完成标记的测试，且以此方式完成了G1六组控制评测。命令明确指定工作目录、解释器、输出目录及stdout/stderr文件；不需要创建计划任务。即使使用独立启动方式，仍按用户要求保留现有UU会话。
- **长输出：**终端以屏幕重绘方式返回内容，快速滚屏可能遗漏前部。短输出可在 PowerShell 中用 `$gpuOutput = @(nvidia-smi)` 暂存，再按实际行数分段打印；每段读回后再继续。训练日志按需读取尾部、错误段或指标摘要，避免一次打印整个文件。
- **观察范围：**此通道可操作命令行，不直接提供图形桌面观察；动作质量仍需评测和图像、视频等证据。
- **监控范围：**可按要求重连查看或跟踪指定训练；聊天结束不代表仍在后台自动监控。
