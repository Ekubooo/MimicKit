# UU CLI 远程开发与训练能力

整理日期：2026-11-05。依据此前对实验室设备的实际连接测试整理；本文编写时未重新连接远端。

本文件记录远程通道、已验证能力、限制及建议工作方式。训练准备、验收与反馈要求继续参照 [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)。能力说明不等于已启动训练，也不自动授权提交、推送、安装软件或修改远端。

## 1. 本机入口与远端设备

通过本机 PowerShell 调用原生 UU CLI，无需安装 MCP：

```powershell
& 'C:\SoftWare\UU netdesk\GameViewer\bin\uuyc-cli.exe' device ls
& 'C:\SoftWare\UU netdesk\GameViewer\bin\uuyc-cli.exe' device info aeawvpf46abdkdoi
& 'C:\SoftWare\UU netdesk\GameViewer\bin\uuyc-cli.exe' term -d aeawvpf46abdkdoi new
```

历史设备查询结果如下；在线状态必须在使用时重新查询：

| 设备 | DEVICE_ID | 测试时状态 |
| --- | --- | --- |
| DESKTOP-EQ9L7HF | `aeawvpf46abdkdoi` | 在线，已实际连接 |
| DESKTOP-HEP2T6V | `aeawrwt5qqal533j` | 离线，未测试 |

用户将目标设备描述为实验室 Win11；CLI 设备信息只显示 Windows，系统版本尚未另行核实。已连接设备的 `hostname` 返回 `DESKTOP-EQ9L7HF`，远端默认 PowerShell 工作目录为 `C:\Users\ClassRoom`。

本机受限沙箱内运行 CLI 曾返回 `Error: Main application is not running`，沙箱外运行则成功。这说明该报错可能来自本地通信隔离，不能仅据此断定 UU 主程序没有运行。本次成功路径使用了获准的沙箱外命令执行。

## 2. 已实际验证的能力

| 项目 | 结果 |
| --- | --- |
| 创建远程终端 | 成功进入远端 Windows PowerShell |
| 保持交互 | 同一个本地执行会话中，可连续发送命令并读取返回 |
| 主机名查询 | `hostname` 返回 `DESKTOP-EQ9L7HF` |
| GPU 查询 | 成功执行 `nvidia-smi`，分段读回完整的 36 行输出，含 17 条进程记录 |
| Conda 环境激活 | 在远端启动 CMD 并激活已有 `WarpLab` 环境 |
| Python 验证 | 环境内 Python 可运行，版本为 3.11.16 |
| 正常退出 | `exit` 可结束远端 Shell；CLI 随后正常退出 |

GPU 历史快照如下，负载、温度、显存占用等是采样时状态，不代表当前状态：

| 项目 | 测试结果 |
| --- | --- |
| GPU | NVIDIA RTX A5000 |
| 驱动版本 | 596.71 |
| `nvidia-smi` 显示的 CUDA Version | 13.2；不等同于已验证安装了对应 CUDA Toolkit 或框架运行时 |
| 显存使用 / 总量 | 800 / 24564 MiB |
| GPU 利用率 | 1% |
| 温度 / 风扇 | 42°C / 30% |
| 功耗 / 上限 | 69 / 230 W |
| 性能状态 / 驱动模式 | P0 / WDDM |

## 3. 通过远程终端使用 Conda Prompt

远端默认 PowerShell 中，`Get-Command conda` 未找到命令；但读取用户的 `.conda\environments.txt` 发现了以下登记路径：

```text
C:\Users\ClassRoom\miniconda3
C:\Users\ClassRoom\miniconda3\envs\WarpLab
C:\Users\ClassRoom\miniconda3\envs\mimickit-newton
```

登记路径不等于所有环境均已验证可用。本次仅实际激活并运行了 `WarpLab`，未测试 `mimickit-newton` 的依赖或训练入口。

在远端 PowerShell 中执行的激活命令：

```powershell
cmd /d /k C:\Users\ClassRoom\miniconda3\condabin\conda.bat activate WarpLab
```

随后在这个 CMD 子进程中执行：

```bat
echo %CONDA_DEFAULT_ENV%
python -B -c "import sys; print(sys.executable); print(sys.version)"
```

实测提示符为 `(WarpLab) C:\Users\ClassRoom>`，环境名为 `WarpLab`，解释器路径为：

```text
C:\Users\ClassRoom\miniconda3\envs\WarpLab\python.exe
```

这种方式具备 Conda Prompt 的环境激活功能，不需要打开图形窗口或执行 `conda init`。测试只改变当前进程的环境状态，没有执行安装、更新、配置编辑或主动写文件操作。

正式使用 `mimickit-newton` 时，可以采用同样的激活方式，但应先核实环境及项目依赖，不能把 `WarpLab` 的成功结果视为训练环境已就绪。

## 4. 会话持久性与身份验证

需要区分聊天、本地 CLI 连接、远端终端会话和训练进程：

- 此前 GPU 与 Conda 两轮成功测试均主动执行了 `exit`，对应测试终端已结束，并未持续保持连接。
- 从嵌套 CMD 退出一次只会返回外层 PowerShell；再退出外层 PowerShell才结束本次远端终端。
- UU 自身提示，关闭连接窗口可以断开连接并让远端会话保持后台活跃；CLI 提供 `ls`、`attach`、`kill` 等会话命令。
- 这表明 UU 支持保留会话及重新附加的机制，但本次没有实测断线重连、主程序退出、机器重启或长时间运行下的行为。
- 当前命令执行工具已实测可通过 PTY 会话持续收发输入输出；不能据此承诺本地工具会话跨应用重启或无限期有效。

以下为 CLI 帮助中提供的会话操作形式；命名会话与重新附加尚未实测：

```powershell
& 'C:\SoftWare\UU netdesk\GameViewer\bin\uuyc-cli.exe' term -d aeawvpf46abdkdoi new mimickit-work
& 'C:\SoftWare\UU netdesk\GameViewer\bin\uuyc-cli.exe' term -d aeawvpf46abdkdoi ls --json
& 'C:\SoftWare\UU netdesk\GameViewer\bin\uuyc-cli.exe' term -d aeawvpf46abdkdoi attach mimickit-work
```

首次连接曾因远端锁屏进入账户密码提示。此时输入 `hostname` 等文本会被当作密码，不能继续当作 Shell 操作。用户解锁远端后重新连接成功。需要验证身份时，由用户在本机 UU 客户端处理，不在聊天或仓库文档中提供账户密码。

## 5. 长输出读取限制

此终端返回的是带 ANSI 控制序列的屏幕更新，并非始终完整保留的逐行标准输出。直接执行 `nvidia-smi` 时，快速滚屏及重绘使首次读取只保留了尾部，不能据此宣称已读完。

已验证的解决方式是在远端 PowerShell 内存中暂存输出，再分段读取：

```powershell
$gpuOutput = @(nvidia-smi)
$gpuOutput.Count
cls; $gpuOutput[0..5]
cls; $gpuOutput[6..11]
cls; $gpuOutput[12..17]
cls; $gpuOutput[18..23]
cls; $gpuOutput[24..29]
cls; $gpuOutput[30..35]
```

上述范围对应那次实际取得的 36 行输出；以后需按实际行数调整。每次分段命令应单独发送并等待读回，不要一次快速执行所有分页，否则仍可能滚屏丢失。`cls` 只清当前终端显示。

训练日志应作为独立文件保存，并按需读取尾部、错误段、指标摘要或指定范围。不要把持续刷新的控制台当作唯一日志，也不要把最后几行当作完整训练证据。大日志应有限量地读取，不宜全部装入变量或一次打印。

## 6. 本地开发、远端同步与训练的建议流程

以下是可采用的工作方式，尚未通过该连接实际执行 Git 同步或 MimicKit 训练：

1. 本地开发代码和配置，完成与改动相称的检查，整理可审阅的差异。
2. 按现有协作约定，由用户提交并推送到远端可访问的 Git 仓库；新的明确指令可调整具体分工。
3. 远端先核实项目路径、分支、工作区改动与提交状态，再按当轮指令同步代码，确认两端目标提交一致。不覆盖远端未提交改动。
4. 核实训练环境、依赖、数据、资产、prior 与 checkpoint。Git 不会自动同步被忽略的大文件、模型和训练输出。
5. 明确本轮配置、预算、输出目录与验收条件后，再按授权启动训练。记录实际提交、启动参数、起止时间和退出码。
6. 训练写入独立日志；通过 UU CLI 查看日志、进程与 GPU 状态，分析结果，再回到本地迭代。

现有流程文档中的训练目录参考值是 `C:\Project\MimicKit`；本次远程能力测试没有核实该目录，实际操作前仍须检查。

长训练应使用经验证能脱离交互连接运行、持续写日志的启动方式，或先验证 UU 保留会话的可靠性。未验证前，不应假定关闭连接一定不会影响训练。监控最好使用另一个终端，避免向正在前台训练的进程发送 Shell 命令或误发中断信号。

## 7. 两端 Git SSH 口令

用户说明本地和远端的 Git SSH 均有密码；尚未核实它们具体是私钥口令还是其他认证提示。

如果是 SSH 私钥口令，建议由用户在两端各自通过 `ssh-agent` 解锁对应密钥。在密钥已加载、Git 使用同一个 SSH 实现且进程可访问该 agent 的条件下，推送和拉取通常无需反复输入口令。实际是否可用需要分别验证。

本地解锁不会自动替远端解锁，UU 身份验证也不替代 Git SSH 身份验证。Agent 生命周期、登录会话和 Git 所用 SSH 实现会影响解锁状态，不能默认它永久有效。若提示的是服务器账户密码，则需按实际认证方式处理，不能用私钥解锁方案代替。

不把私钥、口令或令牌写入仓库、命令脚本、日志或聊天。需要交互输入口令时由用户在对应机器完成。

## 8. 预计能力与尚未验证的边界

| 用途 | 预计可做的事 | 当前证据或限制 |
| --- | --- | --- |
| 远程诊断 | 查询资源、进程、环境，读取日志 | 主机名、GPU 和 Conda 已实测；其他按需验证 |
| 开发维护 | 使用远端 Git、运行测试和编译工具 | 取决于工具、权限与认证；尚未实测 Git 同步 |
| 模型实验 | 在指定环境运行训练、推理、评估 | Python 可运行；MimicKit 训练与依赖尚未验证 |
| 日志分析 | 分段读取文本、提取指标、定位错误 | 已验证长文本分段读取，需知道实际日志路径与格式 |
| 长任务管理 | 断开后重新附加或采用独立进程运行 | 尚未验证断线存活和恢复机制 |
| 图形与产物查看 | 命令行可尝试调用相关程序或导出流程 | 该通道本身不提供图形桌面观察；图片、视频及文件传输需另行安排 |

实际能力受远端账户权限、已安装工具、网络和终端行为约束。阅读日志不等于验收动作质量，进程存活也不等于训练正确。

本文件不是后台监控任务。用户要求查看时可以重连读取；若明确要求跟踪某次训练，可在当前任务中持续检查，不能将聊天结束解释为仍在自动监控。
