# 退出资源清理检查

检查日期：2026-10-12。修复目标：有限测试或训练完成、窗口关闭、Ctrl+C、初始化或运行失败时，主动释放本次运行拥有的资源。

## 原因与修复

用户提供的 `ExitError.txt` 中，10 个测试 episode 已完成，错误随后发生于 pyglet 的 Win32 `on_deactivate` 回调。Newton Viewer 的对象引用环进入垃圾回收时，事件处理器的弱引用可能先失效；窗口析构触发失焦事件，就会出现 `assert handler is not None`。

现在 `run()` 在 `finally` 中关闭环境、agent 的日志资源和本次启动的虚拟显示。环境将关闭操作传递给引擎，Newton 在事件处理器仍然存活时主动关闭可视 Viewer 和隐藏录像 Viewer。已关闭的 OpenGL 上下文不再重复关闭。已有的窗口状态检查和 Ctrl+C 退出码行为继续保留。

关闭窗口表示立即停止运行，不输出未完成测试的汇总，也不增加自动保存 checkpoint 的行为。正常训练原有的 checkpoint 输出保持原样。资源构造失败时也尝试清理部分初始化对象；多个清理操作会全部尝试，保留原来的运行异常。

## 同类问题检查

检查了 `mimickit`、`tools` 和 `tests/Scene_Test` 中的 Python 资源创建及对应释放路径。

| 资源 | 处理结果 | 验证方式 |
| --- | --- | --- |
| Newton 可视窗口、隐藏录像窗口 | 增加显式关闭；覆盖正常结束、异常、部分初始化和重复关闭 | 本机 GPU、Win32 窗口与实际失焦事件；单元测试 |
| 12 类环境的构造与共用退出路径 | 构造失败清理，环境关闭传递给引擎 | 工厂代码检查、边界单元测试、原生初始化失败测试 |
| Isaac Gym viewer/sim | 先关闭录像资源、viewer，再销毁 sim | 实际方法的模拟资源测试；本机未安装该后端 |
| Isaac Gym 相机图像访问 | 用 `finally` 配对 `start/end_access_image_tensors` | 读取失败测试；本机未安装该后端 |
| Isaac Lab 键盘订阅、应用、录像 annotator/render product | 取消订阅、detach、destroy、关闭应用；补上窗口退出检查 | 官方 API 对照、实际方法的模拟资源测试；本机未安装该后端 |
| 文本、TensorBoard、W&B 日志 | 显式关闭文件、writer 和本 logger 拥有的 run；保留退出兜底 | 真实文件/writer 测试、SMP 小训练；W&B 使用 offline 模式 |
| MoviePy 编码器 | 保存完成或失败后关闭 clip | 实际录像保存及编码失败测试 |
| Matplotlib 动画与 SDF 切片 | 在异常路径也关闭本函数创建的 figure | 初始化失败、真实保存失败、原有场景测试 |
| Linux Xvfb | 保留自建子进程句柄并终止/回收；不关闭外部 DISPLAY | 模拟进程测试；本机为 Windows，未实跑 Xvfb |
| multiprocessing worker、通信组 | 正常结束时销毁通信组并 join；窗口关闭/中断时终止并回收 worker | 真实 worker 回收、模拟通信组销毁测试 |
| 常规配置、motion、mesh 文件读写 | 现有路径使用上下文管理，未发现同类析构回调问题 | 源码检查 |
| 独立 prior 训练脚本日志、独立 log 绘图 | Logger 的退出兜底覆盖 prior 的文件；独立 log 绘图使用 Matplotlib 本身的进程退出清理 | 源码检查；未改 prior 训练循环 |

## 验证结果

环境为现有 `mimickit-newton` Conda 环境：Python 3.11.16、PyTorch 2.10.0+cu128、Newton 1.0.0、Warp 1.12.0、pyglet 2.1.16，RTX 5070。未更改依赖、策略权重或正式配置文件。仿真验证默认使用一个环境，双环境检查使用两个。

- 44 项退出/资源生命周期测试通过，7 项原有场景测试通过。
- 17 项原生验证通过，另有 SMP location 正常结束验证通过：包含 SMP steering/location/spinkick/dodgeball、有限 episode、双环境、隐藏录像、TensorBoard/W&B 小训练、模型加载失败、窗口创建后初始化失败、渲染前/事件处理中/暂停时关闭窗口、真实 Ctrl+C 和真实 worker 回收。
- 直接运行与本地 `launch.py` 的真实 Windows Ctrl+C 均返回 130，无中断 traceback；launcher 的子进程已退出。正常测试、正常完成和窗口关闭返回 0。
- 正常完成验证通过实际 Win32 `WM_KILLFOCUS` 回调，并在资源关闭后触发垃圾回收，未出现原报错或其他未处理析构异常。
- 对比 HEAD 的 1,315 个已有 Python 函数。训练更新、奖励、rollout、仿真步进、solver 和 kernel 的函数体保持不变；已有函数的变更限于资源构造、清理、日志与图形边界。

本地验证记录位于 `output/environment_audit/`（该目录被 Git 忽略）：`resource_all_units.log`、`resource_scene_units.log`、`lifecycle_matrix_results.json`、`resource_native_normal_location.log`、`console_shutdown_results.json`、`core_unchanged_results.json`。验证程序也保存在该目录。

在仓库根目录的 Anaconda Prompt 中可重跑已跟踪的测试：

```bat
conda activate mimickit-newton
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s tests/Scene_Test -p "test_*.py" -v
```

直接验证窗口有限测试：

```bat
python mimickit/run.py --arg_file args/smp_steering_humanoid_args.txt --engine_config data/engines/newton_engine.yaml --mode test --devices cuda:0 --num_envs 1 --visualize true --test_episodes 1 --model_file data/models/smp_steering_humanoid_model.pt
```

本地 demo 目录中的旧 `newton_smp_engine.yaml` 引用已改为当前存在的 `newton_engine.yaml`；正式 engine 配置未改。`output/environment_audit/launch.py` 和 catalog 被 Git 忽略，不会随普通 push 同步。另一台电脑可直接使用上面的命令。

## 独立发现与验证边界

一次单环境、32 样本的 TensorBoard 小训练出现 Critic NaN。使用 HEAD 未修改源代码、同样配置进行对照，也复现了 Critic NaN（`resource_baseline_training.log`）。因此退出测试通过不代表训练数值稳定或策略质量验证通过。本次保留训练逻辑，没有针对该独立问题改变算法、优化器或物理参数。

本机 PyTorch 的真实 CPU/Gloo 通信组初始化报 `makeDeviceForHostname(): unsupported gloo device`，因此未完成真实通信组正常训练/销毁验证；真实 worker 回收和通信组销毁调用的单元测试已通过。这与一个进程中 `--num_envs 2` 的仿真测试不同，后者已通过。未为此更换依赖或修改通信计算逻辑。

Isaac Gym、Isaac Lab 和 Linux Xvfb 的原生运行验证仍受本机平台及安装范围限制；表中明确区分原生、模拟与源码检查，未将它们宣称为本机原生通过。

API 参考：[Pyglet 事件弱引用](https://pyglet.readthedocs.io/en/latest/modules/event.html)、[Python 引用环析构顺序](https://peps.python.org/pep-0442/#disposal-of-cyclic-isolates)、[Isaac Sim annotator/render product 清理示例](https://docs.isaacsim.omniverse.nvidia.com/latest/replicator_tutorials/tutorial_replicator_getting_started.html)、[Omniverse 键盘订阅 API](https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html)、[PyTorch distributed shutdown](https://docs.pytorch.org/docs/stable/distributed.html#shutdown)。
