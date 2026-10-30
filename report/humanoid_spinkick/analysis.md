# 官方 Humanoid 回旋踢：SMP 解码与 PCA 分析

本案例把官方回旋踢动作与官方 diffusion prior 生成片段放在同一条分析链中：**采样 → 原生动作解码 → pose/window PCA → 共享基底投影 → 播放与绘图**。原有单文件 PCA、缓存和三种交互视图保留。本轮没有训练模型、运行控制策略或执行远程操作。参数与仅播放命令见 [reproduce.md](reproduce.md)。

主要观察：原始与生成组的 pose 压缩曲线接近；窗口压缩曲线存在差异。最终生成数据在参考 PCA 的前两个坐标上主要沿参考轨迹分布。最早记录的 t=49 干净动作估计较散，后续估计逐渐集中到动作区域。这些是本次有限样本的描述，不能据此判断完整分布或物理控制质量。

## 1. 数据与计算方法

| 数据组 | 片段数 | 每片段帧数 | FPS | 时间戳跨度 | pose 样本 | window 样本 |
|---|---:|---:|---:|---:|---:|---:|
| 官方原始文件 | 1 | 78 | 60 | 1.28333 s | 本轮先重采样 | 本轮先重采样 |
| reference：规则时间网格 | 1 | 39 | 30 | 1.26667 s | 39×28 | 30×280 |
| generated：最终 prior 样本 | 512 | 10 | 30 | 0.30000 s | 5120×28 | 512×280 |
| 各 stage：同一批样本的干净估计 | 512 | 10 | 30 | 0.30000 s | 5120×28 | 512×280 |

原始文件为 `data/motions/humanoid/humanoid_spinkick.pkl`，模型为 `data/models/smp_priors/smp_prior_spinkick.pt`。采用 EMA、DDPM 50 步、seed=0、batch=64。没有复制权重到报告目录。原始文件内容与 60 FPS 元数据保持不变；分析副本通过现有 `MotionLib` 的位置/旋转插值采样到规则 30 FPS 网格，未补入不规则末帧。

**PCA 的一个样本是什么？** pose 的一行是一个姿态的 28 个局部原生 DOF；window 的一行是连续 H=10 帧展平后的 280 个 DOF 坐标，stride=1。每个片段单独切窗，不跨片段连接；10 帧生成片段恰好提供一个合法窗口。原始组与生成组样本数不同，并未人为复制参考样本来配平。

计算沿用 float64 中心化 SVD，不标准化、不 whitening。设样本为 x，均值为 μ，主成分为 u，投影系数为 z，则重建是 `μ + Σ z_k u_k`；均值只加一次，K=0 即均值重建。窗口重建在原片段内等权重叠平均。root 前六维不参与 PCA，重建保留各片段的 root、帧数、FPS 与循环模式。原生球关节的坐标是指数映射分量，不应逐项解释为 Euler 角或真实单轴关节角。

**两类实验分开使用：**

- **独立拟合**：reference 与 generated 分别拟合 pose/window，用于解释方差、K 扫描、载荷和自身重建误差。不同组的 PC 轴不能直接当作同一坐标。
- **共享基底**：只在 reference 拟合，将 reference、generated、六个 stage 投影到这个均值与基底上，用于轨迹、散点、直方图和密度比较。不再为生成组重新选择坐标系。

DOF 指标为原生坐标 RMSE（rad）；FK 指标为沿用 MimicKit CPU FK 得到的身体原点欧氏距离 RMSE（cm），对全部帧和身体等权。FK 使用 float32。指标始终把每个动作的 PCA 重建与**该动作自身**比较；随机生成片段与原始长片段没有时间配对，因此不计算两者逐帧“真实误差”。本轮不 unwrap 指数映射、不裁剪关节限位、不放大 PCA 系数。最终生成组未检测到相邻 DOF 跳变大于 π 的候选事件。

## 2. 图组一：重建质量

### G01–G02：解释方差与 K 扫描

![独立拟合解释方差](assets/plots/G01_variance.png)

![K 与自身重建误差](assets/plots/G02_k_error.png)

K 扫描为 0、1、2、3、4、8、全部有效成分，按有效秩去重。窗口样本的解释方差在展平样本空间计算；DOF/FK 误差在窗口重叠平均后计算，二者不应直接视为同一指标。

| 独立拟合组 | 有效秩 | K=3 累计方差 | K=3 DOF RMSE | K=3 FK RMSE | K=8 FK RMSE |
|---|---:|---:|---:|---:|---:|
| reference / pose | 28 | 77.06% | 0.1804 rad | 12.50 cm | 2.84 cm |
| generated / pose | 28 | 77.49% | 0.1766 rad | 12.94 cm | 2.80 cm |
| reference / window | 29 | 80.24% | 0.1525 rad | 8.84 cm | 1.84 cm |
| generated / window | 280 | 74.27% | 0.1887 rad | 12.20 cm | 3.14 cm |

pose 曲线较接近，说明两组在当前 DOF 表示下的主要方差结构相近。生成窗口的低 K 压缩相对困难，但样本数、起始相位与覆盖方式均有差别，不能把更高的数值秩直接解释为更好的动作多样性。独立拟合的全部有效成分可把 DOF 重建到浮点精度，正式集合逐片段验收的最大绝对误差低于 4×10⁻¹⁵。

共享 reference/window 基底只有 29 个有效方向；generated/window 即使使用这 29 项仍有 **0.09654 rad / 5.23 cm** 的 DOF/FK 残差。生成窗口可以位于参考线性子空间之外，这不等于解码或重建公式出错。pose 参考基底已覆盖全部 28 维，因此其全成分投影没有这一维数限制。

### G03–G04：哪些时间、关节丢失较多？

![代表关节的时间曲线](assets/plots/G03_joint_curves.png)

![逐帧逐 DOF 误差热力图](assets/plots/G04_error_heatmaps.png)

代表坐标为 DOF17（right_knee，铰链角）与 DOF8（right_shoulder.z，指数映射分量，参考中方差最大的 DOF）。G03 展示原始与各自 PCA 的 K=1/2/3，横轴是各片段自身时间；上下两组没有对齐动作相位。低 K 会遗漏膝盖的尖峰变化，即使总体解释方差较高，局部细节仍可能不同。G04 的四面板共用误差色标，横轴实际长度不同。

对应 GIF：以下每格均为独立、完整短片段；没有把生成片段延长成完整回旋踢。

| reference 完整 | reference K=1 | reference K=2 | reference K=3 |
|---|---|---|---|
| ![reference](assets/gifs/reference_00_full.gif) | ![reference K1](assets/gifs/reference_00_K1.gif) | ![reference K2](assets/gifs/reference_00_K2.gif) | ![reference K3](assets/gifs/reference_00_K3.gif) |

| generated 样本0 完整 | generated 样本0 K=1 | generated 样本0 K=2 | generated 样本0 K=3 |
|---|---|---|---|
| ![generated](assets/gifs/generated_00_full.gif) | ![generated K1](assets/gifs/generated_00_K1.gif) | ![generated K2](assets/gifs/generated_00_K2.gif) | ![generated K3](assets/gifs/generated_00_K3.gif) |

## 3. 图组二：姿态结构与时间曲线

![前三个成分载荷](assets/plots/G05_loadings.png)

G05 使用独立基底。pose 每行是一个主成分，每列为一个 DOF；window 每个面板分成三个 PC 区块，每块 28 列、10 行，保留全部时间载荷，未沿窗口平均。颜色是单位范数主成分的无量纲载荷。主成分的符号以最大绝对载荷为正固定，便于数值复现；组间仍不保证一一对应。

![三维 PCA 轨迹](assets/plots/G06_pca_trajectories.png)

G06 的三轴是**同一参考 pose 基底**的 PC1/PC2/PC3 系数（rad），每个点是姿态，连线表示片段内部的时间顺序，黑方块表示起点。生成样本0–3覆盖的是参考轨迹的不同局部区间，并非四条完整动作循环。三维图可以显示时间轨迹的形状，但没有概率轴。

![时间与两个关节坐标](assets/plots/G07_time_joint_curve.png)

G07 的 x 为时间（s），y/z 为 DOF17/DOF8（rad）。这是用户希望看到的“动作随时间变化”的三维曲线。它既不是密度曲面，也不是把姿态编号当作连续姿态轴；两组 y/z 范围一致，时间范围按实际片段长度保留。

## 4. 图组三：经验分布与低维密度

![pose 共享散点和直方图](assets/plots/G08_pose_scatter_histograms.png)

![window 共享散点和直方图](assets/plots/G08_window_scatter_histograms.png)

G08 的横纵轴是共享 PC 系数，不是方差。pose 点代表帧，window 点代表窗口；两种实验的轴不可混用。直方图使用相同 bin 边界并按面积归一化，避免 5120 对 39 的样本量直接影响柱高。参考动作按时间均匀采样得到的是经验姿态占用分布；generated 的帧在片段内相关，5120 帧不等于 5120 个独立动作。

### G09：三维概率密度曲面和配套等高线

![pose 高斯密度](assets/plots/G09_pose_gaussian_density.png)

![pose KDE 密度](assets/plots/G09_pose_kde_density.png)

![window 高斯密度](assets/plots/G09_window_gaussian_density.png)

![window KDE 密度](assets/plots/G09_window_kde_density.png)

这里的三维安排为 **x=PC1 系数、y=PC2 系数、z=该二维投影的概率密度**（1/rad²）。单个点的概率不是曲面的高度；一个区域的概率要对该区域积分。原始数据也可以做密度估计，不需要先经过 SMP。

高斯面用各组二维均值与最大似然协方差拟合，适合观察整体中心与宽度，但会填平环形或多峰结构。KDE 为等权高斯核混合；核协方差只由 reference 的 Scott 带宽确定，再**原封不动用于生成组和所有 stage**。pose 核协方差对角约为 (0.52821, 0.26933) rad²；window 约为 (6.10368, 3.21882) rad²。高斯模型的协方差是拟合参数，不是 KDE 平滑参数。

同一图内原始与生成共用 x/y 网格、z 范围和色标；密度按整平面积分为1定义，不对显示范围重新缩放。曲面只显示有限网格，画面内积分可以小于1。实现已与 SciPy 的 reference KDE 核对，并验证大范围数值积分。G08 散点上的参考轨迹比 G09 的宽曲面更细：这正说明带宽影响观感，KDE 平滑填出的区域不能被当作真实采样或生成能力证据。

最终样本0–3的局部播放与 G06/G08/G09 对照：

| 样本0 | 样本1 | 样本2 | 样本3 |
|---|---|---|---|
| ![sample0](assets/gifs/generated_00_full.gif) | ![sample1](assets/gifs/generated_01_full.gif) | ![sample2](assets/gifs/generated_02_full.gif) | ![sample3](assets/gifs/generated_03_full.gif) |

## 5. 图组四：Diffusion 去噪阶段

记录的实际 timestep 为 49、39、29、19、9、0。**t=49 是第一次反向去噪的输入时刻，不是第49次迭代。** 每个时刻同时保存 normalized 输入 x_t 和预测 clean x₀；后者反归一化后解码。下面绘图、GIF使用的是 clean 估计，不能把这些图称为原始 x_t 噪声分布。六组保持同一批 sample ID，t=0 的估计与最终输出逐位一致。

![pose 各阶段共享密度](assets/plots/G10_pose_stages.png)

![pose 各阶段共享散点](assets/plots/G11_pose_stage_scatter.png)

![window 各阶段共享密度](assets/plots/G10_window_stages.png)

![window 各阶段共享散点](assets/plots/G11_window_stage_scatter.png)

G10 的白圈为参考样本，红色为 sample0；window 每个片段只有一个窗口，因此红色只出现一个点。G11 保留所有样本，蓝色为参考、橙色为生成阶段估计。各阶段共用参考基底、坐标范围、KDE 核与密度色标。t=49 的姿态估计分散且可能不自然，后续逐渐靠近参考动作区域；密度变尖本身并不是质量评分。阶段的演化也不是角色沿物理时间从 t=49 播到 t=0。

相同 sample0 在六个阶段的独立 0.3 s 姿态片段：

| t=49 | t=39 | t=29 |
|---|---|---|
| ![t49](assets/gifs/stage_t49_sample00.gif) | ![t39](assets/gifs/stage_t39_sample00.gif) | ![t29](assets/gifs/stage_t29_sample00.gif) |

| t=19 | t=9 | t=0 |
|---|---|---|
| ![t19](assets/gifs/stage_t19_sample00.gif) | ![t9](assets/gifs/stage_t09_sample00.gif) | ![t0](assets/gifs/stage_t00_sample00.gif) |

## 6. 解码含义、显示与局限

prior 每帧114维：root位置3、root六维旋转6、14个关节六维旋转84、5个关键身体的 root 相对位置15、root线速度/角速度6。原生 motion 每帧34维：root位置3、root指数映射3、局部 DOF28。解码复用既有 `tan_norm_to_quat` 与 `rot_to_dof`，没有另加 Gram–Schmidt、限位或接触修正。时间轴沿训练数据的过去到当前顺序。

原始生成特征完整保留。关键点和速度等冗余量并非 motion `.pkl` 的独立字段，不能声称114维到34维是可逆无损转换。本次最终样本的 raw 关键点与解码后 FK 关键点 RMSE 约 **3.29 cm**；root位置前向差分与相邻预测速度中点值的坐标 RMSE 约 **0.129 m/s**，后者还含离散差分误差。完整诊断在 [decode_diagnostics.json](assets/data/decode_diagnostics.json)。

GIF 通过 Newton 实际渲染，隐藏 UI，只显示一个角色；固定 root 水平位置、朝向与统一参考高度，局部关节旋转保留。17段共用同一镜头与裁剪框，输出520×520，无随帧跟随的裁剪；所有帧已检查角色完整与边界。采样时刻为严格30 FPS；GIF格式仅支持10 ms时间量化，采用30/40 ms交替。39帧的时间戳跨度是1.26667 s，含最后一帧保持后的循环周期为1.30 s；10帧对应0.30 s跨度、0.33 s编码周期。短片段回到开头的跳变是重播，不是生成连续过渡。

固定 root 后，原动作的整体转身、跳起和位移不会显示，脚滑、悬空或穿插不代表这里进行了物理接触处理。交互播放器仍可即时切回 `comparison` 查看原生 root，或使用 `classroom` 查看完整局部动作与 K=1/2/3。

本案例只有一段回旋踢和512个短生成片段，没有独立测试动作、置信区间或物理控制评估。二维/三维投影与低维密度不能代表完整高维分布；在新区域出现样本也不能证明生成质量更好。reference 窗口之间高度重叠，生成片段的起始相位与原始规则切窗也不完全等价。后续可以先增加多个 seed、片段级重采样置信区间及多动作对照，再接入策略实际轨迹。

## 7. 程序组织与验收

```text
采样与解码（不建场景）
run_smp.py → smp/sampling.sample_prior()
  → TinyMDM / EMA / 原 DDPM scheduler
  → 可选 trace_callback(t, x_t, predicted_x0)
  → 反归一化 → 既有旋转解码 → samples.npz + manifest

集合分析（离线）
prepare_case.py → workflows/spinkick.py → data/resampling.resample_motion(reference, 30 FPS)
  → core/collection.fit_collection(reference/generated, pose/window)
  → core/collection.project_collection(all groups, reference model)
  → save_collection() → analysis.json + 去重模型/帧/系数快照

仅播放（不采样、不拟合）
play_analysis.py → core/collection.load_collection() → for_clip(index)
  → AnalysisSession.reconstruct(indices)
  → player/environment.py 中的 InMemoryMotionLib / PcaPlaybackEnv
  → 动画插值 → 可选固定 root → MimicKit FK → Newton/Warp 渲染

报告输出（读取已保存结果）
reporting/metrics.py / reporting/density.py → 数值与密度
case_plots.py → reporting/plots.py → Matplotlib PNG / 数值网格
case_gifs.py → reporting/gifs.py → 同一 Newton 场景逐时刻渲染 → 固定裁剪 GIF
```

PCA公式、窗口还原与成分选择位于 `core/`。原生数据与原子写入位于 `data/`，公共路径位于 `paths.py`；SMP不依赖PCA session进行持久化。`player/`、`smp/`、`reporting/`分开管理，`workflows/`仅负责案例编排，顶层脚本保留现有命令。采样缓存包含权重/配置/env/character内容哈希、种子、batch、EMA、设备及处理版本。完整缓存复用不推理、不执行SVD；损坏结果可由准备入口重建。播放入口仅校验、载入快照，缺少结果时报错，不自动重新拟合。Git仅保留本案例两份Markdown文档，`assets/`保留在本地。

验收使用 `mimickit-newton`：67项 motion/PCA/CLI 回归与9项 SMP采样/案例缓存测试通过（普通测试跳过2项 opt-in GL）；另分三个真实 Newton 进程验证 reference/pose、generated/window、stage_49/pose，覆盖视图切换、暂停、成分更新、单步、重启与关闭。带阶段回调的采样与正常EMA采样输出、随机状态一致。正式512集合验证有限值、形状、片段边界、全成分重建与缓存命中；17图逐图检查，17 GIF逐帧校验时间、帧数和裁剪，抽查仅播放命令未导入采样模块或执行SVD。

## 8. 论文图示对应

| 资料 | 核对的图示/用途 | 本案例取舍 |
|---|---|---|
| [Safonova、Hodgins、Pollard，SIGGRAPH 2004](https://graphics.cs.cmu.edu/nsp/papers/sig04.pdf) | 低维动作表示与随维数变化的重建误差 | G01/G02学习压缩效果；没有复现论文的动力学优化 |
| [Wang、Fleet、Hertzmann，TPAMI 2008](https://www.dgp.toronto.edu/~jmwang/gpdm/pami.pdf) | Fig.3(a)为行走数据的三维PCA初始化轨迹，后续图是GPLVM/GPDM潜空间 | G06采用纯PCA三维轨迹；没有把GPDM置信度当作本案例概率密度 |
| [Urtasun等，CGF 2004](https://www.cs.toronto.edu/~urtasun/publications/urtasun-etal-cgf04.pdf) | Fig.4绘制速度与前两个成分系数；其动作向量含时间离散后的整段关节角 | 当前短窗口并非其整周期风格模型，暂不添加速度控制曲线 |
| [SMP论文](https://arxiv.org/abs/2512.03028) | 冻结motion diffusion prior用于下游策略的score/SDS奖励 | 本轮只观察prior生成与去噪，不等同论文中的策略执行结果 |

G09 的高斯/KDE曲面是本案例为理解数据分布增加的低维统计视图，不声称是上述论文图的逐项复现。
