# Motion PCA

对原生 MimicKit motion 做 pose/window PCA、缓存重建，并用 Newton 播放原始与重建动作。
另外提供冻结 SMP prior 的采样/解码、多片段集合分析和案例报告导出；不在工具中训练控制策略。

## 代码架构

顶层保留命令入口，具体实现按职责分包：

```text
tools/motion_analysis/
├─ run_pca.py                    单个 motion：离线 PCA，可选 --play
├─ run_smp.py                    冻结 SMP prior：采样、阶段记录和原生导出
├─ play_analysis.py              仅播放已保存的集合分析，不采样、不拟合
├─ prepare_case.py               案例准备入口 → workflows/spinkick.py
├─ case_plots.py                 绘图入口 → reporting/plots.py
├─ case_gifs.py                  GIF 入口 → reporting/gifs.py
├─ paths.py                     仓库、数据、缓存位置及动画层导入路径
│
├─ data/                        数据读写与持久化
│  ├─ motion.py                 MotionData、原生 motion 读写、DOF 跳变检查
│  ├─ resampling.py             通过现有 MotionLib 在规则时钟上重采样
│  └─ storage.py                原子写入，供所有子包共用
│
├─ core/                        PCA 计算与缓存，不依赖播放器、SMP 或 Newton
│  ├─ pca.py                    float64 中心化 SVD、投影与逆变换
│  ├─ samples.py                pose/window 样本与窗口重叠合并
│  ├─ selection.py              累计/单项/子集成分索引
│  ├─ session.py                单片段 AnalysisSession、拟合/重建缓存
│  └─ collection.py             多片段拟合、共享基底投影、分析快照读写
│
├─ smp/
│  └─ sampling.py               prior 加载、EMA/DDPM、阶段记录、解码及采样缓存
│
├─ player/                      播放与显示适配，不实现 PCA 或 diffusion 数学
│  ├─ environment.py            内存动作库、同步角色、暂停桥接、play_session()
│  ├─ controls.py               侧栏、K 滑条和±1、子集与播放请求
│  ├─ view.py                   三种布局、固定相机和课件累计选择
│  └─ renderer.py               Newton 可见性与相机输入兼容适配
│
├─ reporting/                   报告与媒体导出
│  ├─ reconstruction.py         单片段 PCA 的误差/方差/系数报告
│  ├─ metrics.py                回旋踢案例 K 扫描、FK 与解码一致性诊断
│  ├─ density.py                低维高斯/KDE 密度估计
│  ├─ plots.py                  保存的回旋踢集合 → 科学绘图
│  └─ gifs.py                   保存的回旋踢集合 → Newton 渲染与固定裁剪
│
├─ workflows/
│  └─ spinkick.py               官方回旋踢：采样 → 重采样 → 集合分析 → 清单
└─ PCA_Result/                  单 motion 缓存，位置保持不变，Git 忽略
```

`data/` 与 `paths.py` 是公共基础；`core/` 和 `smp/` 各自使用基础模块，SMP 不再依赖
PCA session 来写文件。`workflows/` 负责把它们串起来，案例参数与执行顺序不放进数学核心。
`reporting/gifs.py` 是需要播放器的媒体导出路径；普通离线 PCA 与统计绘图不创建 Newton 场景。

PCA 与播放器通过 `AnalysisSession` 连接：播放器调用 `session.reconstruct(indices)`
获取完整帧，调用 `session.save_reconstruction(indices)` 显式保存。集合分析通过
`CollectionAnalysis.for_clip(index)` 适配到同一接口。窗口合并只由 `core/samples.py` 实现，
调整成分不重新拟合。

输入来自 `data/motions/`。单文件结果继续保存到本目录 `PCA_Result/`；回旋踢案例结果
保存到根目录 `report/humanoid_spinkick/assets/`。Git 仅保留该案例的 `analysis.md` 和
`reproduce.md`，生成资产仍保留在本机。现有六个脚本入口、CLI 参数和缓存格式保持兼容。
内部 Python 导入已统一到上述子包路径，不保留旧平铺模块的别名；仓库内调用与测试已同步更新。
Newton 的公共 UI/暂停接口仍位于 `mimickit/engines/newton_engine.py`。

## 命令速查

### 基础命令

在仓库根目录、已安装 MimicKit 依赖的 Python 环境中运行：

```powershell
conda activate mimickit-newton
```

基础命令仅进行离线 PCA、保存重建与生成报告；默认逐帧姿态 PCA，累计前 4 个有效成分：

```powershell
python tools/motion_analysis/run_pca.py --motion humanoid/humanoid_getup_faceup.pkl
```

在同一基础命令后增加 `--play`，处理或命中缓存后打开并排播放器：

```powershell
python tools/motion_analysis/run_pca.py --motion humanoid/humanoid_getup_faceup.pkl --play
```

| 基础选项 | 必填/默认 | 功能 |
|---|---|---|
| `--motion <相对路径>` | 必填 | 输入原始 `.pkl`，相对于 `data/motions/`，如 `humanoid/humanoid_getup_faceup.pkl`。 |
| `-h` / `--help` | 可选 | 查看 CLI 帮助，不处理数据、不打开窗口。 |

### PCA 的 CLI 选项

下列选项控制计算、成分选择和缓存，可追加在基础命令后：

| 选项 | 默认值 | 功能与示例 |
|---|---|---|
| `--mode pose\|window` | `pose` | 样本组织方式：每帧姿态或连续多帧窗口。 |
| `--window <H>` | `10` | 每个窗口的帧数；仅 `window` 模式使用。 |
| `--stride <步长>` | `1` | 相邻窗口起点的帧间隔；仅 `window` 模式使用。 |
| `--selection cumulative\|single\|subset` | `cumulative` | 累计前 K 项、仅第 K 项、任意指定成分集合；始终包含平均项。 |
| `--components <K>` | `min(4, 有效秩)` | `cumulative` 的累计数量或 `single` 的成分编号；显式越界报错。累计模式允许 `0`，表示仅平均特征。 |
| `--pcs "成分集合"` | 无 | `subset` 模式必填，如 `"1,3,5-8"`；编号从 1 开始，重复编号去重，`""` 表示空集合。 |
| `--version <版本名>` | `v1` | 缓存版本标签；例如 `--version experiment2`，更换标签会建立独立缓存。 |
| `--force` | 关闭 | 重算当前请求的拟合与重建，不清空其他缓存。 |

`--pcs` 只用于 `subset`，不能与 `--components` 同时指定。
窗口要求 `1 <= stride <= H <= 帧数`，且至少构成两个样本。

### 播放器的 CLI 选项

单 motion 的处理与播放共用 `run_pca.py`，PCA 成分选项决定打开窗口时的初始重建。
集合分析另有仅播放入口 `play_analysis.py`（见下文）。下列选项用于 `run_pca.py`：

| 选项 | 默认值 | 功能 |
|---|---|---|
| `--play` | 关闭 | 打开原始/重建动作的交互并排播放器；省略时仅离线处理。 |
| `--view comparison\|local\|classroom` | `comparison` | 原始轨迹双角色 / 固定 root 双角色 / 完整动作与累计 K=1、2、3 四角色；窗口内也可即时切换。 |
| `--env-config <路径>` | `data/envs/view_motion_humanoid_env.yaml` | `view_motion` 环境模板，指定角色骨架与环境设置；**同时供离线 FK 误差报告使用**。处理其他角色也需指定匹配配置。 |
| `--engine-config <路径>` | `data/engines/newton_engine.yaml` | 播放器后端配置；仅播放时使用，当前交互播放器要求 Newton。 |
| `--device <设备>` | `cuda:0` | 播放器运行设备；不会改变离线 PCA 的 CPU 计算。 |

`--env-config` 和 `--engine-config` 的相对路径以仓库根目录解析，也可使用绝对路径。
原始动作和 PCA 结果由入口自动传给播放器，不需要另写动作路径到环境 YAML。

### 集合分析的仅播放入口

已有分析清单可以直接播放，不重新采样或拟合。以下命令要求本地已有回旋踢案例资产：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --view classroom
```

| 选项 | 默认值 | 功能 |
|---|---|---|
| `--manifest` | 必填 | 保存的 `analysis.json` 清单。 |
| `--group` | `reference` | 数据组，如 `generated`、`stage_49`。 |
| `--clip` | `0` | 组内片段编号，从 0 开始。 |
| `--mode pose\|window` | `pose` | 选择已经保存的分析方式，不在此入口切窗或拟合。 |
| `--basis shared\|independent` | `shared` | 共享参考基底，或该组自身独立拟合的基底。 |
| `--selection`、`--components`、`--pcs` | 累计前 `min(4, rank)` 项 | 与单 motion 入口相同的成分选择语义。 |
| `--view`、`--device` | `comparison`、`cuda:0` | 与上述播放器选项一致。 |
| `--env-config`、`--engine-config` | 清单配置或 Humanoid/Newton 默认配置 | 可覆盖播放环境与后端。 |

此入口没有 `--play`、`--force`、`--window` 或采样参数；缺少快照时明确报错。
完整案例参数、图/GIF 对应与仅播放命令见
[回旋踢复现文档](../../report/humanoid_spinkick/reproduce.md)。生成资产已忽略，只有两份 Markdown
文档会随 Git 分发；另一台机器需要先生成或取得对应 `assets/data/` 才能使用这些播放命令。

### 常用参数组合

下表参数追加到基础命令后；需要显示窗口时再加 `--play`：

| 目的 | 追加参数 |
|---|---|
| 逐帧 PCA，累计前 4 项 | `--mode pose --components 4` |
| 10 帧窗口 PCA，累计前 8 项 | `--mode window --window 10 --stride 1 --components 8` |
| 仅第 3 个成分 | `--selection single --components 3` |
| 指定成分组合 | `--selection subset --pcs "1,3,5-8"` |
| 仅平均特征 | `--selection subset --pcs ""` |
| 使用新版本名 | `--version experiment2` |
| 强制重新处理 | `--force` |

例如，俯卧起身的窗口 PCA 与并排播放：

```powershell
python tools/motion_analysis/run_pca.py --motion humanoid/humanoid_getup_facedown.pkl --mode window --window 10 --stride 1 --components 8 --play
```

`--motion` 固定相对于 `data/motions/`，必须包含角色子目录。工具自身根据脚本位置定位仓库，
不依赖调用时的工作目录。省略 `--play` 时只处理、保存和生成报告，不加载 Newton。
离线 PCA 在 CPU 上运行；`--device cuda:0` 只用于播放器。

默认使用 Humanoid 的环境模板与骨架。处理其他角色时，通过 `--env-config` 指定匹配的
`view_motion` 配置；播放可用 `--engine-config` 指定 Newton 配置。骨架 DOF 必须与 motion 一致。

## 两种样本组织方式

- `pose`：每帧关节 DOF 是一行，矩阵大小为 `[T, D]`。
- `window`：连续 H 帧是一个样本，矩阵大小为 `[M, H*D]`。只在当前动作内部切窗，
  末尾自动补最后一个合法窗口起点；重建使用等权 overlap-add，使每一帧都得到覆盖。
- `1 <= stride <= H <= T`，至少需要两个样本。`H=1, stride=1` 与 pose 等价。
- 保留源动作的 FPS、帧数、循环模式和 root 前 6 维。窗口模式不意味着整段动作只算一个样本。

PCA 使用 float64 中心化 SVD，不做标准化或 whitening。设选中成分集合为 I，重建统一为：

```text
reconstructed_samples = mean + scores[:, I] @ components[I]
```

`cumulative K` 选择前 K 项；`single K` 只选择第 K 项；`subset` 选择任意集合。
重复编号去重，平均项始终只加一次。默认 K 为 `min(4, 有效秩)`；显式越界时报错。
常量数据有效秩为零，只提供平均特征结果。

窗口模式的平均项是一个长度 H 的平均窗口；重叠合并后不一定是静态姿态，
因此界面的 Mean features only 表示不保留任何主成分，而不是将整段冻结为一帧。

原生角度和指数映射存在表示分支。工具记录可能的相邻 DOF 跳变，不自动展开三维指数映射、
裁剪关节限位或宣称已恢复物理合理性。较低 PCA 误差或较高解释方差也不保证接触和动作自然。

## 缓存与结果

缓存位于本目录的 `PCA_Result/`，已加入 Git 忽略规则。原始 motion 不会被覆盖。

1. 拟合缓存由输入内容哈希、处理版本、pose/window 和窗口参数确定，保存均值、
   主成分、系数和布局；更换 K 不重新拟合。
2. 重建缓存由拟合标识和规范化后的成分集合确定。累计前三项与子集 `1,2,3` 共用结果。
3. `--version v1` 是可读的版本标签，完整内容和参数仍参与缓存校验。
4. `--force` 重算当前请求，不清空其他版本或成分组合。

CLI 输出结果路径；原生 `.pkl` 可以直接由现有 `ViewMotionEnv` 播放。报告记录选中成分、
解释方差、DOF 重建误差，以及用匹配骨架 FK 计算的 body 世界位置平均/最大误差（cm）。
同时生成解释方差与系数曲线；窗口系数按窗口起始时间绘制。

缓存使用临时文件和原子替换；缺失或损坏的缓存重新计算。报告和播放依赖不同：
播放启动失败不删除已经完成的离线结果。

若在受限运行环境中遇到 Warp 默认编译缓存的写入权限错误，可以在启动前设置
`$env:WARP_CACHE_PATH = "$PWD/output/pca-warp-cache"`，将编译缓存指向可写目录。

## 播放器窗口内操作（非 CLI 选项）

**View** 可在三个模式间即时切换，播放时间、暂停状态、自选成分和已编辑子集保留：

| 模式 | 角色和布局 | root 与相机 |
|---|---|---|
| `comparison`（默认） | 蓝色原始动作、橙色当前自选重建，双角色 | 保留原始 root 轨迹、起伏和旋转；相机跟随两者中点，可自由调整。 |
| `local` | 同样的双角色对比 | 固定 root 水平位置、源动作首帧高度与标准朝向，固定斜前方镜头。 |
| `classroom` | 上方蓝色完整局部动作；下方从左到右橙色 K=1、绿色 K=2、紫色 K=3 | 与 `local` 相同的 root 固定规则；角色排列在相机图像平面内，大小一致。 |

课件展示建议使用逐帧姿态 PCA。仓库根目录运行以下单行命令：

```powershell
conda run --no-capture-output -n mimickit-newton python tools/motion_analysis/run_pca.py --motion humanoid/long/humanoid_long_walk0.pkl --mode pose --view classroom --play
```

仅固定 root 的双角色模式：

```powershell
conda run --no-capture-output -n mimickit-newton python tools/motion_analysis/run_pca.py --motion humanoid/long/humanoid_long_walk0.pkl --mode pose --components 3 --view local --play
```

窗口 PCA 也支持三种显示布局；侧栏显示当前 `pose/window` 分析方式。课件布局始终显示
**均值 + 累计前 K 项**，与 CLI 的当前自选重建分别管理；有效成分不足 K 时采用全部可用
成分并显示实际数量，常量数据为均值重建。课件布局隐藏自选与保存按钮，切回双角色恢复它们。

局部模式在动画采样之后、FK 之前固定 root，不修改动作帧、PCA 模型或缓存；导出仍然保留
完整原始 root 运动。腰部等局部关节引起的摇摆仍会显示。固定 root 后可能出现脚部滑动或
离地，这是隔离关节姿态的展示效果，没有加入接触修正。`Restore standard camera` 恢复
局部模式的标准镜头；原始对比模式保留此前的相机视角。课件布局隐藏地面和参考点标记，
避免显示槽位的竖直偏移与地面相交。

PCA 播放器单独使用 2.5 米显示间距。MimicKit 的 Newton 后端统一
将 UI 默认字号设为 18 px，PCA 控件继承该设置，不额外放大；默认缩放下原生控件与 PCA 控件均为 18 px。
Newton 侧栏中可选择：

- **Cumulative / Single**：通过 K 滑块或其下方的 `-1` / `+1` 按钮调整累计数量或单个成分编号；自动限制在有效范围内。
- **Subset**：输入 `1,3,5-8`，点击 Apply 预览。
- **Save**：将当前预览结果写入缓存；滑块和模式切换不会自动保存所有组合。
- **Play/Pause / Step / Restart**：播放、暂停、单步和回到开头；暂停时也能更新 PCA 结果。

Enter 切换播放，Space 单步；编辑输入框时快捷键不触发。单步推进的是一个播放器控制步，
不一定是一个源数据帧。Newton 默认 Pause 与工具暂停控件会同步。

修改成分只重建内存中的短动作，不重新做 PCA、不创建新场景，也不重置当前时间。
单独成分的动作包含均值和该成分的原始系数轨迹，并非随机生成的新动作。

## 输入数据异常

`Motion contains non-finite frame values` 表示输入 `.pkl` 已含 NaN/Inf，检查发生在
pose/window 样本构建之前。错误会给出文件路径、NaN/Inf 数量、受影响帧数与从 0 开始的
列号和位置。切换 PCA 模式或使用 `--force` 不会修复输入数据。

当前仓库的 `data/motions/humanoid/humanoid_walk.pkl` 在全部 125 帧的第 9、10、11 列
含 NaN，共 375 个。可先使用两段 `humanoid_getup_faceup.pkl` / `humanoid_getup_facedown.pkl`
验证窗口 PCA。工具不会自动把缺失值填零；整列缺失时需要依据原始数据修复或重新导出。

## 验证

```powershell
python -m unittest discover -s tests -p "test_motion_pca*.py" -v
```

单元测试覆盖重建、成分选择、窗口边界、缓存和播放控制。实际 GL 窗口仍应检查
并排显示、三种模式、暂停切换、同步单步及关闭资源。

可选的真实 Newton/CUDA 集成测试会打开窗口，通过应用接口验证上述状态转换。
各用例使用单独的 Python 进程：

```powershell
$env:MIMICKIT_PCA_GL_TEST = "1"
$env:MIMICKIT_PCA_GL_CASE = "faceup"
python -m unittest discover -s tests -p "test_motion_pca_gl.py" -v
$env:MIMICKIT_PCA_GL_CASE = "facedown"
python -m unittest discover -s tests -p "test_motion_pca_gl.py" -v
$env:MIMICKIT_PCA_GL_CASE = "walk"
python -m unittest discover -s tests -p "test_motion_pca_gl.py" -v
$env:MIMICKIT_PCA_GL_CASE = "walk_window"
python -m unittest discover -s tests -p "test_motion_pca_gl.py" -v
Remove-Item Env:MIMICKIT_PCA_GL_TEST, Env:MIMICKIT_PCA_GL_CASE
```
