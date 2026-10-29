# Motion PCA

对原生 MimicKit motion 做离线 PCA、缓存重建结果，并用 Newton 比较原始与重建动作。
第一版使用原生关节 DOF，不涉及训练、diffusion 或重定向。

## 代码架构：PCA 计算与播放器分开

```text
run_pca.py                         统一 CLI 入口，连接下面两部分
│
├─ PCA 计算与结果管理              不加载 Newton
│  ├─ motion_data.py              motion 读写、元数据、DOF 跳变检查
│  ├─ samples.py                  pose/window 样本构造与窗口合并
│  ├─ pca.py                      拟合、投影、逆变换的数学核心
│  ├─ selection.py                将累计/单成分/任意集合转换为成分索引
│  ├─ session.py                  AnalysisSession、两级缓存、按需重建
│  └─ report.py                   误差、解释方差、系数图；按需调用 CPU FK
│
└─ 播放器                         仅指定 --play 时加载
   ├─ playback_controls.py        侧栏控件、成分选择、播放操作请求
   └─ playback.py                 双角色、共享时间、暂停同步、内存动作库
       └─ 复用 MimicKit 动画层 → NewtonEngine → Newton Viewer
```

两部分通过 `AnalysisSession` 连接：播放器调用 `session.reconstruct(indices)` 取得完整重建帧，
调用 `session.save_reconstruction(indices)` 显式保存。播放器不实现 PCA 数学或窗口合并；
切换成分只重建，不重新拟合。计算部分也可完全独立于窗口运行。

输入来自仓库 `data/motions/`；生成结果统一进入本目录 `PCA_Result/`。
Newton 适配层的 UI/暂停桥接位于 `mimickit/engines/newton_engine.py`。

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

播放器与 PCA 共用上述入口，没有另一个播放器 CLI。PCA 成分选项决定打开窗口时的初始重建；
下列选项控制是否播放及播放配置：

| 选项 | 默认值 | 功能 |
|---|---|---|
| `--play` | 关闭 | 打开原始/重建动作的交互并排播放器；省略时仅离线处理。 |
| `--env-config <路径>` | `data/envs/view_motion_humanoid_env.yaml` | `view_motion` 环境模板，指定角色骨架与环境设置；**同时供离线 FK 误差报告使用**。处理其他角色也需指定匹配配置。 |
| `--engine-config <路径>` | `data/engines/newton_engine.yaml` | 播放器后端配置；仅播放时使用，当前交互播放器要求 Newton。 |
| `--device <设备>` | `cuda:0` | 播放器运行设备；不会改变离线 PCA 的 CPU 计算。 |

`--env-config` 和 `--engine-config` 的相对路径以仓库根目录解析，也可使用绝对路径。
原始动作和 PCA 结果由入口自动传给播放器，不需要另写动作路径到环境 YAML。

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

两个不同颜色的角色分别代表 Original 与 Reconstructed，共用时间、相同 root 和循环规则。
PCA 播放器单独使用 2.5 米显示间距，相机跟随两者中点。PCA 控件沿用 Newton/ImGui
默认字体，并在绘制时局部增加 2 px；共享引擎配置与其他界面的字号不受影响。
Newton 侧栏中可选择：

- **Cumulative / Single**：通过 K 滑块调整累计数量或单个成分编号。
- **Subset**：输入 `1,3,5-8`，点击 Apply 预览。
- **Save**：将当前预览结果写入缓存；滑块和模式切换不会自动保存所有组合。
- **Play/Pause / Step / Restart**：播放、暂停、单步和回到开头；暂停时也能更新 PCA 结果。

Enter 切换播放，Space 单步；编辑输入框时快捷键不触发。单步推进的是一个播放器控制步，
不一定是一个源数据帧。Newton 默认 Pause 与工具暂停控件会同步。

修改成分只重建内存中的短动作，不重新做 PCA、不创建新场景，也不重置当前时间。
单独成分的动作包含均值和该成分的原始系数轨迹，并非随机生成的新动作。

## 验证

```powershell
python -m unittest discover -s tests -p "test_motion_pca*.py" -v
```

单元测试覆盖重建、成分选择、窗口边界、缓存和播放控制。实际 GL 窗口仍应检查
并排显示、三种模式、暂停切换、同步单步及关闭资源。

可选的真实 Newton/CUDA 集成测试会打开窗口，通过应用接口验证上述状态转换。
两个用例分别启动一个 Python 进程：

```powershell
$env:MIMICKIT_PCA_GL_TEST = "1"
$env:MIMICKIT_PCA_GL_CASE = "faceup"
python -m unittest discover -s tests -p "test_motion_pca_gl.py" -v
$env:MIMICKIT_PCA_GL_CASE = "facedown"
python -m unittest discover -s tests -p "test_motion_pca_gl.py" -v
Remove-Item Env:MIMICKIT_PCA_GL_TEST, Env:MIMICKIT_PCA_GL_CASE
```
