# Humanoid 回旋踢：实验记录与已有结果播放

**中文译版，英文为原版。** [English source](reproduce.md) · [分析与解释](analysis.zh-CN.md)

在仓库根目录激活一次环境：

```powershell
conda activate mimickit-newton
```

后续命令均假定使用该环境与工作目录。下面每条Python示例都是**仅播放命令**：读取已有分析快照、重建所选PCA成分，然后打开播放器。它们不采样prior、不训练模型，也不拟合PCA。快照缺失或损坏时明确报错。

这些命令复现图与GIF对应的数据和PCA选择。交互播放采用两个对比角色或四角色课件布局。导出的GIF还采用单角色、隐藏UI、统一参考root高度与固定裁剪框；下面的命令并非逐像素一致的GIF导出步骤。

## 初次观看顺序

术语可先阅读分析报告的[阅读导引](analysis.zh-CN.md)。然后按以下顺序观看：

1. 在命令目录A中打开参考动作的课件视图。完整动作为对比目标，K=1/2/3展示对同一记录逐渐增加细节的PCA近似。
2. 在B中观看完整生成片段0–3，再看片段0的课件重建。生成片段只覆盖短区间，因此重新开始是播放边界，不是连续回旋踢。
3. 在C中比较相同片段0的t=49与t=0。它们是一次去噪过程中的早期与最终干净动作估计，不是两个策略，也不是两个训练checkpoint。
4. 熟悉这些动作后，再返回共享基底绘图与窗口对比。图概括整个集合，少量GIF则是帮助解释图的示例。

## 1. 固定实验配置

本表固定实验输入与计算设置，记录保存结果的来源；它不是质量评分表，也不是播放命令会重新训练的参数清单。

| 项目 | 记录值 |
|---|---|
| 源motion | `data/motions/humanoid/humanoid_spinkick.pkl` |
| Prior checkpoint | `data/models/smp_priors/smp_prior_spinkick.pt` |
| Prior配置 | `tools/diffusion_model/config/tinymdm_single_clip.yaml` |
| 采样 | EMA；DDPM；50步；seed 0；512片段；batch 64；`cuda:0` |
| 生成片段 | 10帧、30 FPS；CLAMP；首末帧时间戳跨度0.30000 s |
| 记录阶段 | 实际timestep 49、39、29、19、9、0的预测干净x₀ |
| 参考副本 | 39帧、30 FPS；WRAP；首末帧时间戳跨度1.26667 s |
| PCA特征 | 28个局部原生DOF；root前六个坐标直接保留；float64中心化SVD；不标准化、不whitening |
| Pose样本 | 每帧一行：参考39行；生成5120行 |
| Window样本 | H=10，stride=1；不跨片段；参考30行；生成512行 |
| 独立拟合秩 | 参考pose 28；生成pose 28；参考window 29；生成window 280 |
| K扫描 | 0、1、2、3、4、8、全部保留秩；限制范围并去重 |
| 保存清单 | `report/humanoid_spinkick/assets/data/analysis.json` |
| 处理身份 | Sampling schema 1 / version v1；case version 1；collection schema 1 |
| GIF输出 | 30 FPS采样；30/40 ms量化停留；固定root/朝向/镜头/裁剪框；520 × 520 |

checkpoint是保存的已学习模型；EMA选择训练权重的平均版本，DDPM是反复去噪的采样过程。seed控制随机序列，batch size将样本分组计算，FPS控制实际重放速度。CLAMP表示超出片段时采样保持末端状态，WRAP表示时间线循环。播放器仍可重启一个短CLAMP片段，但重启不会生成新的连接帧。

内容身份（SHA-256）：

```text
motion     0e2468b1d2812bf625a6847c8aaa53dceb5cb58a0a0ec8e78f0744656dfc77e3
prior      86d19ae21efda6d494e4d13801d4a82a41ac1ca4ad971243fba955ea2f71e064
config     ca12eee3cb43b44e84dcf658bb5067a56ccfa07860bd8576e51a41c614a7a5c1
env        dcfd00fec05af9ef5b9c2032d80ac150c85a095718e66f62038088c30c39fb4d
character  34d83859e7de61c12ab4ad5adcb2016c8658599a3567a2801e8c377f2a9db5ee
```

## 2. 正确理解选择与视图

一条播放命令选定已保存的数据组、PCA表示、基底、片段与成分集合。基底决定数值重建，视图决定角色及其root如何显示。切换视图不会改变已拟合的PCA模型。

| 选项或约定 | 含义 |
|---|---|
| `--clip` | 从0开始的片段ID；最终片段与各阶段使用对应ID |
| `--basis independent` | 所选集合自身拟合的基底，用于压缩实验 |
| `--basis shared` | 仅在参考组拟合的基底，用于可比较的投影坐标 |
| `--components K` | 累计模式下为均值加前K项贡献；K=0表示仅均值 |
| `--selection single --components K` | 均值加第K个成分，成分编号从1开始 |
| `--selection subset --pcs ...` | 均值加明确指定的成分集合，编号从1开始 |
| `--view local` | 完整源动作与所选重建，固定root和镜头 |
| `--view classroom` | 上方完整局部动作，下方累计K=1/2/3；隐藏自选与Save控件 |
| `--view comparison` | 保留原生root运动，使用跟随镜头 |

已有结果入口不需要 `--play`，也没有采样、强制重拟合或窗口构建选项。请求的数据组/模式/基底组合必须已经存在。阶段数据组仅提供共享基底。

侧栏的滑条与±1按钮用于调整K，还可使用成分选择、暂停/播放、单步、重启和视图切换。Enter切换播放，Space推进一个控制步；编辑文本框时隔离快捷键。预览只保存在内存，Save将所选原生motion写入 `assets/data/reconstructions/`。显示变换不会改变保存的root运动、FPS或循环模式。

## 3. 图与播放内容对应

通过图号，把统计结果与相同类型数据的可见示例联系起来。密度图可能使用全部生成样本，即使配套GIF只展示片段0–3。所选片段不能代替整个集合的测量结果。

下表使用文件名简写；所有图为 `assets/plots/` 下的PNG，所有播放文件为 `assets/gifs/` 下的GIF。

| 图号 | 图文件名前缀 | 动作示例 | 基底与命令目录 |
|---|---|---|---|
| G01–G02 | `G01_variance`、`G02_k_error` | 参考与生成片段0，完整及K1/2/3 | 独立pose/window；A–B |
| G03–G04 | `G03_joint_curves`、`G04_error_heatmaps` | `reference_00_*`、`generated_00_*` | 独立；G03为pose；A–B |
| G05 | `G05_loadings` | 相同K1/2/3示例，包括课件播放 | 独立pose/window；A–B |
| G06 | `G06_pca_trajectories` | 参考完整；`generated_00..03_full` | 共享pose坐标；B |
| G07 | `G07_time_joint_curve` | 参考完整与生成片段0完整 | 原生动作坐标；A–B |
| G08 | `G08_pose/window_scatter_histograms` | 参考与最终片段0–3 | 共享pose/window；B |
| G09 | `G09_pose/window_gaussian/kde_density` | 相同示例；密度使用全部样本 | 共享pose/window；B |
| G10–G11 | `G10_pose/window_stages`、`G11_pose/window_stage_scatter` | `stage_t49/39/29/19/09/00_sample00` | 共享坐标；GIF为原生解码的干净估计；C |

17段GIF均附有同名PNG预览帧。[联系表](assets/gifs/contact_sheet.png)列出其中间帧。片段编号从0开始，成分编号从1开始。十帧生成片段只是一个短区间，并非完整回旋踢。

## 4. 播放命令目录

以下四组从已知记录，逐步过渡到生成样本、去噪估计与自选PCA成分。目录中的命令都不会改变底层结果，也不会开始新实验。

### A. 参考动作与压缩

先建立视觉基准：保存的回旋踢是什么样，以及低维近似保留了哪些细节。不熟悉双角色比较时，可以先看课件布局。K=1表示跨多个关节的一个协调方向，不是只让一个关节运动。

完整参考动作与全成分重建：左侧为源动作，右侧为重建。

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 28 --view local
```

`reference_00_full/K1/K2/K3.gif`对应的课件播放：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --view classroom
```

分别对比K=1、2、3的双角色播放：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 1 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 3 --view local
```

G01/G02/G04/G05的window面板对应的窗口课件播放：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode window --basis independent --view classroom
```

### B. 最终样本及独立/共享基底重建

这里展示冻结生成器完成的样本。片段ID区分不同采样区间，不是质量排名。生成集合的独立基底是在整个生成组上拟合的，`--clip 0`只选择播放哪段，不为该片段单独拟合模型。共享基底重建则使用参考数据得到的方向。

最终片段0–3，对应 `generated_00_full.gif` 至 `generated_03_full.gif`。全部28个pose成分恢复完整的解码姿态序列：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 1 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 2 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 3 --mode pose --basis independent --components 28 --view local
```

生成片段0的独立pose K=1/2/3，对应 `generated_00_K1/K2/K3.gif`：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --view classroom
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 1 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 3 --view local
```

窗口对比依次为：课件、全部独立成分、全部29个共享参考成分。

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis independent --view classroom
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis independent --components 280 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis shared --components 29 --view local
```

上面的最后一条命令显示参考窗口子空间的残差，它不是独立模型的全秩重建。

G06/G08/G09使用共享参考坐标。下面课件布局的下方角色也使用该共享基底，因此低K重建不能与独立基底GIF混为一谈：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis shared --view classroom
```

### C. 相同片段0的六个去噪阶段

这里保持片段ID与神经网络参数不变，单独观察采样过程。早期GIF异常跳动，是强噪声下较差的干净估计。观察单个片段内部帧可看实际动作，比较各阶段GIF可看细化过程。timestep既不是秒，也不是训练轮数。

以下命令对应 `stage_t49_sample00.gif` 至 `stage_t00_sample00.gif`。每个片段是解码后的预测干净x₀，不是归一化噪声输入x_t。使用全部28个共享pose成分时，右侧重建与左侧原生解码在数值精度内一致：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_49 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_39 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_29 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_19 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_9 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_0 --clip 0 --mode pose --basis shared --components 28 --view local
```

共享窗口投影示例：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_49 --clip 0 --mode window --basis shared --components 29 --view local
```

这里左侧是完整解码的干净估计，右侧是其29成分参考窗口投影。修改 `--group` 可检查其他已记录阶段。diffusion timestep是对同一物理片段的不同估计，不是可拼接的连续时间区间。

### D. 原生root运动与自选成分

这一组用于区分整体位移与局部姿态，并观察单个PCA贡献。单成分重建包括均值，以及该成分原有的随时间变化系数；它不是让角色保持在固定基向量上，也不是重新随机生成动作。

使用跟随镜头视图恢复root位移、朝向与高度变化：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 4 --view comparison
```

仅均值、仅成分2、任意子集，均使用已有模型：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --components 0 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --selection single --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --selection subset --pcs "1,3,5-8" --view local
```

## 5. 资产记录与可用性

这些记录使报告可核查。manifest是保存数组及其身份的索引，不是神经网络。哈希是文件内容的指纹，即使文件名不变，也有助于区分输入变化。科学解释来自分析报告，这些记录则明确使用了哪些文件与设置。

- [case.json](assets/data/case.json)：源元数据、重采样后时长与案例身份。
- [采样清单](assets/data/smp/SMP_v1_7e37e84f3edb22024c7b/manifest.json)：checkpoint/配置哈希与采样身份。
- [metrics.json](assets/data/metrics.json)：全部K扫描与共享基底残差。
- [plots.json](assets/data/plots.json)：图清单、密度网格、带宽与代表DOF编号。
- [gifs.json](assets/data/gifs.json)：片段ID、基底/成分选择、时间戳、镜头、裁剪与文件哈希。
- [decode_diagnostics.json](assets/data/decode_diagnostics.json)：冗余特征一致性测量。
- [media_validation.json](assets/data/media_validation.json)及[validation.json](assets/data/validation.json)：帧边界/时间检查及已记录的实现验收。

两种语言引用相同资产。Git保留英文原版与中文译版，生成资产仍保留本地并被忽略。因此，新检出的仓库需要先具备对应的 `assets/data/`，才能运行这些播放命令。本页记录播放方法，不是重新生成数据或训练的步骤。

## 6. 未来策略轨迹的扩展位置

策略轨迹需要在控制器反复选择动作、物理环境推进角色之后记录。它让后续报告能够检查实际执行行为，而不只是采样姿态能否被解码与显示。这是一项额外实验，不是对当前GIF换一种解释。

冻结prior的直接样本没有经过控制器、动力学或接触约束。SMP策略执行使用prior导出的score/SDS奖励，属于另一种数据来源。本报告的GIF不是策略rollout，也没有证明官方控制策略与Newton兼容。

后续若通过SSH在算力机器上训练，应保留训练配置、模型/代码哈希、seed、控制频率、角色资产与轨迹执行环境。将实际执行轨迹导出为原生motion片段，附带来源、episode ID和时间戳。`policy_rollout`数据组可以通过 `core.collection.fit_collection/project_collection` 接收 `MotionData`，然后复用共享投影、报告与播放路径。本案例没有执行SSH或训练操作。
