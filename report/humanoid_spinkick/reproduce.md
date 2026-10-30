# 参数与仅播放命令：Humanoid spinkick

先在仓库根目录进入环境：

```powershell
conda activate mimickit-newton
```

以下均默认已激活环境，且工作目录为仓库根目录。所有 Python 命令**只载入已有分析并播放**，不采样、不训练、不拟合 PCA。清单缺失/损坏会明确报错。本页的命令复现对应数据和成分选择；GIF使用另行导出的单角色、隐藏UI与统一裁剪，交互播放器显示原始/重建对照或课件四角色。

## 固定参数与身份

| 参数 | 本次取值 |
|---|---|
| 源 motion | `data/motions/humanoid/humanoid_spinkick.pkl` |
| prior | `data/models/smp_priors/smp_prior_spinkick.pt` |
| prior 配置 | `tools/diffusion_model/config/tinymdm_single_clip.yaml` |
| 推理 | EMA / DDPM / 50步 / seed=0 / 512样本 / batch=64 / cuda:0 |
| 片段 | 每段10帧、30 FPS、CLAMP；时间戳跨度0.3 s |
| 阶段 | 实际timestep 49、39、29、19、9、0的预测clean x₀ |
| 参考副本 | 39帧、30 FPS、WRAP；时间戳跨度1.26667 s |
| 特征 | root前6维保留；局部原生DOF28；float64中心化SVD；不标准化/whitening |
| pose | 每帧一行；reference 39样本，generated 5120样本 |
| window | H=10，stride=1；不跨片段；reference 30窗口，generated 512窗口 |
| K 扫描 | 0、1、2、3、4、8、全部有效成分；独立秩28/28/29/280 |
| 基底 | `independent`用于自身压缩；`shared`仅在reference拟合 |
| 播放清单 | `report/humanoid_spinkick/assets/data/analysis.json` |
| 处理版本 | sampling schema=1 / version=v1；case_version=1；collection schema=1 |
| GIF | 30 FPS采样，30/40 ms量化，固定root/朝向/统一高度，520×520，同一裁剪框 |

SHA-256：

```text
motion     0e2468b1d2812bf625a6847c8aaa53dceb5cb58a0a0ec8e78f0744656dfc77e3
prior      86d19ae21efda6d494e4d13801d4a82a41ac1ca4ad971243fba955ea2f71e064
config     ca12eee3cb43b44e84dcf658bb5067a56ccfa07860bd8576e51a41c614a7a5c1
env        dcfd00fec05af9ef5b9c2032d80ac150c85a095718e66f62038088c30c39fb4d
character  34d83859e7de61c12ab4ad5adcb2016c8658599a3567a2801e8c377f2a9db5ee
```

完整来源在 [case.json](assets/data/case.json) 与 [采样清单](assets/data/smp/SMP_v1_7e37e84f3edb22024c7b/manifest.json)。[metrics.json](assets/data/metrics.json)包含全部K扫描，[plots.json](assets/data/plots.json)包含图文件/带宽，[gifs.json](assets/data/gifs.json)包含每个GIF的样本/选择/时间/相机/裁剪/哈希，[media_validation.json](assets/data/media_validation.json)包含逐帧边界与时间检查。

## 图组与资产对应

| 图组 | 图文件 | 对应 GIF/片段 | 使用基底 |
|---|---|---|---|
| G01–G02 重建质量 | `G01_variance`、`G02_k_error` | reference/generated样本0完整与K1/2/3 | independent，pose/window |
| G03–G04 时间曲线/热力图 | `G03_joint_curves`、`G04_error_heatmaps` | `reference_00_*`、`generated_00_*` | independent；G03为pose |
| G05 载荷 | `G05_loadings` | 同上，在课件视图查看前三项 | independent，pose/window |
| G06 三维PCA轨迹 | `G06_pca_trajectories` | reference完整；`generated_00..03_full` | shared pose |
| G07 时间—两个DOF | `G07_time_joint_curve` | reference完整、generated样本0完整 | 原生动作；不需要PCA重建 |
| G08 共享散点/直方图 | `G08_pose/window_scatter_histograms` | reference与generated样本0–3 | shared pose/window |
| G09 三维密度/等高线 | `G09_pose/window_gaussian/kde_density` | 同G08；密度使用全部样本，GIF只是示例 | shared pose/window |
| G10–G11 阶段分析 | `G10_pose/window_stages`、`G11_pose/window_stage_scatter` | `stage_t49/39/29/19/09/00_sample00` | shared pose/window，GIF播放原生clean解码 |

所有图均在 `assets/plots/`，GIF与同名中间帧PNG在 `assets/gifs/`；[contact_sheet.png](assets/gifs/contact_sheet.png)是17段GIF的中间帧索引。GIF编号从0开始，CLI的成分编号从1开始。

## A. 原始参考与 K=1/2/3（G01–G07）

完整参考动作与其全成分重建；左侧为保存的完整动作，右侧为重建：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 28 --view local
```

课件视图：完整局部动作在上，累计K=1/2/3在下。对应 `reference_00_full/K1/K2/K3.gif`：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --view classroom
```

两角色下分别检查K=1、2、3：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 1 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 3 --view local
```

窗口课件展示（G01/G02/G04/G05下排）：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode window --basis independent --view classroom
```

## B. 最终生成样本0–3与压缩（G01–G09）

四条命令分别对应 `generated_00_full.gif` 至 `generated_03_full.gif`。全28项重建保留完整pose，显示各片段真实0.3 s时间跨度：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 1 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 2 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 3 --mode pose --basis independent --components 28 --view local
```

样本0与独立拟合的K=1/2/3（`generated_00_K1/K2/K3.gif`）：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --view classroom
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 1 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 3 --view local
```

生成组窗口独立拟合，以及共享参考窗口基底的投影残差对照：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis independent --view classroom
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis independent --components 280 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis shared --components 29 --view local
```

G06/G08/G09使用共享参考pose坐标。下面课件视图中的下排是共享基底的K=1/2/3，不能与独立拟合GIF的下排混称：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis shared --view classroom
```

## C. 相同样本0的六个去噪阶段（G10–G11）

对应 `stage_t49_sample00.gif` 至 `stage_t00_sample00.gif`。每个阶段都是预测clean x₀解码后的10帧动作，使用全28项pose投影，因此右侧与左侧原生解码在浮点精度内一致：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_49 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_39 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_29 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_19 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_9 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_0 --clip 0 --mode pose --basis shared --components 28 --view local
```

窗口阶段投影的示例：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_49 --clip 0 --mode window --basis shared --components 29 --view local
```

该命令左侧始终为完整clean解码，右侧为29维参考窗口子空间重建。其他阶段替换 `--group` 即可；它们没有 `independent` 分析项。

## D. 交互与其他观察方式

显示原生root位移、朝向和起伏，使用默认跟随镜头：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 4 --view comparison
```

均值、单个成分、自选集合也只调用已保存模型的解码，不重新拟合：

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --components 0 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --selection single --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --selection subset --pcs "1,3,5-8" --view local
```

侧栏支持三种视图即时切换、滑条与±1、累计/单项/子集、播放/暂停、单步、重启及保存。Enter播放/暂停，Space单步；编辑输入框时隔离快捷键。`local`固定root和镜头；`classroom`固定完整动作+K1/2/3，隐藏自选/保存控件。预览只在内存；点保存才在案例 `assets/data/reconstructions/` 下持久化选定motion。保存结果保留源root、FPS与循环模式，显示变换不会写回数据。

## prior 片段与未来策略轨迹

本报告的“生成”来自冻结 diffusion prior 的直接采样，短片段没有经过角色控制、动力学或接触约束。SMP论文中的策略使用prior导出的score/SDS奖励学习控制；策略执行轨迹是另一种数据来源。本轮没有将官方控制策略接入Newton，也没有验证其后端兼容性，不能把这里的GIF称为策略rollout。

后续若通过SSH在算力机器上训练，应另行保存训练配置、模型/代码哈希、随机种子、控制频率、角色资产及策略执行环境；导出**执行轨迹**为原生motion与来源元数据。接入位置为 `core.collection.fit_collection/project_collection` 的 `MotionData`片段集合，新增 `policy_rollout`数据组即可复用共享投影、绘图与播放器。长轨迹按明确边界分片，保留episode与时间信息；本轮未执行任何SSH或训练操作。
