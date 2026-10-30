# Humanoid Spinkick: Experiment Record and Saved-Result Playback

**English source edition.** [中文译本](reproduce.zh-CN.md) · [Analysis and interpretation](analysis.md)

From the repository root, activate the environment once:

```powershell
conda activate mimickit-newton
```

All subsequent commands assume that environment and working directory. Every Python example below is a **playback-only** command: it reads an existing analysis snapshot, reconstructs the selected PCA components and opens the viewer. It does not sample the prior, train a model or fit PCA. Missing or corrupt snapshots cause an explicit error.

The commands reproduce the data and PCA selections associated with the figures and GIFs. Interactive playback uses two comparison actors or the four-actor classroom layout. The exported GIFs additionally use a single actor, hidden UI, a common reference root height and a fixed crop; the commands are not a pixel-identical GIF export recipe.

## First viewing pass

For terminology, start with the analysis report's [reader guide](analysis.md#reader-guide). Then use this order:

1. In catalog A, open the reference classroom view. The full motion is the comparison target; K=1/2/3 show progressively richer PCA approximations of that same recording.
2. In B, view full generated clips 0–3, then clip 0's classroom reconstruction. Each generated clip covers a short interval, so its restart is a replay boundary rather than a continuous kick.
3. In C, compare clip 0 at t=49 and t=0. These are early and final clean-motion estimates from one denoising run, not two policies or two training checkpoints.
4. Return to the shared-basis plots and window comparisons once those motions are familiar. The plots summarize the collections; a few GIFs are examples to help interpret them.

## 1. Frozen experiment configuration

This table fixes the experiment's inputs and computation settings. It is a record of how the saved results were obtained, not a list of quality ratings or parameters that the playback commands retrain.

| Item | Recorded value |
|---|---|
| Source motion | `data/motions/humanoid/humanoid_spinkick.pkl` |
| Prior checkpoint | `data/models/smp_priors/smp_prior_spinkick.pt` |
| Prior configuration | `tools/diffusion_model/config/tinymdm_single_clip.yaml` |
| Sampling | EMA; DDPM; 50 steps; seed 0; 512 clips; batch 64; `cuda:0` |
| Generated clip | 10 frames at 30 FPS; CLAMP; 0.30000 s first-to-last timestamp |
| Recorded stages | Predicted clean x₀ at actual timesteps 49, 39, 29, 19, 9, 0 |
| Reference copy | 39 frames at 30 FPS; WRAP; 1.26667 s first-to-last timestamp |
| PCA features | 28 local native DOFs; root's first six coordinates copied through; float64 centered SVD; no standardization or whitening |
| Pose samples | One frame per row: 39 reference rows; 5120 generated rows |
| Window samples | H=10, stride=1; no cross-clip windows; 30 reference rows; 512 generated rows |
| Independent ranks | Reference pose 28; generated pose 28; reference window 29; generated window 280 |
| K scan | 0, 1, 2, 3, 4, 8, full retained rank; clipped and deduplicated |
| Saved index | `report/humanoid_spinkick/assets/data/analysis.json` |
| Processing identity | Sampling schema 1 / version v1; case version 1; collection schema 1 |
| GIF output | 30 FPS sampling; 30/40 ms quantized holds; fixed root/heading/camera/crop; 520 × 520 |

A checkpoint is the saved learned model. EMA selects averaged training weights; DDPM is the iterative denoising sampling procedure. Seed controls the random sequence, batch size groups samples for computation, and FPS controls physical replay speed. CLAMP holds the stored endpoint during sampling beyond the clip, while WRAP repeats its timeline. A short CLAMP clip can still be restarted by the player; restart does not generate new connecting frames.

Content identities (SHA-256):

```text
motion     0e2468b1d2812bf625a6847c8aaa53dceb5cb58a0a0ec8e78f0744656dfc77e3
prior      86d19ae21efda6d494e4d13801d4a82a41ac1ca4ad971243fba955ea2f71e064
config     ca12eee3cb43b44e84dcf658bb5067a56ccfa07860bd8576e51a41c614a7a5c1
env        dcfd00fec05af9ef5b9c2032d80ac150c85a095718e66f62038088c30c39fb4d
character  34d83859e7de61c12ab4ad5adcb2016c8658599a3567a2801e8c377f2a9db5ee
```

## 2. Reading selections and views correctly

A playback command chooses a saved data group, a PCA representation, a basis, a clip and a component selection. The basis determines the numerical reconstruction; the view determines how actors and their roots are displayed. Switching view does not change the fitted PCA model.

| Option or convention | Meaning |
|---|---|
| `--clip` | Zero-based clip ID; final clips and each stage use matching IDs |
| `--basis independent` | The selected group's own fitted basis, for compression experiments |
| `--basis shared` | The reference-fitted basis, for comparable projected coordinates |
| `--components K` | Mean plus the first K contributions in cumulative mode; K=0 is mean-only |
| `--selection single --components K` | Mean plus component K, numbered from 1 |
| `--selection subset --pcs ...` | Mean plus an explicit set of 1-based component IDs |
| `--view local` | Full source and selected reconstruction, with fixed root and camera |
| `--view classroom` | Full local motion above cumulative K=1/2/3; custom selection and Save controls hidden |
| `--view comparison` | Native root motion and a following camera |

The saved-result entry point requires no `--play` flag. It has no sampling, force-refit or window-construction options. Each requested group/mode/basis combination must already exist. Stage groups have a shared basis only.

In the sidebar, a slider and ±1 buttons adjust K; selection modes, pause/play, step, restart and view changes remain available. Enter toggles playback and Space advances one control step. Text fields capture shortcuts while being edited. Preview stays in memory; Save persists a selected native motion under `assets/data/reconstructions/`. Saved root motion, FPS and loop mode remain unchanged by display transforms.

## 3. Figure-to-replay map

Use the figure IDs to connect a statistical result with a visible example of the same type of data. A density plot may use all generated samples even though its companion GIFs show only clips 0–3. The selected clips do not replace the collection-wide measurements.

The table uses filename abbreviations; all plot files are PNGs in `assets/plots/`, and all replay files are GIFs in `assets/gifs/`.

| Figure IDs | Plot stems | Motion examples | Basis and catalog section |
|---|---|---|---|
| G01–G02 | `G01_variance`, `G02_k_error` | Reference and generated clip 0, full and K1/2/3 | Independent pose/window; A–B |
| G03–G04 | `G03_joint_curves`, `G04_error_heatmaps` | `reference_00_*`, `generated_00_*` | Independent; G03 is pose; A–B |
| G05 | `G05_loadings` | The same K1/2/3 examples, including classroom playback | Independent pose/window; A–B |
| G06 | `G06_pca_trajectories` | Reference full; `generated_00..03_full` | Shared pose coordinates; B |
| G07 | `G07_time_joint_curve` | Reference full and generated clip 0 full | Native motion coordinates; A–B |
| G08 | `G08_pose/window_scatter_histograms` | Reference and final clips 0–3 | Shared pose/window; B |
| G09 | `G09_pose/window_gaussian/kde_density` | The same examples; densities use all samples | Shared pose/window; B |
| G10–G11 | `G10_pose/window_stages`, `G11_pose/window_stage_scatter` | `stage_t49/39/29/19/09/00_sample00` | Shared coordinates; native decoded clean estimates in GIFs; C |

The 17 GIFs have same-stem PNG posters. [The contact sheet](assets/gifs/contact_sheet.png) indexes their middle frames. Clip numbers start at 0, while component IDs start at 1. A ten-frame generated clip is a short interval, not a complete spinkick.

## 4. Playback catalog

The four groups below progress from a known recording to generated samples, denoising estimates and custom PCA selections. No command in the catalog changes the underlying results or starts a new experiment.

### A. Reference motion and its compression

This establishes the visual baseline: what the stored kick looks like and which details survive a low-dimensional approximation. Start with the classroom layout if the two-actor comparison is unfamiliar. K=1 means one coordinated direction across many joints, not animation of just one joint.

Full reference motion with a full-component reconstruction; source is on the left, reconstruction on the right:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 28 --view local
```

Classroom playback for `reference_00_full/K1/K2/K3.gif`:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --view classroom
```

Separate two-actor comparisons for K=1, 2 and 3:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 1 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode pose --basis independent --components 3 --view local
```

Window classroom playback for the window panels of G01/G02/G04/G05:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --clip 0 --mode window --basis independent --view classroom
```

### B. Final samples and independent/shared reconstructions

This shows completed samples from the frozen generator. Clip IDs identify different sampled intervals, not quality ranks. The generated collection's independent basis was fitted across the entire generated group; selecting `--clip 0` only chooses the replayed clip and does not fit a separate model for it. A shared-basis reconstruction instead uses reference-derived directions.

Final clips 0–3, corresponding to `generated_00_full.gif` through `generated_03_full.gif`. All 28 pose components recover the complete decoded pose sequence:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 1 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 2 --mode pose --basis independent --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 3 --mode pose --basis independent --components 28 --view local
```

Generated clip 0 with independent pose K=1/2/3, corresponding to `generated_00_K1/K2/K3.gif`:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --view classroom
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 1 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 3 --view local
```

Window comparisons: classroom, all independent components, then all 29 shared reference components:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis independent --view classroom
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis independent --components 280 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode window --basis shared --components 29 --view local
```

The final command above exposes the reference-window subspace residual. It is not an independent full-rank reconstruction.

G06/G08/G09 use reference-shared coordinates. The following classroom's lower actors use that shared basis, so their low-K reconstructions should not be confused with the independent-basis GIFs:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis shared --view classroom
```

### C. The same clip 0 at six denoising stages

This isolates the sampling process while keeping the clip ID and neural-network parameters fixed. An erratic early GIF is a poor clean estimate under strong noise. Watch each clip's frames for physical motion, and compare separate stage GIFs to see refinement. The timestep is not seconds or a training-epoch number.

These commands correspond to `stage_t49_sample00.gif` through `stage_t00_sample00.gif`. Each clip is a decoded predicted clean x₀, not the normalized noisy input x_t. Using all 28 shared pose components makes the right reconstruction agree with the left native decoding up to numerical precision:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_49 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_39 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_29 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_19 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_9 --clip 0 --mode pose --basis shared --components 28 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_0 --clip 0 --mode pose --basis shared --components 28 --view local
```

A shared-window projection example:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group stage_49 --clip 0 --mode window --basis shared --components 29 --view local
```

Here the left actor is the full decoded clean estimate, while the right actor is its 29-component reference-window projection. Change `--group` to inspect another recorded stage. Diffusion timesteps are separate estimates of the same physical clip, not successive intervals to concatenate.

### D. Native root motion and custom component sets

Use this group to separate whole-body travel from local posture and to inspect individual PCA contributions. A single-component reconstruction includes the mean plus that component's original time-varying coefficient. It does not mean holding a character at a fixed basis vector or drawing a new random motion.

Restore root displacement, heading and height changes with the following-camera view:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group generated --clip 0 --mode pose --basis independent --components 4 --view comparison
```

Mean-only, component 2 only, and a custom subset, all using existing models:

```powershell
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --components 0 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --selection single --components 2 --view local
python tools/motion_analysis/play_analysis.py --manifest report/humanoid_spinkick/assets/data/analysis.json --group reference --mode pose --basis independent --selection subset --pcs "1,3,5-8" --view local
```

## 5. Artifact records and availability

These records make the report auditable. The manifest is an index of saved arrays and their identities; it is not a neural network. A hash is a fingerprint of file contents, useful for distinguishing changed inputs even when filenames are unchanged. Scientific interpretation comes from the analysis; these records establish which files and settings produced it.

- [case.json](assets/data/case.json): source metadata, resampled duration and case identity.
- [Sampling manifest](assets/data/smp/SMP_v1_7e37e84f3edb22024c7b/manifest.json): checkpoint/configuration hashes and sampling identity.
- [metrics.json](assets/data/metrics.json): all K scans and shared-basis residuals.
- [plots.json](assets/data/plots.json): plot inventory, density grids, bandwidths and representative DOF IDs.
- [gifs.json](assets/data/gifs.json): clip IDs, basis/selection, timestamps, camera, crop and file hashes.
- [decode_diagnostics.json](assets/data/decode_diagnostics.json): redundant-feature consistency measurements.
- [media_validation.json](assets/data/media_validation.json) and [validation.json](assets/data/validation.json): frame-boundary/timing checks and recorded implementation acceptance.

Both languages reference the same artifacts. Git retains the English sources and Chinese translations, while generated assets remain local and ignored. A fresh checkout therefore needs the corresponding saved `assets/data/` before these playback commands can run. This page records playback, not a regeneration or training recipe.

## 6. Extension point for future policy trajectories

A policy trajectory would be recorded after a controller repeatedly chooses actions and a physics environment advances the character. It would let a later report examine executed behavior, rather than merely whether sampled poses can be decoded and displayed. This is an additional experiment, not another interpretation of the current GIFs.

A direct sample from the frozen prior has not passed through a controller, dynamics or contact constraints. SMP policy execution uses prior-derived score/SDS rewards and produces a different data source. The present GIFs are not policy rollouts, and this case does not establish Newton compatibility for official control policies.

If training later runs over SSH on a compute machine, retain training configuration, model/code hashes, seeds, control frequency, character asset and rollout environment. Export actual execution trajectories as native motion clips with provenance, episode IDs and timestamps. A `policy_rollout` group can enter through `core.collection.fit_collection/project_collection` using `MotionData`, then reuse the shared projection, reporting and playback paths. No SSH operation or training was performed in this case.
