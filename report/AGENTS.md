# Report authoring standard

- Write the English source report first, directly from the experiment artifacts,
  implementation and verified references. Do not create it by translating an
  existing Chinese report.
- Use `analysis.md` and `reproduce.md` for the authoritative English editions in
  each case directory. Then provide `analysis.zh-CN.md` and `reproduce.zh-CN.md`
  as translations of the English editions.
- Identify the source/translation status and link the corresponding language
  edition near the beginning of each document. Update English first when the
  experiment or interpretation changes, then synchronize the Chinese translation.
- Keep the two language editions aligned in findings, limitations, numerical
  values, units, figure/GIF IDs, citations and executable commands. Translation
  must not introduce additional scientific claims or omit material limitations.
- Keep figures, GIFs and numerical artifacts under the case's `assets/` tree and
  reference the same files from both languages. Do not duplicate assets by language.
- In `reproduce.md`, show `conda activate mimickit-newton` once near the beginning,
  assume that environment for subsequent commands, and keep Python command examples
  to single-line playback-only commands. Record parameters, provenance and the
  mapping between figures, GIFs and playback selections.
- Preserve the distinction between prior samples and policy execution trajectories,
  and between projected density estimates and the full motion distribution.
- Write for a reader with basic PCA knowledge but little probability, motion,
  neural-network or reinforcement-learning background, and for a supervisor who
  may be new to character animation. Use clear academic prose without assuming
  that domain abbreviations or implementation names explain themselves.
- Introduce a short reader guide and a suggested reading order. Define pose,
  motion clip, skeleton/DOF, root, distribution/density, sampling, diffusion prior,
  and policy before relying on them. Explain ML/DL/RL relationships accurately;
  do not present RL as a compulsory stage after deep learning.
- Give every major section and result group a brief introduction: the question
  it answers, what data are used, how to read its axes/colors/units, the observed
  result, and the limit of that interpretation. Add explanations beside the
  figures as well as in the opening conceptual overview.
- Include a small number of core mathematical expressions when they clarify
  PCA projection/reconstruction, variance/error, density estimation or denoising.
  Define every symbol and its dimensions or units nearby, explain the equation's
  practical meaning, and connect it to the implementation. Keep derivations and
  unused advanced objectives out of the main reading path.
- Distinguish PCA scores from statistical diffusion scores, PCA reconstruction
  from generated-feature decoding, physical clip time from diffusion timesteps,
  and model sampling from training or policy execution. Describe early-stage
  artifacts, such as unstable t=49 replay, where the relevant results appear.
- Separate measured findings, explanatory examples and hypotheses. Do not turn
  explained variance into a percentage of motion correctness, projected density
  into physical plausibility, or implementation tests into scientific validation.
- For explanation-only revisions, retain result values, plots, GIFs, experiment
  settings and executable commands. Check these against a baseline, and keep the
  English source and Chinese translation synchronized in their explanations and
  formulas as well as in their results.
- Validate local links, numerical consistency, command parity and Git ignore rules.
  Report assets remain local; Git retains this standard and the four named Markdown
  editions per case.
