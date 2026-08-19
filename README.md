# refeas — dynamic-feasibility screen for retargeted humanoid motion data

Answers, per frame of a retargeted reference: **could any controller on this robot supply the
forces this motion demands, given the contacts actually available?** ~1 CPU-second per clip.

Retargeting pipelines routinely emit references that are *kinematically* clean (no joint-limit
violations, smooth velocities) yet *dynamically* impossible: bodies descending at ~1 g with no
contact within reach, or transitions where the required support exceeds what friction cones and
actuator torque limits permit. Kinematic QC cannot see this. In a 10,705-clip AMASS→Unitree-G1
bank, 22.8 % of clips had >10 % dynamically infeasible frames (ground-contact category 39 %,
dynamic 59 %; by source dataset anywhere from 0.1 % to 100 %) — and the single worst training
attractor in our curriculum experiments turned out to be a clip whose kneel descent is airborne
for a full second (~329 N ≈ body weight unsupported). See the companion note.

## Method

Per frame:
1. `q, q̇` from the reference; `q̈` by smoothed central differences;
2. contact-free inverse dynamics (`mj_inverse`, contacts disabled) → the 6-D base wrench **W** the
   environment must supply + joint torques;
3. candidate contacts = collision geoms within `--gap` (default 6 cm) of the plane
   (`mj_geomDistance`);
4. contact forces in pyramidal friction cones explaining W: NNLS (unconstrained residual) and a
   torque-limited LP (HiGHS) for the smallest unsupported wrench achievable within actuator force
   ranges.

Flags: `airborne` (no candidate contact), `infeasible` (torque-limited unsupported wrench
> ½ robot weight). Two contact models: `real` (uniform μ) and `sim` (per-geom condim from the
model, e.g. frictionless shins).

## Install

```bash
pip install mujoco scipy numpy
```

## Worked example (Unitree G1)

```bash
python -m refeas.screen --model examples/g1_flat.xml --bank /path/to/bank \
    --clip demo_hover \
    --gap 0.06 --out out.json            # per-frame + binned report
python -m refeas.screen ... --brief --out brief.json   # clip-level features only (fast batch mode)
```

Reference `.npz` schema (per clip): `joint_pos (T,nj)`, `joint_vel (T,nj)`, `body_pos_w (T,nb,3)`,
`body_quat_w (T,nb,4)` (wxyz), `body_lin_vel_w`, `body_ang_vel_w`, `fps`. Body 0 is the floating
base; joint order must match the model's hinge order.

## Output schema (`--brief`)

| field | meaning |
|---|---|
| `airborne_frac` | fraction of frames with no collision geom within `gap` of the plane |
| `infeasible_frac` | fraction of frames whose torque-limited unsupported wrench > 0.5 × weight |
| `unsupported_impulse_Ns`, `unsupported_impulse_per_weight_s` | ∫ unsupported force dt (absolute, and normalised by weight — seconds of free fall equivalent) |
| `torque_infeasible_frac` | frames where no in-limit torque assignment exists at all |
| `max_tau_ratio_p95` | 95th percentile of max joint |τ|/limit |
| `sim_infeasible_frac` | `infeasible_frac` under the model's own condim (e.g. frictionless knees) |

Full (non-`--brief`) output adds per-frame records and 0.25 s bins (contacts, unsupported force,
worst joints).

## Known limits

Genuine flight (jumps) registers as airborne but usually not as infeasible (free fall needs no
support) — yet take-off/landing frames with mistimed contacts can be flagged; report per-category.
The screen evaluates the *retargeted output on the target robot*: a flag does not imply the source
mocap was wrong. Plane-only terrain.

## Citation / provenance

Version v0.1.0 = the exact screen used in the companion note and the CLIMB flagship
(`screen.py` derived from `n1_knee_id.py`, sha256 of the original pinned in the project's
`GLOBAL_EVAL_ADDENDUM.md`: `94bc3f65…`). Apache-2.0.
