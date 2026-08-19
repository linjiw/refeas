# Appendix: an error-class checklist for coupling a second physics stack to an RL harness

*Status: measured (each S1 instance carries its artifact path); the checklist itself is method.
Ships identically in `refeas/docs/COUPLING_TAXONOMY.md`. Detection instruments referenced:
**paired substeps** (step both stacks from identical (q, q̇, ctrl, warm-start), diff everything),
**referee** (a third independent implementation stepped from the same pre-state), **shadow
solver** (stack A re-integrating stack B's own trajectory substep-by-substep during B's rollout).
S1 instances: `plan/S1_RESULT.md` errors #8–11; conformance after fixes |Δq̇| ≤ 3×10⁻⁵
(`reports/S1_KIT1226_n32_absorb.json`).*

End-of-episode metrics are the wrong test: in every instance below, mean tracking error matched
to ≤ 1 mm while survival forked by up to 40 points. Certify at the substep level or not at all.

| # | error class | symptom | best detector | S1 instance |
|---|---|---|---|---|
| 1 | **State-sync bidirectionality** — the harness writes robot state outside the stepper (episode resets, curriculum teleports, pushes) and only one stack sees the writes | divergence at protocol events (clip wraps, resets), not at contacts; forked stack "chases" a jumped reference | **shadow solver** (matches everywhere except the write instants) + counting absorbed writes | **#11**: clip-wrap teleport at 6.0 s overwritten by the coupled stack's own physics; 57–59 external writes per 32-env rollout. Fix: absorb env-side qpos/qvel writes each substep |
| 2 | **Domain-randomisation mirroring** — per-env DR mutates the live model of one stack only; anything built from the spec is the nominal robot | constant per-env bias in generalized forces from *identical* states; static test (q̇=0) shows it | paired substeps at q̇ = 0 (gravity-only bias forces) | **#8**: torso-CoM/foot-μ DR lived only in `env.sim.wp_model`; 3.3 N·m base gravity-torque residual ≈ 1 cm CoM shift. Fix: mirror expanded fields by name + recompute derived constants (residual → 3×10⁻⁶) |
| 3 | **Observation freshness / ordering** — obs computed before a teleport, or derived quantities stale by one substep relative to the other stack | first action of each episode is a "random kick"; |Δobs| large while |Δq| = 0 | diff obs at t=0 with identical states | **#10**: `assign_clips` teleported after `reset()` computed obs; |Δobs| 3.4–4.5 with Δq₀ = 0. Fix: recompute obs post-teleport |
| 4 | **Precision / dtype parity** — one import path rounds through float32 where the other keeps float64; hard branch conditions (contact iff dist < 0) amplify 1e-7 into discrete events | identical dynamics but occasional extra/missing *contacts* at grazing configurations; forks localized to lightly-loaded geometry | paired substeps + **referee** (referee agrees with the float64 side); contact-set diffs | **#9**: MJCF→float32-transform import left geometry 1e-7 off; one extra frictional foot contact at dist −0.0 forked q̇ by 0.05–3.3 rad/s. Fix: copy the reference stack's exact float32 geometry; 0 mismatches over 300 steps × 8 envs |
| 5 | **Reset semantics** — warm-starts, actuator activations, applied-force buffers surviving or not surviving reset differently per stack | first-step-after-reset divergence only; determinism check fails across repeats | repeat the same reset twice per stack; diff warm-start buffers | (guarded in S1: warm-start zeroed on both sides at rollout start; absorbed writes clear it — no live instance) |
| 6 | **RNG stream parity** — DR draws, noise, initial states drawn from different streams per arm; comparisons become unpaired | arm-to-arm variance ≫ seed-to-seed variance; "paired" designs silently unpaired | draw once, apply to both; assert bitwise-equal initial (q, q̇, obs) | (S1 lesson enforced, not a bug: replicate IC noise drawn once per replicate and shared across all paired worlds — `tools/g1_clip44_gate.py`) |
| 7 | **Contact-margin / geometry parity** — differing default gaps, margins, contact capacities (njmax), CCD settings between stacks | early/soft touchdowns, bouncing at rest, dropped constraint rows exactly at hard impacts | model-field diff (every geom/pair/opt field) + contact counts under load | pre-#8 finds: 0.1 m default `rigid_gap` on one stack (robot bounced); njmax 64 vs 250 silently dropping half the constraint rows at a multi-contact impact (`plan/S1_RESULT.md` fixes #5, and the capacity pin) |
| 8 | **Solver-parameter parity** — iterations, tolerances, integrator, cone, Jacobian layout, disable-flags differing from the *runtime* config (not the file default) | small per-substep force residuals that grow only under constraint pressure | diff the *live* option structs of both stacks at runtime, never the config files | pre-#8 find: XML carried file defaults (0.002 s, 100 iters, Euler) while the runtime used 0.005 s/implicitfast; MULTICCD/ccd_iterations pinned (`plan/S1_RESULT.md`) |
| 9 | **Actuation-path parity** — the coupled stack synthesizes its own actuators or reads a different command channel than the harness writes | policy "holds" toward zero or a constant pose; nu mismatch; torques zero while ctrl nonzero | count actuators; write a distinguishable per-env ctrl pattern and read it back post-step | pre-#8 finds: doubled actuators (nu 58) from setting target gains on top of imported ones; imported `<position>` actuators reading `mujoco.ctrl`, not `joint_target_q` (`plan/S1_RESULT.md` fixes #2–3) |

**Protocol implied by the table.** (1) Diff the live models field-by-field. (2) Paired substeps at
q̇ = 0, then under motion, then across the highest-contact window, with a third-implementation
referee. (3) Shadow-integrate the coupled stack's own rollout. (4) Only then read closed-loop
metrics — and require the *survival distribution*, not the mean error, to match. A harness passing
(1)–(3) but failing (4) has a class-1 (bidirectionality) problem until proven otherwise.

Version pinning is part of the certificate: this table's instances were earned against MuJoCo Warp
3.11.0 / mjlab v1.6.0 / Newton `7bb6d02d` / warp-lang 1.16.0 / MuJoCo 3.11.0, and class-4 and
class-7 in particular are expected to need re-checking across major engine releases (see the
Newton 1.0 re-certification spec, `plan/SPEC_newton_recert.md`).
