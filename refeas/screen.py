#!/usr/bin/env python3
"""refeas.screen -- dynamic-feasibility screen for retargeted humanoid references (CPU, MuJoCo).

Is the reference *dynamically* feasible for the G1 given the contacts it can
make? Per frame:
  1. q, qd from the reference; qdd by central differences (smoothed);
  2. contact-free inverse dynamics (mj_inverse with contacts disabled) ->
     the generalised force the environment must supply: a 6-D base wrench W
     that only contact can produce, and the joint torques tau_free;
  3. candidate contact points = collision geoms within `--gap` of the plane;
     find contact forces in friction cones (pyramidal, mu per geom class) that
     best explain W  -> unsupported wrench residual r = ||W - A f||;
  4. joint torques with those contacts: tau = tau_free - J_c^T f, compared to
     the actuator force ranges -> saturation fraction and worst joints.

Two contact models: `real` (mu 0.6 everywhere) and `sim` (feet 0.6, non-foot
geoms frictionless, as mjlab's G1 model has condim=1 there).

Usage:
    n1_knee_id.py --clip BMLmovi_..._jpos --t0 0 --t1 3.5 --out reports/N1_clip44.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mujoco
import numpy as np
from scipy.optimize import nnls



def latest_model_xml():
    raise SystemExit("pass --model <robot.xml>")


def quat_to_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def torque_limited_lp(A, W, Jj, tau_free, frange):
    """min sum|A f - W| s.t. -frange <= tau_free - Jj f <= frange, f >= 0. Returns (f, resid) or None if infeasible."""
    from scipy.optimize import linprog
    n = A.shape[1]; k = A.shape[0]
    # variables: f (n), s (k)  ; minimise sum s
    c = np.concatenate([np.zeros(n), np.ones(k)])
    # A f - W <= s  ->  A f - s <= W ;  -(A f - W) <= s -> -A f - s <= -W
    A_ub = [np.hstack([A, -np.eye(k)]), np.hstack([-A, -np.eye(k)])]
    b_ub = [W, -W]
    # tau_free - Jj f <= frange  -> -Jj f <= frange - tau_free ; -(tau_free - Jj f) <= frange -> Jj f <= frange + tau_free
    A_ub.append(np.hstack([-Jj, np.zeros((Jj.shape[0], k))])); b_ub.append(frange - tau_free)
    A_ub.append(np.hstack([Jj, np.zeros((Jj.shape[0], k))])); b_ub.append(frange + tau_free)
    res = linprog(c, A_ub=np.vstack(A_ub), b_ub=np.concatenate(b_ub), bounds=[(0, None)] * n + [(0, None)] * k, method="highs")
    if not res.success:
        return None
    f = res.x[:n]
    return f, W - A @ f


def smooth(x, k=5):
    if k <= 1:
        return x
    ker = np.ones(k) / k
    return np.stack([np.convolve(x[:, i], ker, mode="same") for i in range(x.shape[1])], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, help="clip name (basename without .npz)")
    ap.add_argument("--bank", required=True, help="directory of reference .npz files")
    ap.add_argument("--model", required=True, help="MuJoCo robot XML with a plane geom named 'terrain' (or last plane)")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=3.5)
    ap.add_argument("--gap", type=float, default=0.03, help="contact candidate distance to plane [m]")
    ap.add_argument("--mu", type=float, default=0.6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--brief", action="store_true", help="write only clip-level feasibility features (atlas v2.1), no frames")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT   # contact-free ID
    dat = np.load(os.path.join(args.bank, args.clip + ".npz"))
    fps = float(np.asarray(dat["fps"]).reshape(-1)[0])
    jp, jv = dat["joint_pos"], dat["joint_vel"]
    bp, bq, blv, bav = dat["body_pos_w"], dat["body_quat_w"], dat["body_lin_vel_w"], dat["body_ang_vel_w"]
    T = jp.shape[0]
    t = np.arange(T) / fps
    # qpos / qvel in MuJoCo layout
    qpos = np.zeros((T, m.nq)); qvel = np.zeros((T, m.nv))
    qpos[:, 0:3] = bp[:, 0]; qpos[:, 3:7] = bq[:, 0]; qpos[:, 7:] = jp
    for i in range(T):
        R = quat_to_mat(bq[i, 0])
        qvel[i, 0:3] = blv[i, 0]
        qvel[i, 3:6] = R.T @ bav[i, 0]        # body-frame angular velocity
    qvel[:, 6:] = jv
    qvel_s = smooth(qvel, 5)
    qacc = np.gradient(qvel_s, 1.0 / fps, axis=0)
    qacc = smooth(qacc, 5)

    gname = lambda i: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").split("/")[-1]
    coll = [i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]) and gname(i) != "terrain"]
    plane = [i for i in range(m.ngeom) if gname(i) == "terrain"][0]
    is_foot = {i: ("foot" in gname(i)) for i in coll}
    frange = m.actuator_forcerange[:, 1].copy()
    act_dof = np.array([m.jnt_dofadr[m.actuator_trnid[i, 0]] for i in range(m.nu)]) - 6   # joint rows only
    aname = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").split("/")[-1] for i in range(m.nu)]

    # pyramidal basis directions (4 edges) for a normal n = +z
    def cone_basis(mu):
        n = np.array([0, 0, 1.0]); tx = np.array([1.0, 0, 0]); ty = np.array([0, 1.0, 0])
        if mu <= 0:
            return [n]
        return [n + mu * tx, n - mu * tx, n + mu * ty, n - mu * ty]

    args.t1 = min(args.t1, float(t[-1]) + 1e-6)
    rows = []
    for k in range(T):
        if t[k] < args.t0 or t[k] > args.t1:
            continue
        d.qpos[:] = qpos[k]; d.qvel[:] = qvel_s[k]; d.qacc[:] = qacc[k]
        mujoco.mj_inverse(m, d)
        W = d.qfrc_inverse[:6].copy()          # base wrench the environment must supply
        tau_free = d.qfrc_inverse[6:].copy()
        # candidate contacts
        mujoco.mj_forward(m, d)  # kinematics for jacobians / distances (contacts disabled anyway)
        cands = []
        for g in coll:
            fromto = np.zeros(6)
            dist = mujoco.mj_geomDistance(m, d, g, plane, args.gap + 0.05, fromto)
            if dist <= args.gap:
                p = fromto[:3]   # nearest point on geom g (from geom1 to geom2)
                cands.append((g, dist, p))
        out = {"t": round(float(t[k]), 3), "W_force": W[:3].round(2).tolist(), "W_torque_body": W[3:6].round(2).tolist(),
               "n_contacts": len(cands), "contacts": [(gname(g), round(float(dist), 4)) for g, dist, _ in cands],
               "root_z": round(float(qpos[k, 2]), 3)}
        for model_name in ("real", "sim"):
            cols = []; jac_cols = []
            for g, dist, p in cands:
                mu = args.mu if (model_name == "real" or is_foot[g]) else 0.0
                jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jacp, jacr, p, m.geom_bodyid[g])
                for e in cone_basis(mu):
                    # generalised force from a unit force e at p: J^T e -> base rows give W contribution, joint rows give tau
                    gf = jacp.T @ e
                    cols.append(gf[:6]); jac_cols.append(gf[6:])
            if cols:
                A = np.stack(cols, axis=1)  # (6, ncols)
                Jj = np.stack(jac_cols, axis=1)  # (nv-6, ncols)
                # weight torque rows so N and N.m are comparable (~0.5 m lever)
                wgt = np.array([1, 1, 1, 2, 2, 2.0])
                f, _ = nnls(A * wgt[:, None], W * wgt)
                resid = W - A @ f
                tau = tau_free - Jj @ f
                # torque-limited LP: min L1 wrench residual s.t. |tau_free - Jj f| <= frange on actuated dofs, f >= 0
                lp = torque_limited_lp(A * wgt[:, None], W * wgt, Jj[act_dof], tau_free[act_dof], frange)
                if lp is not None:
                    f2, resid2 = lp
                    resid2 = resid2 / wgt
                    tau2 = tau_free - Jj @ f2
                else:
                    f2 = None; resid2 = None; tau2 = None
            else:
                f = np.zeros(0); resid = W.copy(); tau = tau_free.copy(); f2 = None; resid2 = None; tau2 = None
            sat = np.abs(tau[act_dof]) / frange
            out[model_name] = {"unsupported_force_N": round(float(np.linalg.norm(resid[:3])), 1),
                               "unsupported_torque_Nm": round(float(np.linalg.norm(resid[3:6])), 1),
                               "max_tau_ratio": round(float(sat.max()), 2),
                               "n_over_limit": int((sat > 1.0).sum()),
                               "worst": [(aname[i], round(float(sat[i]), 2)) for i in np.argsort(-sat)[:3]],
                               "contact_normal_force_N": round(float(f.sum()), 1) if f.size else 0.0,
                               # torque-limited: the smallest unsupported wrench achievable within actuator limits
                               "tl_unsupported_force_N": (round(float(np.linalg.norm(resid2[:3])), 1) if resid2 is not None else None),
                               "tl_unsupported_torque_Nm": (round(float(np.linalg.norm(resid2[3:6])), 1) if resid2 is not None else None)}
        rows.append(out)

    # summary by 0.25 s bins
    def binsum(model_name):
        res = []
        for lo in np.arange(args.t0, args.t1, 0.25):
            sel = [r for r in rows if lo <= r["t"] < lo + 0.25]
            if not sel:
                continue
            res.append({"t": f"{lo:.2f}-{lo+0.25:.2f}",
                        "root_z": round(float(np.mean([r["root_z"] for r in sel])), 2),
                        "contacts": sorted({c for r in sel for c, _ in r["contacts"]}),
                        "unsup_F_med": round(float(np.median([r[model_name]["unsupported_force_N"] for r in sel])), 1),
                        "unsup_F_max": round(float(np.max([r[model_name]["unsupported_force_N"] for r in sel])), 1),
                        "unsup_M_med": round(float(np.median([r[model_name]["unsupported_torque_Nm"] for r in sel])), 1),
                        "tau_ratio_p50": round(float(np.median([r[model_name]["max_tau_ratio"] for r in sel])), 2),
                        "tau_ratio_max": round(float(np.max([r[model_name]["max_tau_ratio"] for r in sel])), 2),
                        "frames_over_limit": int(sum(r[model_name]["n_over_limit"] > 0 for r in sel)),
                        "tl_unsup_F_med": (round(float(np.median([r[model_name]["tl_unsupported_force_N"] for r in sel if r[model_name]["tl_unsupported_force_N"] is not None])), 1) if any(r[model_name]["tl_unsupported_force_N"] is not None for r in sel) else None),
                        "tl_infeasible_frames": int(sum(r[model_name]["tl_unsupported_force_N"] is None and r["n_contacts"] > 0 for r in sel)),
                        "worst": max(sel, key=lambda r: r[model_name]["max_tau_ratio"])[model_name]["worst"][:2]})
        return res
    summary = {"clip": args.clip, "fps": fps, "gap": args.gap, "mu": args.mu, "model": args.model,
               "total_mass_kg": float(m.body_mass.sum()), "bins": {mn: binsum(mn) for mn in ("real", "sim")}, "frames": rows}
    for mn in ([] if args.brief else ("real", "sim")):
        print(f"\n== contact model: {mn} ==")
        print(f"{'t':11s} {'z':>5s} {'unsupF med/max [N]':>20s} {'unsupM med':>10s} {'tau/lim p50/max':>16s} {'>lim':>4s} {'TL-unsupF':>9s} {'TL-inf':>6s}  contacts | worst")
        for b in summary["bins"][mn]:
            print(f"{b['t']:11s} {b['root_z']:5.2f} {b['unsup_F_med']:9.1f}/{b['unsup_F_max']:<9.1f} {b['unsup_M_med']:10.1f} {b['tau_ratio_p50']:7.2f}/{b['tau_ratio_max']:<7.2f} {b['frames_over_limit']:4d} {str(b['tl_unsup_F_med']):>9s} {b['tl_infeasible_frames']:6d}  {','.join(c.replace('_collision','') for c in b['contacts'])} | {b['worst']}")
    print(f"\ntotal mass {summary['total_mass_kg']:.1f} kg (weight {summary['total_mass_kg']*9.81:.0f} N)")
    if args.out:
        if args.brief:
            W = summary["total_mass_kg"] * 9.81
            fr = rows
            nc = np.array([x["n_contacts"] == 0 for x in fr])
            uf = np.array([x["real"]["unsupported_force_N"] for x in fr])
            tl = np.array([x["real"]["tl_unsupported_force_N"] if x["real"]["tl_unsupported_force_N"] is not None else (x["real"]["unsupported_force_N"] if x["n_contacts"] == 0 else np.nan) for x in fr])
            tl0 = np.nan_to_num(tl, nan=0.0)
            brief = {"clip": args.clip, "frames": len(fr), "fps": fps, "gap": args.gap,
                     "airborne_frac": float(nc.mean()),                       # no contact candidate within gap
                     "infeasible_frac": float((tl0 > 0.5 * W).mean()),        # unsupported > half weight within torque limits
                     "unsupported_impulse_Ns": float((tl0 / fps).sum()),      # unsupported force x duration
                     "unsupported_impulse_per_weight_s": float((tl0 / W / fps).sum()),
                     "torque_infeasible_frac": float(np.mean([x["real"]["tl_unsupported_force_N"] is None and x["n_contacts"] > 0 for x in fr])),
                     "max_tau_ratio_p95": float(np.percentile([x["real"]["max_tau_ratio"] for x in fr], 95)),
                     "sim_infeasible_frac": float((np.nan_to_num(np.array([x["sim"]["tl_unsupported_force_N"] if x["sim"]["tl_unsupported_force_N"] is not None else (x["sim"]["unsupported_force_N"] if x["n_contacts"] == 0 else np.nan) for x in fr]), nan=0.0) > 0.5 * W).mean())}
            json.dump(brief, open(args.out, "w"))
        else:
            json.dump(summary, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
