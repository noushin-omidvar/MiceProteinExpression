#!/usr/bin/env python3
"""Run every mechanistic analysis and write tables plus a machine-readable summary.

Analyses (numbered as in the paper):
  0. Measurement QC: duplicate columns, technical-block structure, missingness.
  1. Gene dosage of triplicated proteins (discovery in saline, replication in memantine).
  2. Learning signature: does trisomy blunt the conditioning (CS - SC) response,
     and does memantine restore it?
  3. Rescue map: how far does memantine move Ts65Dn profiles back toward control?
  4. Pathway activity from phospho/total ratios.
  5. Network rewiring between genotypes.
  6. Power for a follow-up experiment.

Outputs go to ``results/mechanisms/`` (see the README there). The paper's numbers are
read from ``summary.json`` so text and results cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml

from mice_protein import contrasts as C
from mice_protein import signatures as S
from mice_protein.annotations import EXPECTED_DOSAGE_LOG2FC, HSA21_DISOMIC_TS65DN, TRISOMIC_TS65DN
from mice_protein.data import (
    CELLS,
    analysis_panel,
    duplicate_columns,
    load_data,
    log2_animal_matrix,
)
from mice_protein.dosage import dosage_table, set_mean_bootstrap
from mice_protein.networks import rewiring_test
from mice_protein.pathways import pathway_coherence, pathway_scores, phospho_ratios
from mice_protein.power import contrast_power, required_n
from mice_protein.qc import block_profile, missingness_by_cell, variance_components
from mice_protein.statistics import benjamini_hochberg

ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def interval(draws: np.ndarray) -> list[float]:
    """Percentile 95% interval, ignoring non-finite bootstrap draws."""
    finite = draws[np.isfinite(draws)]
    return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


def perm_p(null: np.ndarray, observed: float) -> float:
    """One-sided permutation p-value with the +1 correction (null >= observed)."""
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mechanisms.yaml")
    parser.add_argument("--output-dir", type=Path, help="Override the configured output directory")
    parser.add_argument("--fast", action="store_true",
                        help="Few resamples, for smoke tests only (results are not for reporting)")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.fast:
        cfg.update(n_bootstrap=50, n_permutations=50, n_network_permutations=20)
    out = args.output_dir or ROOT / cfg["output_dir"]
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    summary: dict = {}

    raw = load_data(ROOT / cfg["raw_data"])
    log2_values, labels = log2_animal_matrix(raw)
    cells = labels["class"]
    codes = cells.map({c: k for k, c in enumerate(CELLS)}).to_numpy()
    duplicates = duplicate_columns(raw)
    unique = [c for c in log2_values.columns if c not in duplicates]
    panel = analysis_panel(raw)
    z = S.standardize(log2_values[panel], cells)
    summary["data"] = {
        "rows": len(raw), "mice": len(labels), "proteins": log2_values.shape[1],
        "unique_proteins": len(unique), "panel_proteins": len(panel),
        "duplicates": duplicates,
        "excluded_sparse": sorted(set(unique) - set(panel)),
        "cell_n": cells.value_counts().reindex(CELLS).astype(int).to_dict(),
        "nonpositive_values": int((raw[log2_values.columns] <= 0).sum().sum()),
        "imputed_mouse_means": int(log2_values[panel].isna().sum().sum()),
    }

    # 0. QC -----------------------------------------------------------------
    log("QC")
    components = variance_components(raw)
    components.to_csv(tables / "qc_variance_components.csv", index=False)
    block_profile(raw).to_csv(tables / "qc_block_profile.csv")
    missing = missingness_by_cell(raw)
    missing["q"] = benjamini_hochberg(missing.p.fillna(1).to_numpy())
    missing.to_csv(tables / "qc_missingness_by_cell.csv", index=False)
    comp_unique = components[components.protein.isin(unique)]
    summary["qc"] = {
        "icc_median": float(comp_unique.icc.median()),
        "icc_iqr": [float(comp_unique.icc.quantile(.25)), float(comp_unique.icc.quantile(.75))],
        "block_share_median": float(comp_unique.block_share.median()),
        "block_share_iqr": [float(comp_unique.block_share.quantile(.25)),
                            float(comp_unique.block_share.quantile(.75))],
        "missing_informative": missing.loc[missing.q < cfg["fdr"], "protein"].tolist(),
        "missing_tested": len(missing),
    }

    # Protein-level contrasts (all unique proteins, log2 scale) ---------------
    log("Protein contrasts")
    protein_contrasts = C.add_fdr(
        C.contrast_table(log2_values[unique], cells, C.FACTORIAL | C.MECHANISTIC), by="contrast"
    )
    protein_contrasts.to_csv(tables / "protein_contrasts_log2.csv", index=False)
    summary["protein_contrasts"] = (
        protein_contrasts[protein_contrasts.q < cfg["fdr"]].groupby("contrast").size()
        .reindex(list(C.FACTORIAL) + list(C.MECHANISTIC), fill_value=0).astype(int).to_dict()
    )

    # 1. Gene dosage --------------------------------------------------------
    log("Gene dosage")
    dosage = dosage_table(log2_values[unique], cells)
    dosage["q_saline"] = benjamini_hochberg(dosage.p_saline.fillna(1).to_numpy())
    dosage.to_csv(tables / "dosage_fold_changes.csv", index=False)
    background = [p for p in panel if p not in TRISOMIC_TS65DN + HSA21_DISOMIC_TS65DN]
    sets = {
        "triplicated": list(TRISOMIC_TS65DN),
        "hsa21_disomic": list(HSA21_DISOMIC_TS65DN),
        "background": background,
    }
    set_effects = {
        name: set_mean_bootstrap(log2_values, cells, members, cfg["n_bootstrap"], rng)
        | {"n": len(members)}
        for name, members in sets.items()
    }
    observed = dosage.dropna(subset=["estimate_saline", "estimate_memantine"])
    tri = dosage.set_index("protein").loc[list(TRISOMIC_TS65DN)]
    bg_abs = dosage.set_index("protein").loc[background, "estimate_saline"].abs()
    summary["dosage"] = {
        "expected_log2fc": EXPECTED_DOSAGE_LOG2FC,
        "sets": set_effects,
        "set_fraction_of_expected": set_effects["triplicated"]["estimate"] / EXPECTED_DOSAGE_LOG2FC,
        "replication_pearson": float(observed.estimate_saline.corr(observed.estimate_memantine)),
        "replication_pearson_triplicated": float(tri.estimate_saline.corr(tri.estimate_memantine)),
        "triplicated_up_both_arms": int(((tri.lower_saline > 0) & (tri.lower_memantine > 0)).sum()),
        "triplicated_q05_saline": int((tri.q_saline < cfg["fdr"]).sum()),
        "triplicated_rank_percentile": {
            p: float((bg_abs < abs(tri.loc[p, "estimate_saline"])).mean()) for p in tri.index
        },
        "per_protein": tri[["estimate_saline", "lower_saline", "upper_saline",
                            "estimate_memantine", "q_saline"]].to_dict(orient="index"),
    }

    # 2. Learning signature ---------------------------------------------------
    log("Learning signature")
    learning = S.learning_analysis(z, codes)
    boot = {k: [] for k in ("relative", "response", "cosine", "loadings")}
    for _ in range(cfg["n_bootstrap"]):
        idx, copies = S.stratified_bootstrap(codes, rng)
        res = S.learning_analysis(z.iloc[idx].reset_index(drop=True), codes[idx], copies)
        boot["relative"].append(res.relative.to_numpy())
        boot["response"].append(res.response.to_numpy())
        boot["cosine"].append(res.cosine.to_numpy())
        boot["loadings"].append(res.loadings.to_numpy())
    arms = list(learning.response.index)
    relative_draws = np.array(boot["relative"])
    response_draws = np.array(boot["response"])

    def learning_null(factor: int, within: np.ndarray, statistic) -> np.ndarray:
        null = np.empty(cfg["n_permutations"])
        for b in range(cfg["n_permutations"]):
            permuted = S.permute_factor(codes, factor, within, rng)
            null[b] = statistic(S.learning_analysis(z, permuted))
        return null

    saline = np.isin(codes, [CELLS.index(c) for c in CELLS if c.endswith("-s")])
    trisomic = codes >= 4
    attenuation = lambda r: r.response["control saline"] - r.response["Ts65Dn saline"]
    restoration = lambda r: r.response["Ts65Dn memantine"] - r.response["Ts65Dn saline"]
    null_att = learning_null(0, saline, attenuation)
    null_res = learning_null(2, trisomic, restoration)
    loadings_draws = np.array(boot["loadings"])
    loading_table = pd.DataFrame(
        {
            "protein": panel,
            "loading": learning.loadings.to_numpy(),
            "lower": np.percentile(loadings_draws, 2.5, axis=0),
            "upper": np.percentile(loadings_draws, 97.5, axis=0),
            "sign_stability": np.mean(
                np.sign(loadings_draws) == np.sign(learning.loadings.to_numpy()), axis=0
            ),
        }
    ).sort_values("loading", key=np.abs, ascending=False)
    loading_table.to_csv(tables / "learning_axis_loadings.csv", index=False)
    pd.DataFrame({"mouse": z.index, "cell": cells.to_numpy(), "learning_score": learning.scores}
                 ).to_csv(tables / "learning_scores.csv", index=False)
    summary["learning"] = {
        "response": learning.response.to_dict(),
        "response_ci": {a: interval(response_draws[:, i]) for i, a in enumerate(arms)},
        "relative": learning.relative.to_dict(),
        "relative_ci": {a: interval(relative_draws[:, i]) for i, a in enumerate(arms)},
        "cosine": learning.cosine.to_dict(),
        "cosine_ci": {a: interval(np.array(boot["cosine"])[:, i])
                      for i, a in enumerate(learning.cosine.index)},
        "attenuation": float(attenuation(learning)),
        "attenuation_p": perm_p(null_att, attenuation(learning)),
        "restoration": float(restoration(learning)),
        "restoration_p": perm_p(null_res, restoration(learning)),
        "within_cell_sd": float(pd.Series(learning.scores).groupby(codes).std(ddof=1).pow(2)
                                .mean() ** 0.5),
        "top_loadings": loading_table.head(10).set_index("protein").loading.to_dict(),
    }

    # 3. Rescue -----------------------------------------------------------
    log("Rescue")
    rescue = S.rescue_analysis(z, codes)
    draws = {k: [] for k in ("rescue", "persistence", "control_shift", "by_cs", "by_sc", "share",
                             "arm_means")}
    for _ in range(cfg["n_bootstrap"]):
        idx, copies = S.stratified_bootstrap(codes, rng)
        res = S.rescue_analysis(z.iloc[idx].reset_index(drop=True), codes[idx], copies)
        draws["rescue"].append(res.rescue_fraction)
        draws["persistence"].append(res.persistence)
        draws["control_shift"].append(res.control_shift)
        draws["by_cs"].append(res.rescue_by_conditioning["CS"])
        draws["by_sc"].append(res.rescue_by_conditioning["SC"])
        draws["share"].append(res.along_axis_share)
        draws["arm_means"].append(res.arm_means.to_numpy())
    drug_shift = lambda r: r.arm_means["Ts65Dn saline"] - r.arm_means["Ts65Dn memantine"]
    null_rescue = np.array([
        drug_shift(S.rescue_analysis(z, S.permute_factor(codes, 2, trisomic, rng)))
        for _ in range(cfg["n_permutations"])
    ])
    arm_draws = np.array(draws["arm_means"])
    pd.DataFrame({"mouse": z.index, "cell": cells.to_numpy(), "trisomy_score": rescue.scores}
                 ).to_csv(tables / "trisomy_axis_scores.csv", index=False)
    rescue.loadings.rename("loading").sort_values(key=np.abs, ascending=False).to_csv(
        tables / "trisomy_axis_loadings.csv")

    # Per-protein persistence of the trisomy effect under memantine (disjoint mice).
    per = dosage[["protein", "estimate_saline", "lower_saline", "upper_saline", "p_saline",
                  "q_saline", "estimate_memantine", "lower_memantine", "upper_memantine",
                  "p_memantine"]].copy()
    se_saline = (per.upper_saline - per.lower_saline) / (2 * 1.96)
    selected = per.q_saline < cfg["fdr"]
    per["q_memantine_selected"] = np.nan
    per.loc[selected, "q_memantine_selected"] = benjamini_hochberg(
        per.loc[selected, "p_memantine"].to_numpy())
    same_sign = np.sign(per.estimate_memantine) == np.sign(per.estimate_saline)
    per["category"] = np.where(
        ~selected, "Not affected (saline)",
        np.where((per.q_memantine_selected < cfg["fdr"]) & same_sign, "Persists under memantine",
                 "Not detected under memantine"))
    per.to_csv(tables / "rescue_per_protein.csv", index=False)
    slope = np.polyfit(per.estimate_saline, per.estimate_memantine, 1)[0]
    reliability = 1 - np.mean(se_saline**2) / per.estimate_saline.var()
    summary["rescue"] = {
        "arm_means": rescue.arm_means.to_dict(),
        "arm_means_ci": {a: interval(arm_draws[:, i]) for i, a in enumerate(rescue.arm_means.index)},
        "rescue_fraction": float(rescue.rescue_fraction),
        "rescue_fraction_ci": interval(np.array(draws["rescue"])),
        "rescue_shift": float(drug_shift(rescue)),
        "rescue_shift_p": perm_p(null_rescue, drug_shift(rescue)),
        "persistence": float(rescue.persistence),
        "persistence_ci": interval(np.array(draws["persistence"])),
        "control_shift": float(rescue.control_shift),
        "control_shift_ci": interval(np.array(draws["control_shift"])),
        "rescue_by_conditioning": rescue.rescue_by_conditioning.to_dict(),
        "rescue_by_conditioning_ci": {"CS": interval(np.array(draws["by_cs"])),
                                      "SC": interval(np.array(draws["by_sc"]))},
        "magnitudes": rescue.magnitudes.to_dict(),
        "along_axis_share": rescue.along_axis_share,
        "along_axis_share_ci": interval(np.array(draws["share"])),
        "per_protein_counts": per.category.value_counts().to_dict(),
        "persistence_slope_naive": float(slope),
        "reliability_saline": float(reliability),
        "persistence_slope_corrected": float(slope / reliability),
        "top_loadings": rescue.loadings.sort_values(key=np.abs, ascending=False).head(10).to_dict(),
    }

    # 4. Pathways ---------------------------------------------------------
    log("Pathways")
    ratios = phospho_ratios(log2_values)
    scores = pathway_scores(ratios, cells)
    ratio_tab = C.add_fdr(C.contrast_table(ratios, cells, C.FACTORIAL), by="contrast")
    path_tab = C.add_fdr(C.contrast_table(scores, cells, C.FACTORIAL | C.MECHANISTIC),
                         by="contrast")
    ratio_tab.to_csv(tables / "phospho_ratio_contrasts.csv", index=False)
    path_tab.to_csv(tables / "pathway_contrasts.csv", index=False)
    scores.assign(cell=cells).to_csv(tables / "pathway_scores.csv")
    coherence = pathway_coherence(ratios, cells)
    summary["pathways"] = {
        "coherence": coherence.to_dict(),
        "significant": path_tab.loc[path_tab.q < cfg["fdr"],
                                    ["feature", "contrast", "estimate", "q"]].to_dict("records"),
        "ratio_significant": ratio_tab.loc[ratio_tab.q < cfg["fdr"],
                                           ["feature", "contrast", "estimate", "q"]]
        .to_dict("records"),
    }

    # 5. Network rewiring -------------------------------------------------
    log("Network rewiring")
    net = rewiring_test(z, codes, cfg["n_network_permutations"], rng)
    net["pairs"]["q"] = benjamini_hochberg(net["pairs"].p.to_numpy())
    net["pairs"].sort_values("p").to_csv(tables / "network_pairs.csv", index=False)
    net["control"].to_csv(tables / "network_corr_control.csv")
    net["trisomy"].to_csv(tables / "network_corr_ts65dn.csv")
    pd.Series(net["null"], name="null_statistic").to_csv(tables / "network_null.csv", index=False)
    summary["network"] = {
        "statistic": net["statistic"],
        "null_mean": float(net["null"].mean()),
        "null_95": float(np.percentile(net["null"], 95)),
        "p_global": float(net["p_global"]),
        "pairs_q05": int((net["pairs"].q < cfg["fdr"]).sum()),
        "pairs_total": len(net["pairs"]),
    }

    # 6. Power ------------------------------------------------------------
    log("Power")
    sd = summary["learning"]["within_cell_sd"]
    interaction = S.weight_vector({"c-CS-s": 1, "c-SC-s": -1, "t-CS-s": -1, "t-SC-s": 1})
    three_way = S.weight_vector(C.FACTORIAL["Genotype×Conditioning×Treatment"])
    scenarios = {
        "Learning attenuation (observed)": (summary["learning"]["attenuation"], sd, interaction),
        "Learning attenuation (50% of control response)":
            (0.5 * summary["learning"]["response"]["control saline"], sd, interaction),
        "Protein 3-way interaction, 1 SD": (1.0, 1.0, three_way),
        "Protein 3-way interaction, 2 SD": (2.0, 1.0, three_way),
    }
    power_rows = [
        {"scenario": name, "n_per_cell": n, "power": contrast_power(eff, s, w, n, cfg["alpha"])}
        for name, (eff, s, w) in scenarios.items() for n in cfg["power_n_grid"]
    ]
    pd.DataFrame(power_rows).to_csv(tables / "power_curves.csv", index=False)
    summary["power"] = {
        name: {"effect": eff, "sd": s,
               "power_at_9": contrast_power(eff, s, w, 9, cfg["alpha"]),
               "n_for_80": required_n(eff, s, w, cfg["power_target"], cfg["alpha"])}
        for name, (eff, s, w) in scenarios.items()
    }

    summary["provenance"] = {
        "config": cfg, "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"Wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
