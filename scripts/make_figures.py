#!/usr/bin/env python
"""Generate the ten main figures (Part AE) from on-disk artifacts.

    python scripts/make_figures.py --ml-dir results/ml/resnet_seed42

Figure 10 (external validation) renders a documented status panel until
OASIS-3/ADNI data access is granted; every other figure is produced from
real artifacts in results/ and data/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainvuln.spatial import spatial_corr  # noqa: E402


def _fig(fig_dir: Path, name: str):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    return fig, ax, fig_dir / f"figure_{name}.png"


def fig1_architecture(fig_dir: Path) -> None:
    fig, ax, out = _fig(fig_dir, "1_architecture")
    ax.axis("off")
    boxes = [
        (0.02, 0.55, "OASIS-1 T1 MRI\n(sessions; subject-level splits)"),
        (0.20, 0.55, "Deterministic 3D preprocessing\nT88 masked → 2 mm → 128³ → z"),
        (0.38, 0.70, "3D ResNet-18\n(train; aug; AdamW; val-AUC stop)"),
        (0.38, 0.38, "Baselines\nage+sex · simple 3D CNN"),
        (0.56, 0.70, "Frozen test protocol\nROC/PR · calibration · bootstrap CI"),
        (0.56, 0.38, "Explainable AI\n3D Grad-CAM + occlusion"),
        (0.74, 0.38, "M_CNN(region)\n87-parcel atlas"),
        (0.74, 0.70, "GWAS genes + AHBA\nM_GENE(region)"),
        (0.90, 0.55, "Three-way test\nCNN↔Gene↔Disease\nmatched-gene + spatial nulls"),
    ]
    for x, y, t in boxes:
        ax.add_patch(plt.Rectangle((x, y - 0.07), 0.15, 0.16,
                                   fill=True, facecolor="#eef3fa",
                                   edgecolor="#2b5d8c", lw=1.2))
        ax.text(x + 0.075, y + 0.01, t, ha="center", va="center", fontsize=7.4)
    arrows = [(0.17, 0.61, 0.20, 0.61), (0.35, 0.61, 0.38, 0.74),
              (0.35, 0.61, 0.38, 0.44), (0.53, 0.74, 0.56, 0.74),
              (0.53, 0.44, 0.56, 0.44), (0.71, 0.74, 0.74, 0.74),
              (0.71, 0.44, 0.74, 0.44), (0.89, 0.74, 0.90, 0.62),
              (0.89, 0.44, 0.90, 0.55)]
    for x0, y0, x1, y1 in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#2b5d8c", lw=1.1))
    ax.text(0.5, 0.92, "BrainVuln system architecture",
            ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.10, "Model-derived relevance is an association probe — "
            "not causal proof, not a clinical diagnostic.",
            ha="center", fontsize=7.5, style="italic", color="#555")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2_performance(ml_dir: Path, fig_dir: Path) -> bool:
    pred_csv = ml_dir / "predictions" / "test_predictions.csv"
    if not pred_csv.exists():
        return False
    df = pd.read_csv(pred_csv)
    metrics = json.loads((ml_dir / "metrics" / "test_metrics.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    RocCurveDisplay.from_predictions(df.label, df.probability, ax=axes[0])
    axes[0].set_title(f"ROC (AUC {metrics['test']['roc_auc']:.3f})")
    PrecisionRecallDisplay.from_predictions(df.label, df.probability, ax=axes[1])
    axes[1].set_title(f"Precision–Recall (AP {metrics['test']['pr_auc']:.3f})")
    cm = metrics["test"]["confusion"]
    cm_mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    im = axes[2].imshow(cm_mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm_mat):
        axes[2].text(j, i, str(v), ha="center", va="center", fontsize=13)
    axes[2].set_xticks([0, 1], ["CN", "AD"]); axes[2].set_yticks([0, 1], ["CN", "AD"])
    axes[2].set_xlabel("predicted"); axes[2].set_ylabel("true")
    axes[2].set_title(f"Confusion (thr {metrics['test']['threshold']:.2f})")
    fig.suptitle("Held-out test performance (frozen model, subject-level "
                 f"bootstrap CI ROC [{metrics['test']['bootstrap_ci']['roc_auc']['lo']:.3f}, "
                 f"{metrics['test']['bootstrap_ci']['roc_auc']['hi']:.3f}])", fontsize=10)
    fig.savefig(fig_dir / "figure_2_performance.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    return True


def fig3_example_gradcam(ml_dir: Path, fig_dir: Path) -> bool:
    cams = sorted((ml_dir / "gradcam").glob("*_planes.png"))
    if not cams:
        return False
    src = cams[len(cams) // 2]  # a representative subject
    img = plt.imread(src)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"Example MRI + 3D Grad-CAM ({src.stem.replace('_planes', '')}) — "
                 "regions the classifier USED, not proof of pathology",
                 fontsize=9)
    fig.savefig(fig_dir / "figure_3_example_gradcam.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    return True


def fig4_m_cnn(ml_dir: Path, fig_dir: Path) -> bool:
    csv = ml_dir / "regional_relevance" / "M_CNN_regional.csv"
    if not csv.exists():
        return False
    df = pd.read_csv(csv, index_col=0)
    s = df["M_CNN"].dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(range(len(s)), s.values, color=np.where(s.values >= 0, "#2b5d8c", "#b0603f"))
    ax.set_xticks(range(0, len(s), 5), s.index[::5], rotation=60, fontsize=6)
    ax.set_ylabel("M_CNN (mean Grad-CAM per parcel)")
    ax.set_title(f"Model-derived regional relevance, population mean "
                 f"({len(s)} atlas parcels; full predefined atlas, no selection)")
    fig.savefig(fig_dir / "figure_4_M_CNN.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def _load_m_gene() -> pd.Series | None:
    from brainvuln import ahba
    from brainvuln.config import ensure_output_tree
    dirs = ensure_output_tree()
    expr, _, cov = ahba.load_expression(dirs["data_derived"], "dk68_tianS1")
    expr, _ = ahba.drop_unsampled_parcels(expr, cov)
    gs = Path("results/gene_sets/alzheimer_gwascat.json")
    if not gs.exists():
        return None
    genes = [g for g in json.loads(gs.read_text())["genes"]
             if g in expr.columns]
    from brainvuln.scoring import compute_score
    return compute_score(expr, genes, "mean_z", "genes", None)


def fig5_m_gene(m_gene: pd.Series, fig_dir: Path) -> bool:
    if m_gene is None:
        return False
    s = m_gene.dropna().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(range(len(s)), s.values, color="#3f7d4e")
    ax.set_xticks(range(0, len(s), 5), s.index[::5], rotation=60, fontsize=6)
    ax.set_ylabel("M_GENE (mean gene z)")
    ax.set_title("Regional expression of Alzheimer's GWAS-associated genes "
                 "(genetically implicated, not causal)")
    fig.savefig(fig_dir / "figure_5_M_GENE.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def fig6_scatter(m_cnn: pd.Series, m_gene: pd.Series, fig_dir: Path) -> bool:
    if m_cnn is None or m_gene is None:
        return False
    j = pd.concat([m_cnn.rename("cnn"), m_gene.rename("gene")], axis=1).dropna()
    rho = j["cnn"].corr(j["gene"], method="spearman")
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(j["gene"], j["cnn"], s=22, alpha=0.75, edgecolor="k", lw=0.3)
    z = np.polyfit(j["gene"], j["cnn"], 1)
    xs = np.linspace(j["gene"].min(), j["gene"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), "r--", lw=1.2)
    ax.set_xlabel("M_GENE (genetic expression map)")
    ax.set_ylabel("M_CNN (model-derived relevance)")
    ax.set_title(f"Core hypothesis: Spearman ρ = {rho:.3f} "
                 f"(n={len(j)} parcels)")
    fig.savefig(fig_dir / "figure_6_cnn_vs_gene.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    return True


def fig7_spatial_nulls(fig_dir: Path) -> bool:
    bridge = Path("results/bridge/bridge_results.json")
    null_npy = Path("results/bridge/cnn_gene_matched_null_r.npy")
    if not bridge.exists():
        return False
    res = json.loads(bridge.read_text())
    cg = res["cnn_gene"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    if null_npy.exists():
        nr = np.load(null_npy)
        axes[0].hist(nr, bins=40, color="#9db8d2")
        axes[0].axvline(cg["rho_observed"], color="red", lw=2,
                        label=f"observed ρ = {cg['rho_observed']:.3f}")
        axes[0].set_title(f"Matched-gene null (p = {cg['matched_null']['p']:.4f})")
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "run scripts/bridge_cnn_gene.py",
                     ha="center", transform=axes[0].transAxes)
    fams = cg.get("spatial_nulls", {})
    labels = []
    for i, (name, r) in enumerate(fams.items()):
        if "p_spatial" not in r or r.get("p_spatial") is None:
            continue
        lo = r.get("null_r_p2.5")
        hi = r.get("null_r_p97.5")
        if lo is not None and hi is not None:
            axes[1].plot([lo, hi], [i, i], color="#9db8d2", lw=6,
                         solid_capstyle="butt",
                         label="null 95% interval" if not labels else None)
        axes[1].plot(r["r_obs"], i, "D", color="red", ms=7,
                     label="observed ρ" if not labels else None)
        labels.append(f"{name} (p={r['p_spatial']:.4f})")
    axes[1].set_yticks(range(len(labels)), labels, fontsize=8)
    axes[1].set_xlabel("ρ with M_GENE (observed; bar = null 95% interval)")
    axes[1].set_title("Spatially constrained nulls")
    fig.savefig(fig_dir / "figure_7_spatial_nulls.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    return True


def fig8_cnn_vs_disease(fig_dir: Path) -> bool:
    bridge = Path("results/bridge/bridge_results.json")
    if not bridge.exists():
        return False
    tri = json.loads(bridge.read_text()).get("three_way", {})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    items = {k: v for k, v in tri.items() if k.startswith("CNN_vs")}
    if not items:
        return False
    ax.bar(range(len(items)), [v["rho"] for v in items.values()],
           color=["#2b5d8c", "#3f7d4e"][:len(items)])
    ax.set_xticks(range(len(items)),
                  [f"{k.split('_vs_')[1]}\np={v['p_spatial_moran']:.3f}"
                   for k, v in items.items()], fontsize=7)
    ax.set_ylabel("Spearman ρ (M_CNN vs disease map)")
    ax.set_title("Model relevance vs INDEPENDENT disease maps "
                 "(never used in training or gene definition)")
    fig.savefig(fig_dir / "figure_8_cnn_vs_disease.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)
    return True


def fig9_donor_loo(ml_dir: Path, fig_dir: Path) -> bool:
    """CNN↔Gene correlation recomputed with each donor left out (Part U)."""
    m_cnn_csv = ml_dir / "regional_relevance" / "M_CNN_regional.csv"
    if not m_cnn_csv.exists():
        return False
    from brainvuln import ahba
    from brainvuln.config import ensure_output_tree
    from brainvuln.scoring import compute_score
    dirs = ensure_output_tree()
    expr, donor_long, cov = ahba.load_expression(dirs["data_derived"],
                                                 "dk68_tianS1")
    expr, _ = ahba.drop_unsampled_parcels(expr, cov)
    gs = Path("results/gene_sets/alzheimer_gwascat.json")
    genes = [g for g in json.loads(gs.read_text())["genes"]
             if g in expr.columns]
    m_cnn = pd.read_csv(m_cnn_csv, index_col=0)["M_CNN"]
    donors = sorted(donor_long["donor"].unique()) if donor_long is not None else []
    if not donors:
        return False
    rhos = {}
    for d in donors:
        keep = donor_long[donor_long["donor"] != d]
        expr_d = (keep.pivot_table(index="label", columns="gene",
                                   values="value", aggfunc="mean"))
        m_gene_d = compute_score(expr_d, [g for g in genes
                                          if g in expr_d.columns],
                                 "mean_z", "genes", None)
        j = pd.concat([m_cnn.rename("a"), m_gene_d.rename("b")],
                      axis=1).dropna()
        rhos[d] = float(j["a"].corr(j["b"], method="spearman"))
    full = spatial_corr(*[s.to_numpy() for s in
                          [m_cnn.reindex(expr.index).dropna(),
                           _full_gene(expr, genes).reindex(expr.index).dropna()]])
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.bar(range(len(rhos)), list(rhos.values()), color="#7a5ea8")
    ax.axhline(full, color="red", ls="--", label=f"full AHBA ({full:.3f})")
    ax.set_xticks(range(len(rhos)), [f"without\n{d}" for d in rhos], fontsize=7)
    vals = np.array(list(rhos.values()))
    ax.set_title(f"Leave-one-donor-out: median {np.median(vals):.3f}, "
                 f"IQR {np.percentile(vals, 25):.3f}–{np.percentile(vals, 75):.3f}")
    ax.legend(fontsize=8)
    fig.savefig(fig_dir / "figure_9_donor_loo.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    dr = Path("results/donor_robustness")
    dr.mkdir(parents=True, exist_ok=True)
    (dr / "cnn_gene_loo.json").write_text(
        json.dumps({"per_donor": rhos, "full": full,
                    "median": float(np.median(vals)),
                    "iqr": [float(np.percentile(vals, 25)),
                            float(np.percentile(vals, 75))],
                    "min": float(vals.min()), "max": float(vals.max())},
                   indent=2))
    return True


def _full_gene(expr, genes) -> pd.Series:
    from brainvuln.scoring import compute_score
    return compute_score(expr, genes, "mean_z", "genes", None)


def fig10_external(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.axis("off")
    ax.text(0.5, 0.62, "External validation (OASIS-3 / ADNI)",
            ha="center", fontsize=11, weight="bold")
    ax.text(0.5, 0.38, "PENDING DATA ACCESS\n\n"
            "OASIS-3 and ADNI are registration-gated (DUA). The frozen-model\n"
            "validation pipeline is implemented (src/brainvuln/mri/external.py,\n"
            "scripts/stage_oasis3.py, scripts/stage_adni.py,\n"
            "scripts/validate_external.py) and runs as soon as data are staged.\n"
            "The OASIS-1 model will never be tuned on external data.",
            ha="center", fontsize=8, color="#444")
    fig.savefig(fig_dir / "figure_10_external_validation.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ml-dir", default="results/ml/resnet_seed42")
    ap.add_argument("--fig-dir", default="results/figures/manuscript")
    args = ap.parse_args(argv)
    ml_dir, fig_dir = Path(args.ml_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig1_architecture(fig_dir)
    ok = {
        "2": fig2_performance(ml_dir, fig_dir),
        "3": fig3_example_gradcam(ml_dir, fig_dir),
        "4": fig4_m_cnn(ml_dir, fig_dir),
    }
    m_gene = _load_m_gene()
    m_cnn_csv = ml_dir / "regional_relevance" / "M_CNN_regional.csv"
    m_cnn = (pd.read_csv(m_cnn_csv, index_col=0)["M_CNN"]
             if m_cnn_csv.exists() else None)
    ok["5"] = fig5_m_gene(m_gene, fig_dir)
    ok["6"] = fig6_scatter(m_cnn, m_gene, fig_dir)
    ok["7"] = fig7_spatial_nulls(fig_dir)
    ok["8"] = fig8_cnn_vs_disease(fig_dir)
    ok["9"] = fig9_donor_loo(ml_dir, fig_dir)
    fig10_external(fig_dir)
    for k, v in ok.items():
        print(f"figure {k}: {'written' if v else 'SKIPPED (artifacts pending)'}")
    print(f"figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    main()
