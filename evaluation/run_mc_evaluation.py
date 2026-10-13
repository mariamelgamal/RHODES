"""Monte Carlo evaluation of nominal vs robust RHODES designs on P95 total carbon.

For each distribution family, samples the uncertain carbon parameters, evaluates the
nominal and robust designs on the same samples, and reports their mean/P95 carbon. For
one family (--summary-distribution) it also summarizes the whole design space, counts
designs that violate the constraints, and reports time/carbon/tCDP ratios.

Usage (defaults come from evaluation/mc_evaluation/config.py):
    python evaluation/run_mc_evaluation.py
    python evaluation/run_mc_evaluation.py --workload bfs --nominal c3_g46_f765 --robust c2_g10_f765
    python evaluation/run_mc_evaluation.py --distributions Normal Uniform --verify
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from mc_evaluation.analysis import (compare_nominal_robust, find_design, tcdp_report, verify_scenarios,
                                    violates_constraints)
from mc_evaluation.config import (EXAMPLE_WORKLOAD_CONSTRAINTS, RESULTS_DIR, Design, MCConfig,
                                  load_designs_from_results)
from mc_evaluation.design_space import build_design_space
from mc_evaluation.distributions import SAMPLERS, build_scenarios, sample_parameters
from mc_evaluation.monte_carlo import DesignArrays, summarize_designs, total_carbon_samples


def parse_args(argv=None) -> argparse.Namespace:
    d = MCConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workload", default=d.workload, choices=sorted(EXAMPLE_WORKLOAD_CONSTRAINTS))
    p.add_argument("--nwork", type=float, default=d.Nwork, help="workload runs before disposal")
    p.add_argument("--tc-threshold", type=float, default=d.tC_threshold, help="total carbon threshold, kg CO2e")
    p.add_argument("--nominal", default=f"{d.nominal.label}@{d.nominal.gpu_clock:g}",
                   help="nominal design, e.g. c21g20@765 or c21_g20_f765")
    p.add_argument("--robust", default=f"{d.robust.label}@{d.robust.gpu_clock:g}", help="robust design")
    p.add_argument("--results-csv", help="read the nominal/robust designs for --workload and --nwork from this file")
    p.add_argument("--samples", type=int, default=d.n_samples, help="MC samples per distribution")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--distributions", nargs="+", default=list(d.distributions), choices=list(SAMPLERS))
    p.add_argument("--summary-distribution", default=d.summary_distribution, choices=list(SAMPLERS),
                   help="distribution used for the per-design summaries, violations, and tCDP")
    p.add_argument("--verify", action="store_true", help="print realized vs target moments of every sampler")
    p.add_argument("--save-design-summary", action="store_true", help="also save per-design MC summaries (large CSV)")
    p.add_argument("--output-dir", default=str(RESULTS_DIR))
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MCConfig:
    if args.results_csv:
        nominal, robust = load_designs_from_results(args.results_csv, args.workload, args.nwork)
    else:
        nominal, robust = Design.parse(args.nominal), Design.parse(args.robust)
    return replace(
        MCConfig(),
        workload=args.workload, Nwork=args.nwork,
        tC_threshold=args.tc_threshold, nominal=nominal, robust=robust, n_samples=args.samples,
        seed=args.seed, distributions=tuple(args.distributions), summary_distribution=args.summary_distribution,
    )


def report_design_space(cfg: MCConfig, designs: pd.DataFrame, arrays: DesignArrays, samples, idx_n: int,
                        idx_r: int, args: argparse.Namespace, tag: str) -> None:
    """Per-design MC summaries, constraint violations, and tCDP ratios for one distribution."""
    designs = pd.concat([designs, summarize_designs(arrays, samples, cfg.Nwork)], axis=1)
    area_max = cfg.rhs["area"]
    violates_mc = violates_constraints(designs, area_max, cfg.tC_threshold, "Total_p95")
    violates_det = violates_constraints(designs, area_max, cfg.tC_threshold, "Total_carbon")
    print(f"\n[{cfg.summary_distribution}] designs violating area <= {area_max} mm² or carbon <= {cfg.tC_threshold} kg:")
    print(f"  deterministic carbon: {violates_det.sum()} of {len(designs)}")
    print(f"  P95 carbon:           {violates_mc.sum()} of {len(designs)}")

    r = tcdp_report(designs, idx_n, idx_r)
    print(f"\n{cfg.workload} - Nwork = {cfg.Nwork:.0e}, D0 = {cfg.params.D0}, tC < {cfg.tC_threshold}")
    print(f"  Execution time ratio (nominal / robust): {r['time_ratio']:.3f}")
    print(f"  P95 CO₂ ratio (nominal / robust):        {r['carbon_ratio']:.3f}")
    print(f"  tCDP of nominal design:                  {r['tCDP_nominal']:.3f}")
    print(f"  tCDP of robust design:                   {r['tCDP_robust']:.3f}")
    print(f"  tCDP ratio (nominal / robust):           {r['tCDP_ratio']:.3f}")

    if args.save_design_summary:
        path = f"{args.output_dir}/mc_design_summary_{tag}.csv"
        designs.to_csv(path, index=False)
        print(f"Saved per-design summary → {path}")


def main(argv=None) -> pd.DataFrame:
    args = parse_args(argv)
    cfg = config_from_args(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Workload: {cfg.workload}, Nwork ≈ {cfg.Nwork:.1e}, constraints: {cfg.rhs}, "
          f"tC threshold: {cfg.tC_threshold}")
    print(f"Nominal design: {cfg.nominal}")
    print(f"Robust design:  {cfg.robust}")

    designs = build_design_space(cfg.workload, cfg.Nwork, cfg.params)
    idx_n, idx_r = find_design(designs, cfg.nominal), find_design(designs, cfg.robust)
    arrays = DesignArrays.from_design_space(designs, cfg.params)
    scenarios = build_scenarios(cfg.params)
    print(f"Design space: {len(designs)} designs; {cfg.n_samples} MC samples per distribution, seed {cfg.seed}")

    if args.verify:
        print("\nSampler check: realized vs target moments")
        print(verify_scenarios(scenarios, np.random.default_rng(cfg.seed)).to_string(index=False, float_format="%.4g"))

    rows = []
    for dist in cfg.distributions:
        # a fresh generator per distribution: each family's results don't depend on which others run
        samples = sample_parameters(scenarios[dist], cfg.n_samples, np.random.default_rng(cfg.seed))
        totals = total_carbon_samples(arrays, samples, cfg.Nwork, [idx_n, idx_r])
        stats = compare_nominal_robust(totals[:, 0], totals[:, 1], np.random.default_rng([cfg.seed, 1]))
        rows.append({"Distribution": dist, **stats})
        print(f"  {dist}: P95 improvement = {stats['P95_Improvement_%']:.2f}% "
              f"(P95 ratio {stats['P95_Ratio']:.3f}, 95% CI {stats['P95_Ratio_CI95_low']:.3f}-"
              f"{stats['P95_Ratio_CI95_high']:.3f})")
        if dist == cfg.summary_distribution:
            tag = f"{cfg.workload}_tC{cfg.tC_threshold:g}_Nwork{cfg.Nwork:.0e}_{dist}"
            report_design_space(cfg, designs, arrays, samples, idx_n, idx_r, args, tag)

    table = pd.DataFrame(rows)
    print("\n" + "=" * 120)
    print("NOMINAL VS ROBUST ACROSS DISTRIBUTIONS (total carbon, kg CO₂e)")
    print(f"Workload: {cfg.workload} | Nwork: {cfg.Nwork:.1e} | Constraints: {cfg.rhs} | "
          f"tC < {cfg.tC_threshold} | Nominal: {cfg.nominal} | Robust: {cfg.robust}")
    print("=" * 120)
    print(table.to_string(index=False, float_format="%.3f"))
    path = f"{args.output_dir}/mc_comparison_all_distributions_{cfg.workload}_tC{cfg.tC_threshold:g}_Nwork{cfg.Nwork:.0e}.csv"
    table.to_csv(path, index=False)
    print(f"\nSaved comparison table → {path}")
    return table


if __name__ == "__main__":
    main()
