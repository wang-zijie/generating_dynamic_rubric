"""
Statistical significance testing between two evaluation methods.

For pairwise benchmarks (AlpacaEval, MTBench, BiGGen):
  - Metric: human agreement %
  - Test: McNemar's test (paired binary outcomes) + bootstrap CI

For pointwise benchmarks (HelpSteer2, HealthBench):
  - Metric: Spearman/Pearson correlation
  - Test: Bootstrap CI on correlation difference

Usage:
    # Compare CheckEval vs dynamic rubrics with Claude on AlpacaEval
    python src/significance_test.py \
        --benchmark alpacaeval \
        --model claude-sonnet-4 \
        --method-a checkeval \
        --method-b dynamic

    # Compare on pointwise benchmark
    python src/significance_test.py \
        --benchmark helpsteer2 \
        --model claude-sonnet-4 \
        --method-a checkeval \
        --method-b dynamic

    # List available methods for a benchmark
    python src/significance_test.py --benchmark alpacaeval --list-methods
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)

PAIRWISE_BENCHMARKS = ["alpacaeval", "mtbench", "biggen"]
POINTWISE_BENCHMARKS = ["helpsteer2", "healthbench"]

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"

# Map method names to their output directory structure
METHOD_PATHS = {
    # Baselines
    "checkeval": ("checkeval/{benchmark}", "judge_{model}.json"),
    "dna": ("dna/{benchmark}", "judge_{model}.json"),
    "rubrichub": ("rubrichub/{benchmark}", "judge_{model}.json"),
    # Our methods - pairwise
    "fixed": ("pairwise/{benchmark}/fixed", "judge_{model}.json"),
    "dynamic": ("pairwise/{benchmark}/dynamic", "judge_{model}.json"),
    "generated_fixed": ("pairwise/{benchmark}/generated_fixed", "judge_{model}.json"),
    # Our methods - pointwise
    "fixed_pw": ("pointwise/{benchmark}/fixed", "judge_{model}.json"),
    "dynamic_pw": ("pointwise/{benchmark}/dynamic", "judge_{model}.json"),
    "generated_fixed_pw": ("pointwise/{benchmark}/generated_fixed", "judge_{model}.json"),
    "fixed_instance": ("pointwise/{benchmark}/fixed_instance", "judge_{model}.json"),
    # Human-crafted instance-specific (BiGGen pairwise only)
    "existing_fixed_instance": ("pairwise/{benchmark}/existing_fixed_instance", "judge_{model}.json"),
    # DPO methods
    "pairwise_rubric_gen_v1": (
        "pairwise/{benchmark}/pairwise_rubric_gen_v1",
        "judge_{model}_pairwise_rubric_gen_v1.json",
    ),
    "pointwise_rubric_gen_v1": (
        "pointwise/{benchmark}/pointwise_rubric_gen_v1",
        "judge_{model}_pointwise_rubric_gen_v1.json",
    ),
    "all_rubric_gen_v1": None,  # special handling below
}

MODEL_ALIASES = {
    "claude": "claude-sonnet-4",
    "claude-sonnet-4": "claude-sonnet-4",
    "70b": "llama-3.1-70b",
    "llama-70b": "llama-3.1-70b",
    "llama-3.1-70b": "llama-3.1-70b",
    "8b": "llama-3.1-8b",
    "llama-8b": "llama-3.1-8b",
    "llama-3.1-8b": "llama-3.1-8b",
}


def resolve_result_path(method: str, benchmark: str, model: str) -> Path:
    """Resolve the file path for a given method/benchmark/model combination."""
    if method == "all_rubric_gen_v1":
        if benchmark in PAIRWISE_BENCHMARKS:
            dir_path = OUTPUTS_DIR / "pairwise" / benchmark / "all_rubric_gen_v1"
        else:
            dir_path = OUTPUTS_DIR / "pointwise" / benchmark / "all_rubric_gen_v1"
        return dir_path / f"judge_{model}_all_rubric_gen_v1.json"

    path_info = METHOD_PATHS.get(method)
    if path_info is None:
        raise ValueError(f"Unknown method: {method}")

    dir_template, file_template = path_info
    dir_path = OUTPUTS_DIR / dir_template.format(benchmark=benchmark)
    file_path = dir_path / file_template.format(model=model)

    # For methods that work on both pairwise and pointwise, try pointwise if pairwise not found
    if not file_path.exists() and benchmark in POINTWISE_BENCHMARKS:
        for suffix in ["_pw", ""]:
            alt_method = method + suffix if suffix else method
            alt_info = METHOD_PATHS.get(alt_method)
            if alt_info:
                alt_dir = OUTPUTS_DIR / alt_info[0].format(benchmark=benchmark)
                alt_file = alt_dir / alt_info[1].format(model=model)
                if alt_file.exists():
                    return alt_file

    return file_path


def load_pairwise_results(path: Path) -> list[dict]:
    """Load pairwise judge results. Returns list with 'preference' and 'human_majority' keys."""
    with open(path) as f:
        data = json.load(f)
    return data


def load_pointwise_results(path: Path) -> list[dict]:
    """Load pointwise judge results. Returns list with 'judge_score' and 'human_score' keys."""
    with open(path) as f:
        data = json.load(f)
    return data


def get_pairwise_correctness(results: list[dict]) -> np.ndarray:
    """Extract binary correctness array from pairwise results."""
    correct = []
    for r in results:
        pref = r.get("preference")
        human = r.get("human_majority")
        if pref is None or human is None:
            correct.append(0)
        else:
            correct.append(int(pref == human))
    return np.array(correct)


def get_pointwise_scores(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract judge and human scores from pointwise results."""
    judge_scores = []
    human_scores = []
    for r in results:
        js = r.get("judge_score")
        hs = r.get("human_score")
        if js is not None and hs is not None:
            judge_scores.append(float(js))
            human_scores.append(float(hs))
    return np.array(judge_scores), np.array(human_scores)


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """
    McNemar's test for paired binary outcomes.
    Tests whether the two methods have significantly different error rates.
    """
    n = len(correct_a)
    assert len(correct_b) == n

    # Contingency: b01 = A wrong, B right; b10 = A right, B wrong
    b01 = np.sum((correct_a == 0) & (correct_b == 1))
    b10 = np.sum((correct_a == 1) & (correct_b == 0))
    b00 = np.sum((correct_a == 0) & (correct_b == 0))
    b11 = np.sum((correct_a == 1) & (correct_b == 1))

    # McNemar's test with continuity correction
    if b01 + b10 == 0:
        p_value = 1.0
        chi2 = 0.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)

    return {
        "test": "McNemar",
        "chi2": chi2,
        "p_value": p_value,
        "n": n,
        "both_correct": int(b11),
        "both_wrong": int(b00),
        "a_right_b_wrong": int(b10),
        "a_wrong_b_right": int(b01),
        "accuracy_a": correct_a.mean(),
        "accuracy_b": correct_b.mean(),
    }


def bootstrap_accuracy_diff(
    correct_a: np.ndarray, correct_b: np.ndarray, n_bootstrap: int = 10000, seed: int = 42
) -> dict:
    """Bootstrap confidence interval for difference in accuracy (A - B)."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        acc_a = correct_a[idx].mean()
        acc_b = correct_b[idx].mean()
        diffs.append(acc_a - acc_b)
    diffs = np.array(diffs)

    observed_diff = correct_a.mean() - correct_b.mean()
    ci_lower = np.percentile(diffs, 2.5)
    ci_upper = np.percentile(diffs, 97.5)
    p_value = np.mean(diffs <= 0) if observed_diff > 0 else np.mean(diffs >= 0)
    p_value = min(2 * p_value, 1.0)  # two-sided

    return {
        "test": "bootstrap_accuracy_diff",
        "observed_diff": observed_diff,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_correlation_diff(
    judge_a: np.ndarray,
    judge_b: np.ndarray,
    human: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap test for difference in correlations.
    Tests whether corr(judge_a, human) differs from corr(judge_b, human).
    """
    rng = np.random.default_rng(seed)
    n = len(human)

    obs_spearman_a = spearmanr(judge_a, human).statistic
    obs_spearman_b = spearmanr(judge_b, human).statistic
    obs_pearson_a = pearsonr(judge_a, human).statistic
    obs_pearson_b = pearsonr(judge_b, human).statistic

    spearman_diffs = []
    pearson_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ja, jb, h = judge_a[idx], judge_b[idx], human[idx]
        if np.std(ja) == 0 or np.std(jb) == 0 or np.std(h) == 0:
            continue
        sp_a = spearmanr(ja, h).statistic
        sp_b = spearmanr(jb, h).statistic
        pe_a = pearsonr(ja, h).statistic
        pe_b = pearsonr(jb, h).statistic
        spearman_diffs.append(sp_a - sp_b)
        pearson_diffs.append(pe_a - pe_b)

    spearman_diffs = np.array(spearman_diffs)
    pearson_diffs = np.array(pearson_diffs)

    obs_sp_diff = obs_spearman_a - obs_spearman_b
    obs_pe_diff = obs_pearson_a - obs_pearson_b

    sp_p = np.mean(spearman_diffs <= 0) if obs_sp_diff > 0 else np.mean(spearman_diffs >= 0)
    sp_p = min(2 * sp_p, 1.0)
    pe_p = np.mean(pearson_diffs <= 0) if obs_pe_diff > 0 else np.mean(pearson_diffs >= 0)
    pe_p = min(2 * pe_p, 1.0)

    return {
        "test": "bootstrap_correlation_diff",
        "spearman_a": obs_spearman_a,
        "spearman_b": obs_spearman_b,
        "spearman_diff": obs_sp_diff,
        "spearman_ci_95": (np.percentile(spearman_diffs, 2.5), np.percentile(spearman_diffs, 97.5)),
        "spearman_p_value": sp_p,
        "pearson_a": obs_pearson_a,
        "pearson_b": obs_pearson_b,
        "pearson_diff": obs_pe_diff,
        "pearson_ci_95": (np.percentile(pearson_diffs, 2.5), np.percentile(pearson_diffs, 97.5)),
        "pearson_p_value": pe_p,
        "n_bootstrap": n_bootstrap,
        "n_samples": n,
    }


def find_available_methods(benchmark: str) -> list[str]:
    """List methods that have results for a given benchmark."""
    available = []
    for method in METHOD_PATHS:
        try:
            # Try any model
            for model in ["claude-sonnet-4", "llama-3.1-70b", "llama-3.1-8b"]:
                path = resolve_result_path(method, benchmark, model)
                if path.exists():
                    available.append(method)
                    break
        except (ValueError, KeyError):
            continue
    return sorted(set(available))


def run_significance_test(benchmark: str, model: str, method_a: str, method_b: str, n_bootstrap: int = 10000) -> dict:
    """Run the appropriate significance test for the given benchmark type."""
    path_a = resolve_result_path(method_a, benchmark, model)
    path_b = resolve_result_path(method_b, benchmark, model)

    if not path_a.exists():
        raise FileNotFoundError(f"Results not found: {path_a}")
    if not path_b.exists():
        raise FileNotFoundError(f"Results not found: {path_b}")

    if benchmark in PAIRWISE_BENCHMARKS:
        results_a = load_pairwise_results(path_a)
        results_b = load_pairwise_results(path_b)

        correct_a_full = get_pairwise_correctness(results_a)
        correct_b_full = get_pairwise_correctness(results_b)

        # Try to align by (instruction, generator)
        def make_pairwise_key(r):
            return (r["instruction"], r.get("generator", ""))

        key_to_a = {make_pairwise_key(r): i for i, r in enumerate(results_a)}
        key_to_b = {make_pairwise_key(r): i for i, r in enumerate(results_b)}
        common = set(key_to_a.keys()) & set(key_to_b.keys())

        if len(common) == 0 and len(results_a) == len(results_b):
            # Fall back to positional alignment (same length, assume same order)
            # Verify by checking instructions match
            instr_match = sum(
                1 for i in range(len(results_a))
                if results_a[i]["instruction"] == results_b[i]["instruction"]
            )
            if instr_match > 0.9 * len(results_a):
                logger.info("Using positional alignment (%d/%d instructions match)",
                           instr_match, len(results_a))
                correct_a = correct_a_full
                correct_b = correct_b_full
            else:
                raise ValueError("Cannot align results: no key overlap and positional mismatch")
        else:
            common = sorted(common, key=lambda x: x[0])
            if len(common) < len(results_a) or len(common) < len(results_b):
                logger.warning(
                    f"Only {len(common)} common instances "
                    f"(A={len(results_a)}, B={len(results_b)})"
                )
            idx_a = [key_to_a[k] for k in common]
            idx_b = [key_to_b[k] for k in common]
            correct_a = correct_a_full[idx_a]
            correct_b = correct_b_full[idx_b]

        mcnemar = mcnemar_test(correct_a, correct_b)
        bootstrap = bootstrap_accuracy_diff(correct_a, correct_b, n_bootstrap=n_bootstrap)

        return {
            "benchmark": benchmark,
            "model": model,
            "method_a": method_a,
            "method_b": method_b,
            "metric": "human_agreement",
            "mcnemar": mcnemar,
            "bootstrap": bootstrap,
        }

    else:
        results_a = load_pointwise_results(path_a)
        results_b = load_pointwise_results(path_b)

        # Align by (prompt, response) to handle duplicate prompts
        def make_key(r):
            return (r["prompt"], r.get("response", "")[:200])

        key_to_a = {make_key(r): i for i, r in enumerate(results_a)}
        key_to_b = {make_key(r): i for i, r in enumerate(results_b)}
        common = sorted(set(key_to_a.keys()) & set(key_to_b.keys()), key=lambda x: x[0])

        if len(common) < len(results_a) or len(common) < len(results_b):
            logger.warning(
                f"Only {len(common)} common instances "
                f"(A={len(results_a)}, B={len(results_b)})"
            )

        judge_a_scores = []
        judge_b_scores = []
        human_scores = []
        for key in common:
            ra = results_a[key_to_a[key]]
            rb = results_b[key_to_b[key]]
            ja = ra.get("judge_score")
            jb = rb.get("judge_score")
            ha = ra.get("human_score")
            if ja is not None and jb is not None and ha is not None:
                judge_a_scores.append(float(ja))
                judge_b_scores.append(float(jb))
                human_scores.append(float(ha))

        judge_a_arr = np.array(judge_a_scores)
        judge_b_arr = np.array(judge_b_scores)
        human_arr = np.array(human_scores)

        bootstrap = bootstrap_correlation_diff(
            judge_a_arr, judge_b_arr, human_arr, n_bootstrap=n_bootstrap
        )

        return {
            "benchmark": benchmark,
            "model": model,
            "method_a": method_a,
            "method_b": method_b,
            "metric": "correlation",
            "bootstrap": bootstrap,
        }


def format_results(results: dict) -> str:
    """Format significance test results for display."""
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"Significance Test: {results['method_a']} vs {results['method_b']}")
    lines.append(f"Benchmark: {results['benchmark']}, Model: {results['model']}")
    lines.append(f"{'='*60}")

    if results["metric"] == "human_agreement":
        mc = results["mcnemar"]
        bs = results["bootstrap"]
        lines.append(f"\nAccuracy A ({results['method_a']}): {mc['accuracy_a']*100:.2f}%")
        lines.append(f"Accuracy B ({results['method_b']}): {mc['accuracy_b']*100:.2f}%")
        lines.append(f"Difference (A - B): {bs['observed_diff']*100:.2f}%")
        lines.append(f"\nMcNemar's Test:")
        lines.append(f"  chi2 = {mc['chi2']:.4f}, p = {mc['p_value']:.4f}")
        lines.append(f"  A right, B wrong: {mc['a_right_b_wrong']}")
        lines.append(f"  A wrong, B right: {mc['a_wrong_b_right']}")
        lines.append(f"\nBootstrap (n={bs['n_bootstrap']}):")
        lines.append(f"  95% CI: [{bs['ci_95_lower']*100:.2f}%, {bs['ci_95_upper']*100:.2f}%]")
        lines.append(f"  p-value: {bs['p_value']:.4f}")
        sig = "YES" if bs["p_value"] < 0.05 else "NO"
        lines.append(f"\nSignificant at alpha=0.05? {sig}")

    else:
        bs = results["bootstrap"]
        lines.append(f"\nSpearman A ({results['method_a']}): {bs['spearman_a']:.4f}")
        lines.append(f"Spearman B ({results['method_b']}): {bs['spearman_b']:.4f}")
        lines.append(f"Spearman Diff (A - B): {bs['spearman_diff']:.4f}")
        lines.append(f"  95% CI: [{bs['spearman_ci_95'][0]:.4f}, {bs['spearman_ci_95'][1]:.4f}]")
        lines.append(f"  p-value: {bs['spearman_p_value']:.4f}")
        sig_sp = "YES" if bs["spearman_p_value"] < 0.05 else "NO"
        lines.append(f"  Significant? {sig_sp}")
        lines.append(f"\nPearson A ({results['method_a']}): {bs['pearson_a']:.4f}")
        lines.append(f"Pearson B ({results['method_b']}): {bs['pearson_b']:.4f}")
        lines.append(f"Pearson Diff (A - B): {bs['pearson_diff']:.4f}")
        lines.append(f"  95% CI: [{bs['pearson_ci_95'][0]:.4f}, {bs['pearson_ci_95'][1]:.4f}]")
        lines.append(f"  p-value: {bs['pearson_p_value']:.4f}")
        sig_pe = "YES" if bs["pearson_p_value"] < 0.05 else "NO"
        lines.append(f"  Significant? {sig_pe}")
        lines.append(f"\n  n_samples: {bs['n_samples']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Statistical significance testing between evaluation methods")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=PAIRWISE_BENCHMARKS + POINTWISE_BENCHMARKS)
    parser.add_argument("--model", type=str, default="claude-sonnet-4",
                        help="Judge model (e.g., claude-sonnet-4, llama-3.1-70b)")
    parser.add_argument("--method-a", type=str, help="First method")
    parser.add_argument("--method-b", type=str, help="Second method")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--list-methods", action="store_true",
                        help="List available methods for the benchmark")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Resolve model alias
    model = MODEL_ALIASES.get(args.model, args.model)

    if args.list_methods:
        methods = find_available_methods(args.benchmark)
        print(f"Available methods for {args.benchmark}:")
        for m in methods:
            print(f"  {m}")
        return

    if not args.method_a or not args.method_b:
        parser.error("--method-a and --method-b are required (or use --list-methods)")

    results = run_significance_test(
        benchmark=args.benchmark,
        model=model,
        method_a=args.method_a,
        method_b=args.method_b,
        n_bootstrap=args.n_bootstrap,
    )
    print(format_results(results))


if __name__ == "__main__":
    main()
