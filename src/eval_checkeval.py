#!/usr/bin/env python3
"""
Evaluate LLM judges using the CheckEval methodology (binary Yes/No checklists).

CheckEval decomposes evaluation into a checklist of binary Yes/No questions.
The score for a response is the proportion of "Yes" answers. For pairwise
benchmarks, each response is scored independently and the higher score wins.

Benchmarks:
    Pairwise:  AlpacaEval, MTBench, BiGGen Bench
    Pointwise: HelpSteer2, ProfBench, HealthBench

Each judge model generates its own checklist of 15-20 binary questions per
benchmark. The checklist is cached to disk and reused for all items.

Reference:
    Lee et al. (2025). CheckEval: Robust LLM Evaluation via Checklist.
    EMNLP 2025. https://aclanthology.org/2025.emnlp-main.796.pdf

Authentication (environment variables):
    AWS_BEARER_TOKEN_BEDROCK  -- Bearer token for Bedrock API
    AWS_REGION                -- AWS region (default: us-east-1)

Usage:
    # Pairwise: AlpacaEval
    python src/eval_checkeval.py --benchmark alpacaeval --judges claude-sonnet-4

    # Pairwise: MTBench
    python src/eval_checkeval.py --benchmark mtbench --judges llama-3.1-8b

    # Pointwise: HelpSteer2
    python src/eval_checkeval.py --benchmark helpsteer2 --judges llama-3.1-70b

    # Quick test
    python src/eval_checkeval.py --benchmark helpsteer2 --judges llama-3.1-8b \\
        --max-examples 5
"""

import argparse
import io
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from client import BedrockClient, call_judge, JUDGES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge model registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Benchmark type classification
# ---------------------------------------------------------------------------
PAIRWISE_BENCHMARKS = {"alpacaeval", "mtbench", "biggen"}
POINTWISE_BENCHMARKS = {"helpsteer2", "profbench", "healthbench", "biggen_pointwise"}

# ---------------------------------------------------------------------------
# Benchmark descriptions (for checklist generation)
# ---------------------------------------------------------------------------
BENCHMARK_DESCRIPTIONS = {
    "alpacaeval": (
        "responses to general user instructions, covering tasks like Q&A, "
        "creative writing, reasoning, and advice"
    ),
    "mtbench": (
        "multi-turn conversation responses, covering writing, roleplay, "
        "reasoning, math, coding, extraction, STEM, and humanities"
    ),
    "biggen": (
        "responses to capability-testing prompts spanning reasoning, "
        "planning, instruction following, safety, and tool usage"
    ),
    "biggen_pointwise": (
        "individual AI responses to capability-testing prompts, scored on "
        "a 1-5 scale for accuracy, completeness, clarity, instruction "
        "following, and quality"
    ),
    "helpsteer2": (
        "AI assistant responses, focusing on helpfulness, correctness, "
        "coherence, and appropriate detail"
    ),
    "profbench": (
        "professional-grade responses in specialized domains like "
        "chemistry, law, and medicine"
    ),
    "healthbench": (
        "health and medical responses, focusing on accuracy, safety, "
        "and appropriate clinical guidance"
    ),
}

# ---------------------------------------------------------------------------
# CheckEval prompt templates
# ---------------------------------------------------------------------------

CHECKLIST_GEN_SYSTEM_PROMPT = """\
You are an expert evaluator designing a checklist for evaluating AI \
responses. Generate a checklist of {num_questions} binary (Yes/No) questions \
for evaluating {benchmark_description}.

Rules:
1. Each question must be answerable with 'Yes' or 'No'
2. A 'Yes' answer should indicate positive quality
3. Cover different aspects of response quality (accuracy, completeness, \
clarity, safety, etc.)
4. Be specific enough for consistent binary judgment
5. Minimize redundancy between questions

Output ONLY numbered questions, one per line. For example:
1. Does the response directly address the user's question or request?
2. Is the information provided factually accurate?
..."""

CHECKLIST_GEN_USER_PROMPT = """\
Generate a checklist of {num_questions} binary (Yes/No) evaluation questions \
for assessing {benchmark_description}."""

CHECKEVAL_SYSTEM_PROMPT = """\
Answer each question about the following response with 'Yes' or 'No'. \
Do not provide explanations."""

CHECKEVAL_USER_TEMPLATE = """\
[Prompt]
{prompt}

[Response]
{response}

[Questions]
{questions}

Answer each question in this exact format:
Q1: Yes
Q2: No
..."""


# ---------------------------------------------------------------------------
# HuggingFace dataset download helper
# ---------------------------------------------------------------------------

def _download_hf_dataset(
    dataset_id: str, split: str, cache_dir: Path,
) -> pd.DataFrame:
    """Download a HuggingFace dataset split as a DataFrame (via Parquet)."""
    safe_name = dataset_id.replace("/", "_")
    cache_path = cache_dir / f"{safe_name}_{split}.parquet"

    if cache_path.exists():
        logger.info("Loading cached %s", cache_path)
        return pd.read_parquet(cache_path)

    api_url = (
        f"https://datasets-server.huggingface.co/parquet?dataset={dataset_id}"
    )
    logger.info("Fetching parquet info for %s ...", dataset_id)

    try:
        resp = requests.get(api_url, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise ValueError(
            f"Failed to fetch parquet info for {dataset_id}: {e}. "
            f"The dataset may not be public or may require authentication."
        ) from e

    info = resp.json()
    if "error" in info:
        raise ValueError(f"HF API error for {dataset_id}: {info['error']}")

    parquet_files = info.get("parquet_files", [])
    split_files = [f for f in parquet_files if f["split"] == split]

    if not split_files:
        available = sorted(
            set(
                (f.get("config", "?"), f["split"]) for f in parquet_files
            )
        )
        raise ValueError(
            f"Split '{split}' not found for {dataset_id}. "
            f"Available (config, split) pairs: {available}"
        )

    dfs = []
    for pf in split_files:
        url = pf["url"]
        logger.info("Downloading %s ...", url)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content))
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path)
    logger.info("Cached %d rows -> %s", len(result), cache_path)
    return result


# ---------------------------------------------------------------------------
# Data loaders -- Pairwise
# ---------------------------------------------------------------------------

HF_ALPACAEVAL_BASE = (
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main"
)


def _download_json(filename: str, cache_dir: Path) -> list[dict]:
    cache_path = cache_dir / filename
    if cache_path.exists():
        logger.info("Loading cached %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)
    url = f"{HF_ALPACAEVAL_BASE}/{filename}"
    logger.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Cached %d entries -> %s", len(data), cache_path)
    return data


def load_alpacaeval(cache_dir: Path) -> list[dict]:
    """Load AlpacaEval cross-annotations as pairwise evaluation pairs.

    Returns list of dicts with keys:
        instruction, output_1, output_2, generator,
        human_majority (1 or 2), human_preferences
    """
    ae_dir = cache_dir / "alpacaeval"
    ae_dir.mkdir(parents=True, exist_ok=True)

    cross_raw = _download_json(
        "alpaca_farm_human_crossannotations.json", ae_dir,
    )
    human_ann_raw = _download_json(
        "alpaca_farm_human_annotations.json", ae_dir,
    )

    key_to_gen: dict[tuple[str, str, str], str] = {}
    for item in human_ann_raw:
        key = (item["instruction"], item["output_1"], item["output_2"])
        key_to_gen.setdefault(key, item["generator"])

    cross_eval = [a for a in cross_raw if a.get("datasplit") == "eval"]
    logger.info(
        "AlpacaEval cross-annotations: %d total, %d in eval split",
        len(cross_raw), len(cross_eval),
    )

    human_annotations: dict[tuple, list[int]] = {}
    pair_meta: dict[tuple, dict] = {}

    for ann in cross_eval:
        key = (ann["instruction"], ann["output_1"], ann["output_2"])
        human_annotations.setdefault(key, []).append(ann["preference"])
        if key not in pair_meta:
            pair_meta[key] = {
                "instruction": ann["instruction"],
                "output_1": ann["output_1"],
                "output_2": ann["output_2"],
                "generator": key_to_gen.get(key, "unknown"),
            }

    rng = random.Random(42)
    pairs = []
    for key, prefs in human_annotations.items():
        counts = Counter(prefs)
        max_count = max(counts.values())
        modes = [k for k, v in counts.items() if v == max_count]
        majority = rng.choice(modes)

        entry = dict(pair_meta[key])
        entry["human_majority"] = majority
        entry["human_preferences"] = prefs
        pairs.append(entry)

    logger.info("AlpacaEval: %d unique pairs", len(pairs))
    return pairs


def load_mtbench(cache_dir: Path) -> list[dict]:
    """Load MTBench human judgments as pairwise evaluation pairs.

    Returns list of dicts with keys:
        instruction, output_1, output_2, human_majority (1 or 2)
    """
    bm_dir = cache_dir / "mtbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "lmsys/mt_bench_human_judgments", "human", bm_dir,
    )
    logger.info("MTBench columns: %s", list(df.columns))
    logger.info("MTBench shape: %s", df.shape)

    # Each row has: question_id, model_a, model_b, winner,
    #               conversation_a, conversation_b, turn
    # winner is one of: "model_a", "model_b", "tie"

    # Group by (question_id, model_a, model_b) and compute majority winner
    groups: dict[tuple, list[str]] = defaultdict(list)
    group_data: dict[tuple, dict] = {}

    for _, row in df.iterrows():
        qid = row["question_id"]
        model_a = row.get("model_a", "model_a")
        model_b = row.get("model_b", "model_b")
        winner = row["winner"]
        key = (qid, model_a, model_b)

        groups[key].append(winner)

        if key not in group_data:
            # Extract instruction and responses from conversations
            conv_a = row["conversation_a"]
            conv_b = row["conversation_b"]

            # Conversations are lists/arrays of dicts with 'role'+'content'
            if isinstance(conv_a, (list, np.ndarray)):
                instr_parts = []
                resp_a_parts = []
                for msg in conv_a:
                    if msg["role"] == "user":
                        instr_parts.append(msg["content"])
                    elif msg["role"] == "assistant":
                        resp_a_parts.append(msg["content"])
                instruction = "\n\n".join(instr_parts)
                output_1 = "\n\n".join(resp_a_parts)
            else:
                instruction = str(conv_a)
                output_1 = str(conv_a)

            if isinstance(conv_b, (list, np.ndarray)):
                resp_b_parts = []
                for msg in conv_b:
                    if msg["role"] == "assistant":
                        resp_b_parts.append(msg["content"])
                output_2 = "\n\n".join(resp_b_parts)
            else:
                output_2 = str(conv_b)

            group_data[key] = {
                "instruction": instruction,
                "output_1": output_1,
                "output_2": output_2,
            }

    # Compute majority winner per group
    rng = random.Random(42)
    pairs = []
    n_ties = 0

    for key, winners in groups.items():
        counts = Counter(winners)
        max_count = max(counts.values())
        modes = [k for k, v in counts.items() if v == max_count]
        majority_winner = rng.choice(modes)

        if majority_winner == "tie":
            n_ties += 1
            continue

        # model_a = output_1, model_b = output_2
        human_majority = 1 if majority_winner == "model_a" else 2

        entry = dict(group_data[key])
        entry["human_majority"] = human_majority
        pairs.append(entry)

    logger.info(
        "MTBench: %d pairs (%d ties excluded)",
        len(pairs), n_ties,
    )
    return pairs


def load_biggen(cache_dir: Path) -> list[dict]:
    """Load BiGGen Bench as synthetic pairwise evaluation pairs.

    BiGGen Bench is pointwise (1-5 human scores per model per prompt).
    We construct synthetic pairwise pairs by comparing models on the same
    prompt and using the higher human score as the preferred response.

    Returns list of dicts with keys:
        instruction, output_1, output_2, human_majority (1 or 2)
    """
    bm_dir = cache_dir / "biggen"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "prometheus-eval/BiGGen-Bench-Results", "human_eval", bm_dir,
    )
    logger.info("BiGGen Bench columns: %s", list(df.columns))
    logger.info("BiGGen Bench shape: %s", df.shape)

    # Group by prompt -- collect (model, response, score)
    prompt_data: dict[str, list[dict]] = defaultdict(list)

    for _, row in df.iterrows():
        # Try various column name possibilities
        prompt = str(row.get("input", row.get("prompt", "")))
        response = str(row.get("response", row.get("output", "")))
        human_score = float(
            row.get("human_score", row.get("score", row.get("human_eval", 0)))
        )
        model = str(
            row.get(
                "model_name",
                row.get("model", row.get("generator", "unknown")),
            )
        )

        prompt_data[prompt].append(
            {
                "model": model,
                "response": response,
                "human_score": human_score,
            }
        )

    # Construct synthetic pairwise pairs
    pairs = []
    n_ties = 0

    for prompt, models in prompt_data.items():
        if len(models) < 2:
            continue

        for m1, m2 in combinations(range(len(models)), 2):
            s1 = models[m1]["human_score"]
            s2 = models[m2]["human_score"]

            if s1 == s2:
                n_ties += 1
                continue

            # output_1 = m1's response, output_2 = m2's response
            human_majority = 1 if s1 > s2 else 2

            pairs.append(
                {
                    "instruction": prompt,
                    "output_1": models[m1]["response"],
                    "output_2": models[m2]["response"],
                    "human_majority": human_majority,
                }
            )

    logger.info(
        "BiGGen Bench: %d pairs (%d ties excluded)",
        len(pairs), n_ties,
    )
    return pairs


# ---------------------------------------------------------------------------
# Data loaders -- Pointwise
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS = {"Critical": 4, "Major": 3, "Minor": 2, "Additional": 1}


def load_helpsteer2(cache_dir: Path) -> list[dict]:
    """Load HelpSteer2 validation split.

    Returns list of dicts with keys:
        prompt, response, human_score
    """
    bm_dir = cache_dir / "helpsteer2"
    bm_dir.mkdir(parents=True, exist_ok=True)
    df = _download_hf_dataset("nvidia/HelpSteer2", "validation", bm_dir)

    items = []
    for _, row in df.iterrows():
        avg_score = (float(row["helpfulness"]) + float(row["correctness"]) + float(row["coherence"])) / 3.0
        items.append(
            {
                "prompt": str(row["prompt"]),
                "response": str(row["response"]),
                "human_score": avg_score,
            }
        )

    logger.info(
        "HelpSteer2: %d items, score range [%.1f, %.1f]",
        len(items),
        min(it["human_score"] for it in items),
        max(it["human_score"] for it in items),
    )
    return items


def _compute_profbench_score(
    rubric_criteria: list[dict], model_key: str,
) -> float:
    """Compute weighted fulfilment score (0-100) for a model on ProfBench."""
    total_weight = 0
    weighted_sum = 0

    for c in rubric_criteria:
        severity = c.get(
            "criterion_weight",
            c.get("severity", c.get("weight", "Minor")),
        )
        if isinstance(severity, str):
            weight = SEVERITY_WEIGHTS.get(severity, 1)
        else:
            weight = float(severity)

        fulfilled = None
        for key in [
            f"{model_key}_fulfilment",
            f"{model_key}_fulfilled",
            f"fulfilled_{model_key}",
            f"fulfilment_{model_key}",
            "fulfilled",
        ]:
            if key in c:
                fulfilled = c[key]
                break

        if fulfilled is None:
            continue

        total_weight += weight
        if fulfilled:
            weighted_sum += weight

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight * 100


def load_profbench(cache_dir: Path) -> list[dict]:
    """Load ProfBench dataset.

    Returns list of dicts with keys:
        prompt, response, human_score, task_id, domain, response_model
    """
    bm_dir = cache_dir / "profbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = None
    for split in ["test", "train", "validation"]:
        try:
            df = _download_hf_dataset("nvidia/ProfBench", split, bm_dir)
            break
        except ValueError:
            continue
    if df is None:
        raise ValueError("Could not find a valid split for nvidia/ProfBench")

    logger.info("ProfBench columns: %s", list(df.columns))

    response_cols = [c for c in df.columns if c.endswith("_response")]
    if not response_cols:
        response_cols = [
            c for c in df.columns
            if "response" in c.lower() and c.lower() != "response"
        ]
    if not response_cols and "response" in df.columns:
        response_cols = ["response"]

    rubric_col = None
    for candidate in ["rubric_criteria", "rubrics", "criteria", "rubric"]:
        if candidate in df.columns:
            rubric_col = candidate
            break

    items = []
    for _, row in df.iterrows():
        prompt = str(
            row.get("prompt", row.get("task", row.get("instruction", "")))
        )
        task_id = row.get("task_id", row.get("id", None))
        domain = row.get("domain", None)

        raw_rubric = row.get(rubric_col) if rubric_col else None
        if isinstance(raw_rubric, str):
            try:
                rubric_criteria = json.loads(raw_rubric)
            except json.JSONDecodeError:
                rubric_criteria = []
        elif isinstance(raw_rubric, (list, np.ndarray)):
            rubric_criteria = []
            for c in raw_rubric:
                clean = {}
                for k, v in c.items():
                    if isinstance(v, np.ndarray):
                        clean[k] = v.tolist()
                    elif isinstance(v, (np.bool_, np.integer)):
                        clean[k] = v.item()
                    else:
                        clean[k] = v
                rubric_criteria.append(clean)
        else:
            rubric_criteria = []

        for col in response_cols:
            response = str(row[col])
            if not response or response == "nan":
                continue

            model_key = col.replace("_response", "").replace("_output", "")
            human_score = _compute_profbench_score(rubric_criteria, model_key)

            items.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "human_score": human_score,
                    "task_id": task_id,
                    "domain": domain,
                    "response_model": model_key,
                }
            )

    if items:
        logger.info(
            "ProfBench: %d items, score range [%.1f, %.1f]",
            len(items),
            min(it["human_score"] for it in items),
            max(it["human_score"] for it in items),
        )
    return items


def _format_messages(messages) -> str:
    """Format a multi-turn message list into readable text."""
    if isinstance(messages, str):
        return messages
    parts = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", str(part)))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


HF_HEALTHBENCH_BASE = (
    "https://huggingface.co/datasets/openai/healthbench/resolve/main"
)


def _download_healthbench_jsonl(
    filename: str, cache_dir: Path,
) -> list[dict]:
    """Download and cache a HealthBench JSONL file."""
    cache_path = cache_dir / filename
    if cache_path.exists():
        logger.info("Loading cached %s", cache_path)
        with open(cache_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    url = f"{HF_HEALTHBENCH_BASE}/{filename}"
    logger.info("Downloading %s ...", url)
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(resp.content)

    data = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    logger.info("Cached %d rows -> %s", len(data), cache_path)
    return data


def load_healthbench(cache_dir: Path) -> list[dict]:
    """Load HealthBench dataset.

    Returns list of dicts with keys:
        prompt, response, human_score
    """
    bm_dir = cache_dir / "healthbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    eval_data = _download_healthbench_jsonl(
        "2025-05-07-06-14-12_oss_eval.jsonl", bm_dir,
    )
    prompt_rubrics: dict[str, list[dict]] = {}
    for item in eval_data:
        prompt_rubrics[item["prompt_id"]] = item.get("rubrics", [])
    logger.info(
        "HealthBench oss_eval: %d prompts with structured rubrics",
        len(prompt_rubrics),
    )

    meta_data = _download_healthbench_jsonl(
        "2025-05-07-06-14-12_oss_meta_eval.jsonl", bm_dir,
    )
    logger.info("HealthBench meta_eval: %d rows", len(meta_data))

    pair_data: dict[tuple, dict] = defaultdict(
        lambda: {"labels": [], "completion": None, "prompt": None, "pid": None},
    )
    for row in meta_data:
        pid = row["prompt_id"]
        cid = row["completion_id"]
        key = (pid, cid)
        pair_data[key]["completion"] = row["completion"]
        pair_data[key]["prompt"] = row["prompt"]
        pair_data[key]["pid"] = pid
        labels = row["binary_labels"]
        majority = sum(labels) > len(labels) / 2
        pair_data[key]["labels"].append(majority)

    items = []
    for (pid, cid), data in pair_data.items():
        prompt_text = _format_messages(data["prompt"])
        response = data["completion"]
        labels = data["labels"]
        human_score = sum(labels) / len(labels) if labels else 0.0

        items.append(
            {
                "prompt": prompt_text,
                "response": response,
                "human_score": human_score,
            }
        )

    if items:
        logger.info(
            "HealthBench: %d items, score range [%.3f, %.3f]",
            len(items),
            min(it["human_score"] for it in items),
            max(it["human_score"] for it in items),
        )
    return items


# ---------------------------------------------------------------------------
# Benchmark loader registry
# ---------------------------------------------------------------------------

PAIRWISE_LOADERS = {
    "alpacaeval": load_alpacaeval,
    "mtbench": load_mtbench,
    "biggen": load_biggen,
}

def load_biggen_pointwise(cache_dir: Path) -> list[dict]:
    """Load BiGGen Bench as pointwise items with human 1-5 scores.

    Returns list of dicts with keys:
        prompt, response, human_score
    """
    bm_dir = cache_dir / "biggen"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "prometheus-eval/BiGGen-Bench-Results", "human_eval", bm_dir,
    )

    df = df[df["human_score"] > 0].copy()

    items = []
    for _, row in df.iterrows():
        items.append(
            {
                "prompt": str(row["input"]),
                "response": str(row["response"]),
                "human_score": float(row["human_score"]),
            }
        )

    logger.info(
        "BiGGen Pointwise: %d items, score range [%.1f, %.1f]",
        len(items),
        min(it["human_score"] for it in items),
        max(it["human_score"] for it in items),
    )
    return items


POINTWISE_LOADERS = {
    "helpsteer2": load_helpsteer2,
    "profbench": load_profbench,
    "healthbench": load_healthbench,
    "biggen_pointwise": load_biggen_pointwise,
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> list[dict]:
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        logger.info("Resuming from checkpoint -- %d results", len(data))
        return data
    return []


def save_checkpoint(path: Path, results: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Checklist generation
# ---------------------------------------------------------------------------

def generate_checklist(
    client: BedrockClient,
    judge_name: str,
    model_id: str,
    benchmark: str,
    output_dir: Path,
    *,
    num_questions: int = 20,
    temperature: float = 0.0,
) -> list[str]:
    """Generate a checklist of binary Yes/No questions for a benchmark.

    The checklist is cached to disk and reused across runs.
    Returns a list of question strings.
    """
    checklist_file = output_dir / f"checklist_{judge_name}.json"

    if checklist_file.exists():
        with open(checklist_file) as f:
            questions = json.load(f)
        logger.info(
            "[%s] Loaded cached checklist (%d questions) from %s",
            judge_name, len(questions), checklist_file,
        )
        return questions

    description = BENCHMARK_DESCRIPTIONS[benchmark]

    system_prompt = CHECKLIST_GEN_SYSTEM_PROMPT.format(
        num_questions=num_questions,
        benchmark_description=description,
    )
    user_prompt = CHECKLIST_GEN_USER_PROMPT.format(
        num_questions=num_questions,
        benchmark_description=description,
    )

    logger.info(
        "[%s] Generating %d-question checklist for %s (model=%s)",
        judge_name, num_questions, benchmark, model_id,
    )

    raw = call_judge(
        client,
        model_id,
        system=system_prompt,
        user_message=user_prompt,
        max_tokens=1024,
        temperature=temperature,
    )

    # Parse numbered questions from the output
    questions = parse_checklist(raw)

    if not questions:
        logger.warning(
            "[%s] Failed to parse checklist from output: %s",
            judge_name, raw[:200],
        )
        # Fallback: split by newlines and filter
        questions = [
            line.strip()
            for line in raw.strip().split("\n")
            if line.strip() and "?" in line
        ]

    checklist_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checklist_file, "w") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    logger.info(
        "[%s] Generated checklist with %d questions -> %s",
        judge_name, len(questions), checklist_file,
    )
    return questions


def parse_checklist(raw: str) -> list[str]:
    """Parse numbered questions from LLM output.

    Matches patterns like:
        1. Does the response...?
        1) Does the response...?
    """
    pattern = r"^\s*\d+[\.\)]\s*(.+)$"
    questions = []
    for line in raw.strip().split("\n"):
        m = re.match(pattern, line.strip())
        if m:
            q = m.group(1).strip()
            if q:
                questions.append(q)
    return questions


# ---------------------------------------------------------------------------
# CheckEval answer parsing
# ---------------------------------------------------------------------------

def parse_checkeval_answers(raw: str, num_questions: int) -> list[bool | None]:
    """Parse Yes/No answers from CheckEval output.

    Returns a list of booleans (True=Yes, False=No, None=parse error)
    matching the question order.
    """
    answers: list[bool | None] = [None] * num_questions

    # Pattern: Q1: Yes, Q2: No, etc.
    for m in re.finditer(
        r"Q(\d+)\s*[:\.]\s*(Yes|No|yes|no|YES|NO)", raw, re.IGNORECASE,
    ):
        idx = int(m.group(1)) - 1
        if 0 <= idx < num_questions:
            answers[idx] = m.group(2).lower() == "yes"

    # If Q-style parsing found nothing, try line-by-line
    if all(a is None for a in answers):
        lines = raw.strip().split("\n")
        for i, line in enumerate(lines):
            if i >= num_questions:
                break
            line_lower = line.strip().lower()
            if "yes" in line_lower and "no" not in line_lower:
                answers[i] = True
            elif "no" in line_lower and "yes" not in line_lower:
                answers[i] = False

    return answers


def compute_checkeval_score(answers: list[bool | None]) -> float | None:
    """Compute CheckEval score = proportion of 'Yes' answers.

    Returns None if no valid answers were parsed.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return sum(1 for a in valid if a) / len(valid)


# ---------------------------------------------------------------------------
# Format checklist questions for evaluation prompt
# ---------------------------------------------------------------------------

def format_questions(questions: list[str]) -> str:
    """Format checklist questions as Q1: ..., Q2: ..., etc."""
    return "\n".join(
        f"Q{i+1}: {q}" for i, q in enumerate(questions)
    )


# ---------------------------------------------------------------------------
# Pairwise evaluation loop
# ---------------------------------------------------------------------------

def run_pairwise_judge(
    client: BedrockClient,
    judge_name: str,
    judge_config: dict,
    pairs: list[dict],
    questions: list[str],
    output_dir: Path,
    *,
    max_examples: int | None = None,
    checkpoint_every: int = 20,
) -> tuple[Path, dict]:
    """Run CheckEval on pairwise benchmark pairs.

    Each response is scored independently. The higher-scoring response wins.
    """
    if max_examples is not None:
        pairs = pairs[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(pairs):
        logger.info(
            "[%s] Already complete (%d results)", judge_name, start_idx,
        )
        return output_file, compute_pairwise_metrics(results)

    logger.info(
        "[%s] Evaluating pairs %d -> %d (model=%s)",
        judge_name, start_idx, len(pairs), judge_config["model_id"],
    )

    formatted_qs = format_questions(questions)
    num_q = len(questions)

    for i, pair in enumerate(
        tqdm(
            pairs[start_idx:],
            desc=judge_name,
            initial=start_idx,
            total=len(pairs),
        ),
        start=start_idx,
    ):
        instruction = pair["instruction"]

        # Score output_1
        user_msg_1 = CHECKEVAL_USER_TEMPLATE.format(
            prompt=instruction,
            response=pair["output_1"],
            questions=formatted_qs,
        )
        raw_1 = call_judge(
            client,
            judge_config["model_id"],
            system=CHECKEVAL_SYSTEM_PROMPT,
            user_message=user_msg_1,
            max_tokens=judge_config["max_tokens"],
            temperature=judge_config["temperature"],
        )
        answers_1 = parse_checkeval_answers(raw_1, num_q)
        score_1 = compute_checkeval_score(answers_1)

        # Score output_2
        user_msg_2 = CHECKEVAL_USER_TEMPLATE.format(
            prompt=instruction,
            response=pair["output_2"],
            questions=formatted_qs,
        )
        raw_2 = call_judge(
            client,
            judge_config["model_id"],
            system=CHECKEVAL_SYSTEM_PROMPT,
            user_message=user_msg_2,
            max_tokens=judge_config["max_tokens"],
            temperature=judge_config["temperature"],
        )
        answers_2 = parse_checkeval_answers(raw_2, num_q)
        score_2 = compute_checkeval_score(answers_2)

        # Determine preference
        preference = None
        if score_1 is not None and score_2 is not None:
            if score_1 > score_2:
                preference = 1
            elif score_2 > score_1:
                preference = 2
            # tie -> preference stays None (treated as parse error)

        result_entry = {
            "instruction": instruction,
            "score_1": score_1,
            "score_2": score_2,
            "raw_output_1": raw_1,
            "raw_output_2": raw_2,
            "preference": preference,
            "human_majority": pair["human_majority"],
        }
        # Include optional fields
        for key in ["generator", "human_preferences"]:
            if key in pair:
                result_entry[key] = pair[key]

        results.append(result_entry)

        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(output_file, results)
            interim = compute_pairwise_metrics(results)
            logger.info(
                "[%s] Checkpoint %d -- agreement=%.1f%%  errors=%d",
                judge_name, i + 1,
                interim["human_agreement_pct"],
                interim["n_parse_errors"],
            )

    save_checkpoint(output_file, results)
    metrics = compute_pairwise_metrics(results)
    logger.info(
        "[%s] Done -- %d results -> %s",
        judge_name, len(results), output_file,
    )
    return output_file, metrics


# ---------------------------------------------------------------------------
# Pointwise evaluation loop
# ---------------------------------------------------------------------------

def run_pointwise_judge(
    client: BedrockClient,
    judge_name: str,
    judge_config: dict,
    items: list[dict],
    questions: list[str],
    output_dir: Path,
    *,
    max_examples: int | None = None,
    checkpoint_every: int = 20,
) -> tuple[Path, dict]:
    """Run CheckEval on pointwise benchmark items."""
    if max_examples is not None:
        items = items[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(items):
        logger.info(
            "[%s] Already complete (%d results)", judge_name, start_idx,
        )
        return output_file, compute_pointwise_metrics(results)

    logger.info(
        "[%s] Scoring items %d -> %d (model=%s)",
        judge_name, start_idx, len(items), judge_config["model_id"],
    )

    formatted_qs = format_questions(questions)
    num_q = len(questions)

    for i, item in enumerate(
        tqdm(
            items[start_idx:],
            desc=judge_name,
            initial=start_idx,
            total=len(items),
        ),
        start=start_idx,
    ):
        user_msg = CHECKEVAL_USER_TEMPLATE.format(
            prompt=item["prompt"],
            response=item["response"],
            questions=formatted_qs,
        )

        raw = call_judge(
            client,
            judge_config["model_id"],
            system=CHECKEVAL_SYSTEM_PROMPT,
            user_message=user_msg,
            max_tokens=judge_config["max_tokens"],
            temperature=judge_config["temperature"],
        )

        answers = parse_checkeval_answers(raw, num_q)
        score = compute_checkeval_score(answers)

        result_entry = {
            "prompt": item["prompt"],
            "response": item["response"],
            "raw_judge_output": raw,
            "judge_score": score,
            "human_score": item["human_score"],
            "answers": answers,
        }
        # Include optional metadata
        for key in ["task_id", "domain", "response_model"]:
            if key in item:
                result_entry[key] = item[key]

        results.append(result_entry)

        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(output_file, results)
            interim = compute_pointwise_metrics(results)
            logger.info(
                "[%s] Checkpoint %d -- spearman=%s  pearson=%s  errors=%d",
                judge_name, i + 1,
                f'{interim["spearman_corr"]:.4f}'
                if interim["spearman_corr"] is not None else "N/A",
                f'{interim["pearson_corr"]:.4f}'
                if interim["pearson_corr"] is not None else "N/A",
                interim["n_parse_errors"],
            )

    save_checkpoint(output_file, results)
    metrics = compute_pointwise_metrics(results)
    logger.info(
        "[%s] Done -- %d results -> %s",
        judge_name, len(results), output_file,
    )
    return output_file, metrics


# ---------------------------------------------------------------------------
# vLLM-based evaluation (batched)
# ---------------------------------------------------------------------------

def run_pairwise_judge_vllm(
    vllm_judge,
    judge_name: str,
    pairs: list[dict],
    questions: list[str],
    output_dir: Path,
    *,
    max_examples: int | None = None,
    checkpoint_every: int = 20,
    batch_size: int = 32,
) -> tuple[Path, dict]:
    """Run CheckEval on pairwise benchmark pairs using vLLM batched inference."""
    if max_examples is not None:
        pairs = pairs[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(pairs):
        logger.info("[%s] Already complete (%d results)", judge_name, start_idx)
        return output_file, compute_pairwise_metrics(results)

    logger.info(
        "[%s] Evaluating pairs %d -> %d (vLLM, batch_size=%d)",
        judge_name, start_idx, len(pairs), batch_size,
    )

    formatted_qs = format_questions(questions)
    num_q = len(questions)
    remaining_pairs = pairs[start_idx:]

    for batch_start in tqdm(
        range(0, len(remaining_pairs), batch_size // 2),
        desc=judge_name,
        total=(len(remaining_pairs) + batch_size // 2 - 1) // (batch_size // 2),
    ):
        batch = remaining_pairs[batch_start:batch_start + batch_size // 2]

        # Build prompts: 2 per pair (output_1 and output_2)
        prompts = []
        for pair in batch:
            user_msg_1 = CHECKEVAL_USER_TEMPLATE.format(
                prompt=pair["instruction"],
                response=pair["output_1"],
                questions=formatted_qs,
            )
            user_msg_2 = CHECKEVAL_USER_TEMPLATE.format(
                prompt=pair["instruction"],
                response=pair["output_2"],
                questions=formatted_qs,
            )
            prompts.append((CHECKEVAL_SYSTEM_PROMPT, user_msg_1))
            prompts.append((CHECKEVAL_SYSTEM_PROMPT, user_msg_2))

        outputs = vllm_judge.generate_batch(prompts, batch_size=batch_size)

        for j, pair in enumerate(batch):
            raw_1 = outputs[j * 2] or ""
            raw_2 = outputs[j * 2 + 1] or ""

            answers_1 = parse_checkeval_answers(raw_1, num_q)
            score_1 = compute_checkeval_score(answers_1)
            answers_2 = parse_checkeval_answers(raw_2, num_q)
            score_2 = compute_checkeval_score(answers_2)

            preference = None
            if score_1 is not None and score_2 is not None:
                if score_1 > score_2:
                    preference = 1
                elif score_2 > score_1:
                    preference = 2

            result_entry = {
                "instruction": pair["instruction"],
                "score_1": score_1,
                "score_2": score_2,
                "raw_output_1": raw_1,
                "raw_output_2": raw_2,
                "preference": preference,
                "human_majority": pair["human_majority"],
            }
            for key in ["generator", "human_preferences"]:
                if key in pair:
                    result_entry[key] = pair[key]
            results.append(result_entry)

        if len(results) % checkpoint_every == 0 or batch_start + batch_size // 2 >= len(remaining_pairs):
            save_checkpoint(output_file, results)

    metrics = compute_pairwise_metrics(results)
    logger.info("[%s] Done -- %d results -> %s", judge_name, len(results), output_file)
    return output_file, metrics


def run_pointwise_judge_vllm(
    vllm_judge,
    judge_name: str,
    items: list[dict],
    questions: list[str],
    output_dir: Path,
    *,
    max_examples: int | None = None,
    checkpoint_every: int = 20,
    batch_size: int = 32,
) -> tuple[Path, dict]:
    """Run CheckEval on pointwise benchmark items using vLLM batched inference."""
    if max_examples is not None:
        items = items[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(items):
        logger.info("[%s] Already complete (%d results)", judge_name, start_idx)
        return output_file, compute_pointwise_metrics(results)

    logger.info(
        "[%s] Scoring items %d -> %d (vLLM, batch_size=%d)",
        judge_name, start_idx, len(items), batch_size,
    )

    formatted_qs = format_questions(questions)
    num_q = len(questions)
    remaining_items = items[start_idx:]

    for batch_start in tqdm(
        range(0, len(remaining_items), batch_size),
        desc=judge_name,
        total=(len(remaining_items) + batch_size - 1) // batch_size,
    ):
        batch = remaining_items[batch_start:batch_start + batch_size]

        prompts = []
        for item in batch:
            user_msg = CHECKEVAL_USER_TEMPLATE.format(
                prompt=item["prompt"],
                response=item["response"],
                questions=formatted_qs,
            )
            prompts.append((CHECKEVAL_SYSTEM_PROMPT, user_msg))

        outputs = vllm_judge.generate_batch(prompts, batch_size=batch_size)

        for j, item in enumerate(batch):
            raw = outputs[j] or ""
            answers = parse_checkeval_answers(raw, num_q)
            score = compute_checkeval_score(answers)

            result_entry = {
                "prompt": item["prompt"],
                "response": item["response"],
                "raw_judge_output": raw,
                "judge_score": score,
                "human_score": item["human_score"],
                "answers": answers,
            }
            for key in ["task_id", "domain", "response_model"]:
                if key in item:
                    result_entry[key] = item[key]
            results.append(result_entry)

        if len(results) % checkpoint_every == 0 or batch_start + batch_size >= len(remaining_items):
            save_checkpoint(output_file, results)

    metrics = compute_pointwise_metrics(results)
    logger.info("[%s] Done -- %d results -> %s", judge_name, len(results), output_file)
    return output_file, metrics


# ---------------------------------------------------------------------------
# Pairwise metrics
# ---------------------------------------------------------------------------

def compute_pairwise_metrics(results: list[dict]) -> dict:
    """Compute human agreement for pairwise CheckEval results."""
    valid = [r for r in results if r["preference"] is not None]
    n_valid = len(valid)
    n_total = len(results)

    if n_valid == 0:
        return {
            "human_agreement_pct": 0.0,
            "n_valid": 0,
            "n_total": n_total,
            "n_parse_errors": n_total,
            "n_ties": n_total - n_valid,
        }

    n_agree = sum(
        1 for r in valid if r["preference"] == r["human_majority"]
    )
    human_agreement = n_agree / n_valid * 100

    return {
        "human_agreement_pct": round(human_agreement, 2),
        "n_valid": n_valid,
        "n_total": n_total,
        "n_parse_errors": n_total - n_valid,
    }


# ---------------------------------------------------------------------------
# Pointwise metrics
# ---------------------------------------------------------------------------

def compute_pointwise_metrics(results: list[dict]) -> dict:
    """Compute Spearman and Pearson correlations for pointwise results."""
    valid = [r for r in results if r["judge_score"] is not None]
    n_valid = len(valid)
    n_total = len(results)

    if n_valid < 3:
        return {
            "spearman_corr": None,
            "pearson_corr": None,
            "n_valid": n_valid,
            "n_total": n_total,
            "n_parse_errors": n_total - n_valid,
        }

    judge_scores = [r["judge_score"] for r in valid]
    human_scores = [r["human_score"] for r in valid]

    spearman_val = round(spearmanr(judge_scores, human_scores).statistic, 4)
    pearson_val = round(pearsonr(judge_scores, human_scores).statistic, 4)

    return {
        "spearman_corr": spearman_val,
        "pearson_corr": pearson_val,
        "n_valid": n_valid,
        "n_total": n_total,
        "n_parse_errors": n_total - n_valid,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_BENCHMARKS = sorted(PAIRWISE_BENCHMARKS | POINTWISE_BENCHMARKS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judges using CheckEval (binary checklist)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=ALL_BENCHMARKS,
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=list(JUDGES.keys()),
        help="Which judge models to use (default: all). For vLLM backend, use a custom name.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of checklist questions to generate (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs/)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit items to evaluate (useful for testing)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Save progress every N items (default: 20)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data",
        help="Cache dir for dataset files (default: data/)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region (default: us-east-1 or $AWS_REGION)",
    )
    # vLLM backend arguments
    from vllm_judge import add_vllm_args
    add_vllm_args(parser)
    parser.add_argument(
        "--checklist-file",
        type=str,
        default=None,
        help="Path to pre-generated checklist JSON (required for vLLM backend)",
    )
    args = parser.parse_args()

    if args.backend == "vllm" and not args.model:
        parser.error("--model is required when using --backend vllm")

    benchmark = args.benchmark
    is_pairwise = benchmark in PAIRWISE_BENCHMARKS

    # Output: outputs/checkeval/{benchmark}/
    output_dir = Path(args.output_dir) / "checkeval" / benchmark
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bm_type = "pairwise" if is_pairwise else "pointwise"
    logger.info(
        "Benchmark: %s (%s) | Output: %s",
        benchmark, bm_type, output_dir,
    )

    # 1. Load data
    if is_pairwise:
        data = PAIRWISE_LOADERS[benchmark](cache_dir)
    else:
        data = POINTWISE_LOADERS[benchmark](cache_dir)

    # 2. Initialize backend
    all_metrics = {}

    if args.backend == "vllm":
        from vllm_judge import VLLMJudge

        vllm_judge = VLLMJudge(
            model_path=args.model,
            max_new_tokens=512,
            temperature=0.0,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

        for judge_name in args.judges:
            # Load or generate checklist
            if args.checklist_file:
                with open(args.checklist_file) as f:
                    questions = json.load(f)
                logger.info("Loaded checklist from %s (%d questions)", args.checklist_file, len(questions))
            else:
                # Generate checklist using vLLM
                checklist_file = output_dir / f"checklist_{judge_name}.json"
                if checklist_file.exists():
                    with open(checklist_file) as f:
                        questions = json.load(f)
                    logger.info("Loaded cached checklist (%d questions)", len(questions))
                else:
                    description = BENCHMARK_DESCRIPTIONS[benchmark]
                    system_prompt = CHECKLIST_GEN_SYSTEM_PROMPT.format(
                        num_questions=args.num_questions,
                        benchmark_description=description,
                    )
                    user_prompt = CHECKLIST_GEN_USER_PROMPT.format(
                        num_questions=args.num_questions,
                        benchmark_description=description,
                    )
                    raw = vllm_judge.generate(system_prompt, user_prompt)
                    questions = parse_checklist(raw)
                    if questions:
                        with open(checklist_file, "w") as f:
                            json.dump(questions, f, indent=2)
                        logger.info("Generated and cached %d questions", len(questions))
                    else:
                        logger.error("Failed to generate checklist via vLLM")
                        return

            if is_pairwise:
                _, metrics = run_pairwise_judge_vllm(
                    vllm_judge, judge_name, data, questions, output_dir,
                    max_examples=args.max_examples,
                    checkpoint_every=args.checkpoint_every,
                    batch_size=args.batch_size,
                )
            else:
                _, metrics = run_pointwise_judge_vllm(
                    vllm_judge, judge_name, data, questions, output_dir,
                    max_examples=args.max_examples,
                    checkpoint_every=args.checkpoint_every,
                    batch_size=args.batch_size,
                )
            all_metrics[judge_name] = metrics

    else:
        # Bedrock API backend
        client = BedrockClient(region=args.region)

        for judge_name in args.judges:
            judge_config = JUDGES[judge_name]

            # Generate checklist (per benchmark, per judge)
            questions = generate_checklist(
                client,
                judge_name,
                judge_config["model_id"],
                benchmark,
                output_dir,
                num_questions=args.num_questions,
                temperature=judge_config["temperature"],
            )

            # Run evaluation
            if is_pairwise:
                _, metrics = run_pairwise_judge(
                    client,
                    judge_name,
                    judge_config,
                    data,
                    questions,
                    output_dir,
                    max_examples=args.max_examples,
                    checkpoint_every=args.checkpoint_every,
                )
            else:
                _, metrics = run_pointwise_judge(
                    client,
                    judge_name,
                    judge_config,
                    data,
                    questions,
                    output_dir,
                    max_examples=args.max_examples,
                    checkpoint_every=args.checkpoint_every,
                )

            all_metrics[judge_name] = metrics

    # 4. Save and display results
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("=" * 70)
    logger.info("CheckEval Results -- %s (%s)", benchmark, bm_type)

    if is_pairwise:
        logger.info(
            "%-20s  %10s  %5s  %6s",
            "Judge", "Agreement", "Valid", "Errors",
        )
        logger.info("-" * 50)
        for name, m in all_metrics.items():
            logger.info(
                "%-20s  %9.2f%%  %5d  %6d",
                name,
                m["human_agreement_pct"],
                m["n_valid"],
                m["n_parse_errors"],
            )
    else:
        logger.info(
            "%-20s  %10s  %10s  %5s  %6s",
            "Judge", "Spearman", "Pearson", "Valid", "Errors",
        )
        logger.info("-" * 70)
        for name, m in all_metrics.items():
            logger.info(
                "%-20s  %10s  %10s  %5d  %6d",
                name,
                f'{m["spearman_corr"]:.4f}'
                if m["spearman_corr"] is not None else "N/A",
                f'{m["pearson_corr"]:.4f}'
                if m["pearson_corr"] is not None else "N/A",
                m["n_valid"],
                m["n_parse_errors"],
            )

    logger.info("=" * 70)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
