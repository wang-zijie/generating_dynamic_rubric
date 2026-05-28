#!/usr/bin/env python3
"""
Evaluate LLM judges using the RubricHub methodology (coarse-to-fine rubric generation).

RubricHub generates fine-grained, weighted binary rubric criteria per instruction,
then grades each response criterion-by-criterion. The pipeline:
  1. Generates a reference response for the instruction (1 LLM call, cached)
  2. Generates weighted rubric criteria grounded in the reference (1 LLM call, cached)
  3. Grades the response on each criterion (N LLM calls, one per criterion)
  4. Aggregates via weighted sum: score = sum(w_i * b_i) / sum(w_i)

For pairwise benchmarks, each response is scored independently and the higher
score wins.

Benchmarks:
    Pairwise:  AlpacaEval, MTBench, BiGGen Bench
    Pointwise: HelpSteer2, ProfBench, HealthBench

Reference responses and rubrics are cached per instruction to avoid redundant
calls when the same instruction appears in multiple pairs.

Reference:
    Zhu et al. (2025). RubricHub: Coarse-to-Fine Automated Rubric Generation
    with LLMs. arXiv:2601.08430.

Authentication (environment variables):
    AWS_BEARER_TOKEN_BEDROCK  -- Bearer token for Bedrock API
    AWS_REGION                -- AWS region (default: us-east-1)

Usage:
    # Pairwise: AlpacaEval
    python src/eval_rubrichub.py --benchmark alpacaeval --judges claude-sonnet-4

    # Pointwise: HelpSteer2
    python src/eval_rubrichub.py --benchmark helpsteer2 --judges llama-3.1-8b

    # Quick test
    python src/eval_rubrichub.py --benchmark helpsteer2 --judges llama-3.1-8b \\
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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# RubricHub prompt templates
# ---------------------------------------------------------------------------

# Stage 0: Reference response generation
REFERENCE_GEN_SYSTEM = (
    "You are a helpful AI assistant. Provide a high-quality, comprehensive "
    "response to the user's request."
)

REFERENCE_GEN_USER = """\
{instruction}"""

# Stage 1: Principle-Guided & Response-Grounded Rubric Generation
RUBRIC_GEN_SYSTEM = """\
You are a top-tier Rubric Designer. Your task is to create a set of \
evaluation criteria (rubric) for assessing the quality of an AI assistant's \
response to a given question.

You will be provided with:
1. A Question (the user's prompt)
2. A Reference Answer (an authoritative response to the question)

Your task:
1. Analyze the Question carefully.
2. Leverage the Reference Answer as authoritative context for what a good \
response should contain.
3. Create 3-15 evaluation criteria.
4. Output JSON only.

Each criterion must have:
- "title": A 2-5 word core summary (String)
- "description": A clear description, max 40 words (String). Must be \
phrased so it can be verified as True/False.
- "weight": An integer score between 1 and 10 indicating importance

Design Rules:
1. Instruction & Reference Alignment: Criteria must reflect what the \
question asks and what the reference answer demonstrates.
2. Atomicity and Independence: Each criterion covers exactly one distinct \
aspect. No overlap between criteria.
3. Clarity and Verifiability: Each criterion must be objectively verifiable \
as met or not met (binary True/False).
4. Specificity and Contextualization: Criteria should be specific to this \
question, not generic quality standards.
5. Information Completeness: Cover key information points from the reference.
6. Balance: Cover accuracy, completeness, structure, safety as relevant.

Output ONLY a JSON array wrapped in a Markdown code block. No other text."""

RUBRIC_GEN_USER = """\
[Question]
{instruction}

[Reference Answer]
{reference}"""

# Stage 3: Grader prompt (per-criterion binary evaluation)
GRADER_SYSTEM = """\
You are an expert evaluator. You will be given a conversation (a user \
prompt and an AI response) and a single rubric criterion. Your task is to \
determine whether the response meets the criterion.

Rules:
- If a criterion uses phrases like "such as", "for example", or \
"including", the response does NOT need to cover all listed examples to \
pass. It only needs to demonstrate the core concept.
- Be strict but fair in your assessment.

Output ONLY a JSON object with two fields:
- "explanation": A brief (1-2 sentence) justification
- "criteria_met": A boolean (true or false)"""

GRADER_USER = """\
[User Prompt]
{instruction}

[AI Response]
{response}

[Criterion]
Title: {title}
Description: {description}

Does the response meet this criterion? Output JSON only."""


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
    """Load AlpacaEval cross-annotations as pairwise evaluation pairs."""
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
    """Load MTBench human judgments as pairwise evaluation pairs."""
    bm_dir = cache_dir / "mtbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "lmsys/mt_bench_human_judgments", "human", bm_dir,
    )
    logger.info("MTBench columns: %s", list(df.columns))
    logger.info("MTBench shape: %s", df.shape)

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
            conv_a = row["conversation_a"]
            conv_b = row["conversation_b"]

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
    """Load BiGGen Bench as synthetic pairwise evaluation pairs."""
    bm_dir = cache_dir / "biggen"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "prometheus-eval/BiGGen-Bench-Results", "human_eval", bm_dir,
    )
    logger.info("BiGGen Bench columns: %s", list(df.columns))
    logger.info("BiGGen Bench shape: %s", df.shape)

    prompt_data: dict[str, list[dict]] = defaultdict(list)

    for _, row in df.iterrows():
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
    """Load HelpSteer2 validation split."""
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
    """Load ProfBench dataset."""
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
    """Load HealthBench dataset."""
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
# Checkpoint / cache helpers
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


def load_cache(path: Path) -> dict:
    """Load a JSON dict cache file (instruction -> value)."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(path: Path, data: dict) -> None:
    """Save a JSON dict cache file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# RubricHub: Reference response generation (cached per instruction)
# ---------------------------------------------------------------------------

def generate_reference(
    client: BedrockClient,
    model_id: str,
    instruction: str,
    reference_cache: dict,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Generate a reference response for an instruction. Cached."""
    if instruction in reference_cache:
        return reference_cache[instruction]

    raw = call_judge(
        client,
        model_id,
        system=REFERENCE_GEN_SYSTEM,
        user_message=REFERENCE_GEN_USER.format(instruction=instruction),
        max_tokens=max_tokens,
        temperature=temperature,
    )

    reference_cache[instruction] = raw
    return raw


# ---------------------------------------------------------------------------
# RubricHub: Rubric generation (cached per instruction)
# ---------------------------------------------------------------------------

def parse_rubric(raw: str) -> list[dict]:
    """Parse rubric JSON from LLM output.

    Extracts a JSON array from a code block or raw text.
    Each item should have title, description, weight.
    """
    # Try to extract from code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    json_str = m.group(1).strip() if m else raw.strip()

    # Try to find a JSON array
    bracket_start = json_str.find("[")
    bracket_end = json_str.rfind("]")
    if bracket_start != -1 and bracket_end != -1:
        json_str = json_str[bracket_start:bracket_end + 1]

    try:
        criteria = json.loads(json_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse rubric JSON: %s", raw[:300])
        return []

    if not isinstance(criteria, list):
        return []

    validated = []
    for c in criteria:
        if not isinstance(c, dict):
            continue
        title = c.get("title", "")
        description = c.get("description", "")
        weight = c.get("weight", 5)
        if isinstance(weight, str):
            try:
                weight = int(weight)
            except ValueError:
                weight = 5
        weight = max(1, min(10, int(weight)))
        if title and description:
            validated.append(
                {"title": title, "description": description, "weight": weight}
            )

    return validated


def generate_rubric(
    client: BedrockClient,
    model_id: str,
    instruction: str,
    reference: str,
    rubric_cache: dict,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> list[dict]:
    """Generate rubric criteria for an instruction. Cached."""
    if instruction in rubric_cache:
        return rubric_cache[instruction]

    user_msg = RUBRIC_GEN_USER.format(
        instruction=instruction,
        reference=reference,
    )

    raw = call_judge(
        client,
        model_id,
        system=RUBRIC_GEN_SYSTEM,
        user_message=user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    criteria = parse_rubric(raw)

    if not criteria:
        logger.warning(
            "No criteria parsed for instruction: %s...", instruction[:80],
        )
        # Fallback: single generic criterion
        criteria = [
            {
                "title": "Overall Quality",
                "description": "The response adequately addresses the user's request with accurate and helpful information.",
                "weight": 5,
            }
        ]

    rubric_cache[instruction] = criteria
    return criteria


# ---------------------------------------------------------------------------
# RubricHub: Per-criterion grading
# ---------------------------------------------------------------------------

def parse_grader_output(raw: str) -> bool | None:
    """Parse the grader's JSON output to extract criteria_met boolean."""
    # Try JSON parsing
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            val = obj.get("criteria_met")
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ("true", "yes")
        except json.JSONDecodeError:
            pass

    # Fallback: look for true/false keywords
    lower = raw.lower()
    if '"criteria_met": true' in lower or '"criteria_met":true' in lower:
        return True
    if '"criteria_met": false' in lower or '"criteria_met":false' in lower:
        return False

    return None


def grade_criterion(
    client: BedrockClient,
    model_id: str,
    instruction: str,
    response: str,
    criterion: dict,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[bool | None, str]:
    """Grade a single criterion against a response.

    Returns (criteria_met, raw_output).
    """
    user_msg = GRADER_USER.format(
        instruction=instruction,
        response=response,
        title=criterion["title"],
        description=criterion["description"],
    )

    raw = call_judge(
        client,
        model_id,
        system=GRADER_SYSTEM,
        user_message=user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    met = parse_grader_output(raw)
    return met, raw


# ---------------------------------------------------------------------------
# RubricHub: Score aggregation
# ---------------------------------------------------------------------------

def compute_rubrichub_score(
    criteria: list[dict],
    grades: list[bool | None],
) -> float | None:
    """Compute weighted score = sum(w_i * b_i) / sum(w_i).

    Returns None if no valid grades.
    """
    total_weight = 0
    weighted_sum = 0
    for c, g in zip(criteria, grades):
        if g is None:
            continue
        w = c["weight"]
        total_weight += w
        if g:
            weighted_sum += w

    if total_weight == 0:
        return None
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Pairwise evaluation loop
# ---------------------------------------------------------------------------

def run_pairwise_judge(
    client: BedrockClient,
    judge_name: str,
    judge_config: dict,
    pairs: list[dict],
    output_dir: Path,
    *,
    max_examples: int | None = None,
    checkpoint_every: int = 10,
    max_workers: int = 4,
) -> tuple[Path, dict]:
    """Run RubricHub on pairwise benchmark pairs.

    Each response is scored independently against the generated rubric.
    The higher-scoring response wins. Uses multithreading for grading.
    """
    if max_examples is not None:
        pairs = pairs[:max_examples]

    model_id = judge_config["model_id"]
    temperature = judge_config["temperature"]

    output_file = output_dir / f"judge_{judge_name}.json"
    reference_file = output_dir / f"references_{judge_name}.json"
    rubric_file = output_dir / f"rubrics_{judge_name}.json"

    results = load_checkpoint(output_file)
    reference_cache = load_cache(reference_file)
    rubric_cache = load_cache(rubric_file)
    start_idx = len(results)

    if start_idx >= len(pairs):
        logger.info(
            "[%s] Already complete (%d results)", judge_name, start_idx,
        )
        return output_file, compute_pairwise_metrics(results)

    logger.info(
        "[%s] Evaluating pairs %d -> %d (model=%s, workers=%d)",
        judge_name, start_idx, len(pairs), model_id, max_workers,
    )

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

        # Step 1: Generate reference response (cached)
        reference = generate_reference(
            client, model_id, instruction, reference_cache,
            max_tokens=1024,
            temperature=temperature,
        )

        # Step 2: Generate rubric criteria (cached)
        criteria = generate_rubric(
            client, model_id, instruction, reference, rubric_cache,
            max_tokens=2048,
            temperature=temperature,
        )

        # Step 3+4: Grade both outputs on all criteria in parallel
        grades_1 = [None] * len(criteria)
        raw_grades_1 = [None] * len(criteria)
        grades_2 = [None] * len(criteria)
        raw_grades_2 = [None] * len(criteria)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {}
            for c_idx, criterion in enumerate(criteria):
                f1 = executor.submit(
                    grade_criterion,
                    client, model_id, instruction, pair["output_1"], criterion,
                    max_tokens=256, temperature=temperature,
                )
                future_to_key[f1] = (1, c_idx)
                f2 = executor.submit(
                    grade_criterion,
                    client, model_id, instruction, pair["output_2"], criterion,
                    max_tokens=256, temperature=temperature,
                )
                future_to_key[f2] = (2, c_idx)

            for future in as_completed(future_to_key):
                out_num, c_idx = future_to_key[future]
                met, raw = future.result()
                if out_num == 1:
                    grades_1[c_idx] = met
                    raw_grades_1[c_idx] = raw
                else:
                    grades_2[c_idx] = met
                    raw_grades_2[c_idx] = raw

        # Step 5: Aggregate scores
        score_1 = compute_rubrichub_score(criteria, grades_1)
        score_2 = compute_rubrichub_score(criteria, grades_2)

        preference = None
        if score_1 is not None and score_2 is not None:
            if score_1 > score_2:
                preference = 1
            elif score_2 > score_1:
                preference = 2

        result_entry = {
            "instruction": instruction,
            "criteria": criteria,
            "grades_1": grades_1,
            "grades_2": grades_2,
            "raw_grades_1": raw_grades_1,
            "raw_grades_2": raw_grades_2,
            "score_1": score_1,
            "score_2": score_2,
            "preference": preference,
            "human_majority": pair["human_majority"],
        }
        for key in ["generator", "human_preferences"]:
            if key in pair:
                result_entry[key] = pair[key]

        results.append(result_entry)

        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(output_file, results)
            save_cache(reference_file, reference_cache)
            save_cache(rubric_file, rubric_cache)
            interim = compute_pairwise_metrics(results)
            logger.info(
                "[%s] Checkpoint %d -- agreement=%.1f%%  errors=%d",
                judge_name, i + 1,
                interim["human_agreement_pct"],
                interim["n_parse_errors"],
            )

    save_checkpoint(output_file, results)
    save_cache(reference_file, reference_cache)
    save_cache(rubric_file, rubric_cache)
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
    output_dir: Path,
    *,
    max_examples: int | None = None,
    checkpoint_every: int = 10,
    max_workers: int = 4,
) -> tuple[Path, dict]:
    """Run RubricHub on pointwise benchmark items. Uses multithreading for grading."""
    if max_examples is not None:
        items = items[:max_examples]

    model_id = judge_config["model_id"]
    temperature = judge_config["temperature"]

    output_file = output_dir / f"judge_{judge_name}.json"
    reference_file = output_dir / f"references_{judge_name}.json"
    rubric_file = output_dir / f"rubrics_{judge_name}.json"

    results = load_checkpoint(output_file)
    reference_cache = load_cache(reference_file)
    rubric_cache = load_cache(rubric_file)
    start_idx = len(results)

    if start_idx >= len(items):
        logger.info(
            "[%s] Already complete (%d results)", judge_name, start_idx,
        )
        return output_file, compute_pointwise_metrics(results)

    logger.info(
        "[%s] Scoring items %d -> %d (model=%s, workers=%d)",
        judge_name, start_idx, len(items), model_id, max_workers,
    )

    for i, item in enumerate(
        tqdm(
            items[start_idx:],
            desc=judge_name,
            initial=start_idx,
            total=len(items),
        ),
        start=start_idx,
    ):
        instruction = item["prompt"]

        # Step 1: Generate reference response (cached)
        reference = generate_reference(
            client, model_id, instruction, reference_cache,
            max_tokens=1024,
            temperature=temperature,
        )

        # Step 2: Generate rubric criteria (cached)
        criteria = generate_rubric(
            client, model_id, instruction, reference, rubric_cache,
            max_tokens=2048,
            temperature=temperature,
        )

        # Step 3: Grade response on all criteria in parallel
        grades = [None] * len(criteria)
        raw_grades = [None] * len(criteria)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for c_idx, criterion in enumerate(criteria):
                future = executor.submit(
                    grade_criterion,
                    client, model_id, instruction, item["response"], criterion,
                    max_tokens=256, temperature=temperature,
                )
                future_to_idx[future] = c_idx

            for future in as_completed(future_to_idx):
                c_idx = future_to_idx[future]
                met, raw = future.result()
                grades[c_idx] = met
                raw_grades[c_idx] = raw

        # Step 4: Aggregate
        judge_score = compute_rubrichub_score(criteria, grades)

        result_entry = {
            "prompt": item["prompt"],
            "response": item["response"],
            "criteria": criteria,
            "grades": grades,
            "raw_grades": raw_grades,
            "judge_score": judge_score,
            "human_score": item["human_score"],
        }
        for key in ["task_id", "domain", "response_model"]:
            if key in item:
                result_entry[key] = item[key]

        results.append(result_entry)

        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(output_file, results)
            save_cache(reference_file, reference_cache)
            save_cache(rubric_file, rubric_cache)
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
    save_cache(reference_file, reference_cache)
    save_cache(rubric_file, rubric_cache)
    metrics = compute_pointwise_metrics(results)
    logger.info(
        "[%s] Done -- %d results -> %s",
        judge_name, len(results), output_file,
    )
    return output_file, metrics


# ---------------------------------------------------------------------------
# vLLM-based evaluation
# ---------------------------------------------------------------------------

def run_pairwise_judge_vllm(
    vllm_judge,
    judge_name: str,
    pairs: list[dict],
    output_dir: Path,
    *,
    max_examples: int | None = None,
    checkpoint_every: int = 10,
    batch_size: int = 32,
) -> tuple[Path, dict]:
    """Run RubricHub on pairwise benchmark pairs using vLLM."""
    if max_examples is not None:
        pairs = pairs[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    reference_file = output_dir / f"references_{judge_name}.json"
    rubric_file = output_dir / f"rubrics_{judge_name}.json"

    results = load_checkpoint(output_file)
    reference_cache = load_cache(reference_file)
    rubric_cache = load_cache(rubric_file)
    start_idx = len(results)

    if start_idx >= len(pairs):
        logger.info("[%s] Already complete (%d results)", judge_name, start_idx)
        return output_file, compute_pairwise_metrics(results)

    logger.info(
        "[%s] Evaluating pairs %d -> %d (vLLM, batch_size=%d)",
        judge_name, start_idx, len(pairs), batch_size,
    )

    remaining_pairs = pairs[start_idx:]
    unique_instructions = list(dict.fromkeys(p["instruction"] for p in remaining_pairs))

    # Batch generate references for uncached instructions
    uncached_refs = [inst for inst in unique_instructions if inst not in reference_cache]
    if uncached_refs:
        logger.info("Generating references for %d instructions...", len(uncached_refs))
        ref_prompts = [(REFERENCE_GEN_SYSTEM, REFERENCE_GEN_USER.format(instruction=inst)) for inst in uncached_refs]
        ref_outputs = vllm_judge.generate_batch(ref_prompts, batch_size=batch_size)
        for inst, raw in zip(uncached_refs, ref_outputs):
            reference_cache[inst] = raw or ""
        save_cache(reference_file, reference_cache)

    # Batch generate rubrics for uncached instructions
    uncached_rubrics = [inst for inst in unique_instructions if inst not in rubric_cache]
    if uncached_rubrics:
        logger.info("Generating rubrics for %d instructions...", len(uncached_rubrics))
        rubric_prompts = []
        for inst in uncached_rubrics:
            reference = reference_cache.get(inst, "")
            user_msg = RUBRIC_GEN_USER.format(instruction=inst, reference=reference)
            rubric_prompts.append((RUBRIC_GEN_SYSTEM, user_msg))
        rubric_outputs = vllm_judge.generate_batch(rubric_prompts, batch_size=batch_size)
        for inst, raw in zip(uncached_rubrics, rubric_outputs):
            if raw:
                criteria = parse_rubric(raw)
                if not criteria:
                    criteria = [{"title": "Overall Quality", "description": "The response adequately addresses the user's request.", "weight": 5}]
                rubric_cache[inst] = criteria
        save_cache(rubric_file, rubric_cache)

    # Batch grading per batch of pairs
    for batch_start in tqdm(
        range(0, len(remaining_pairs), batch_size),
        desc=judge_name,
        total=(len(remaining_pairs) + batch_size - 1) // batch_size,
    ):
        batch = remaining_pairs[batch_start:batch_start + batch_size]

        # Build all grading prompts for this batch (2 outputs × N criteria per pair)
        grading_prompts = []
        prompt_map = []  # (pair_idx, output_num, criterion_idx)
        for j, pair in enumerate(batch):
            instruction = pair["instruction"]
            criteria = rubric_cache.get(instruction, [])
            for c_idx, criterion in enumerate(criteria):
                for out_num, response in [(1, pair["output_1"]), (2, pair["output_2"])]:
                    user_msg = GRADER_USER.format(
                        instruction=instruction,
                        response=response,
                        title=criterion["title"],
                        description=criterion["description"],
                    )
                    grading_prompts.append((GRADER_SYSTEM, user_msg))
                    prompt_map.append((j, out_num, c_idx))

        grading_outputs = vllm_judge.generate_batch(grading_prompts, batch_size=batch_size * 4)

        # Reconstruct results
        pair_grades = {}  # (j, out_num) -> list of (met, raw)
        for (j, out_num, c_idx), raw in zip(prompt_map, grading_outputs):
            key = (j, out_num)
            if key not in pair_grades:
                criteria = rubric_cache.get(batch[j]["instruction"], [])
                pair_grades[key] = [None] * len(criteria)
            pair_grades[key][c_idx] = (parse_grader_output(raw or ""), raw or "")

        for j, pair in enumerate(batch):
            instruction = pair["instruction"]
            criteria = rubric_cache.get(instruction, [])

            grades_1 = [g[0] for g in (pair_grades.get((j, 1), []) or [(None, "")] * len(criteria))]
            raw_grades_1 = [g[1] for g in (pair_grades.get((j, 1), []) or [(None, "")] * len(criteria))]
            grades_2 = [g[0] for g in (pair_grades.get((j, 2), []) or [(None, "")] * len(criteria))]
            raw_grades_2 = [g[1] for g in (pair_grades.get((j, 2), []) or [(None, "")] * len(criteria))]

            score_1 = compute_rubrichub_score(criteria, grades_1)
            score_2 = compute_rubrichub_score(criteria, grades_2)

            preference = None
            if score_1 is not None and score_2 is not None:
                if score_1 > score_2:
                    preference = 1
                elif score_2 > score_1:
                    preference = 2

            result_entry = {
                "instruction": instruction,
                "criteria": criteria,
                "grades_1": grades_1,
                "grades_2": grades_2,
                "raw_grades_1": raw_grades_1,
                "raw_grades_2": raw_grades_2,
                "score_1": score_1,
                "score_2": score_2,
                "preference": preference,
                "human_majority": pair["human_majority"],
            }
            for key in ["generator", "human_preferences"]:
                if key in pair:
                    result_entry[key] = pair[key]
            results.append(result_entry)

        if len(results) % checkpoint_every == 0 or batch_start + batch_size >= len(remaining_pairs):
            save_checkpoint(output_file, results)

    save_cache(reference_file, reference_cache)
    save_cache(rubric_file, rubric_cache)
    metrics = compute_pairwise_metrics(results)
    logger.info("[%s] Done -- %d results -> %s", judge_name, len(results), output_file)
    return output_file, metrics


def run_pointwise_judge_vllm(
    vllm_judge,
    judge_name: str,
    items: list[dict],
    output_dir: Path,
    *,
    max_examples: int | None = None,
    checkpoint_every: int = 10,
    batch_size: int = 32,
) -> tuple[Path, dict]:
    """Run RubricHub on pointwise benchmark items using vLLM."""
    if max_examples is not None:
        items = items[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    reference_file = output_dir / f"references_{judge_name}.json"
    rubric_file = output_dir / f"rubrics_{judge_name}.json"

    results = load_checkpoint(output_file)
    reference_cache = load_cache(reference_file)
    rubric_cache = load_cache(rubric_file)
    start_idx = len(results)

    if start_idx >= len(items):
        logger.info("[%s] Already complete (%d results)", judge_name, start_idx)
        return output_file, compute_pointwise_metrics(results)

    logger.info(
        "[%s] Scoring items %d -> %d (vLLM, batch_size=%d)",
        judge_name, start_idx, len(items), batch_size,
    )

    remaining_items = items[start_idx:]
    unique_instructions = list(dict.fromkeys(it["prompt"] for it in remaining_items))

    # Batch generate references
    uncached_refs = [inst for inst in unique_instructions if inst not in reference_cache]
    if uncached_refs:
        logger.info("Generating references for %d instructions...", len(uncached_refs))
        ref_prompts = [(REFERENCE_GEN_SYSTEM, REFERENCE_GEN_USER.format(instruction=inst)) for inst in uncached_refs]
        ref_outputs = vllm_judge.generate_batch(ref_prompts, batch_size=batch_size)
        for inst, raw in zip(uncached_refs, ref_outputs):
            reference_cache[inst] = raw or ""
        save_cache(reference_file, reference_cache)

    # Batch generate rubrics
    uncached_rubrics = [inst for inst in unique_instructions if inst not in rubric_cache]
    if uncached_rubrics:
        logger.info("Generating rubrics for %d instructions...", len(uncached_rubrics))
        rubric_prompts = []
        for inst in uncached_rubrics:
            reference = reference_cache.get(inst, "")
            user_msg = RUBRIC_GEN_USER.format(instruction=inst, reference=reference)
            rubric_prompts.append((RUBRIC_GEN_SYSTEM, user_msg))
        rubric_outputs = vllm_judge.generate_batch(rubric_prompts, batch_size=batch_size)
        for inst, raw in zip(uncached_rubrics, rubric_outputs):
            if raw:
                criteria = parse_rubric(raw)
                if not criteria:
                    criteria = [{"title": "Overall Quality", "description": "The response adequately addresses the user's request.", "weight": 5}]
                rubric_cache[inst] = criteria
        save_cache(rubric_file, rubric_cache)

    # Batch grading
    for batch_start in tqdm(
        range(0, len(remaining_items), batch_size),
        desc=judge_name,
        total=(len(remaining_items) + batch_size - 1) // batch_size,
    ):
        batch = remaining_items[batch_start:batch_start + batch_size]

        grading_prompts = []
        prompt_map = []  # (item_idx, criterion_idx)
        for j, item in enumerate(batch):
            instruction = item["prompt"]
            criteria = rubric_cache.get(instruction, [])
            for c_idx, criterion in enumerate(criteria):
                user_msg = GRADER_USER.format(
                    instruction=instruction,
                    response=item["response"],
                    title=criterion["title"],
                    description=criterion["description"],
                )
                grading_prompts.append((GRADER_SYSTEM, user_msg))
                prompt_map.append((j, c_idx))

        grading_outputs = vllm_judge.generate_batch(grading_prompts, batch_size=batch_size * 4)

        item_grades = {}  # j -> list of (met, raw)
        for (j, c_idx), raw in zip(prompt_map, grading_outputs):
            if j not in item_grades:
                criteria = rubric_cache.get(batch[j]["prompt"], [])
                item_grades[j] = [None] * len(criteria)
            item_grades[j][c_idx] = (parse_grader_output(raw or ""), raw or "")

        for j, item in enumerate(batch):
            instruction = item["prompt"]
            criteria = rubric_cache.get(instruction, [])

            grade_list = item_grades.get(j, [(None, "")] * len(criteria))
            grades = [g[0] for g in grade_list]
            raw_grades = [g[1] for g in grade_list]

            judge_score = compute_rubrichub_score(criteria, grades)

            result_entry = {
                "prompt": item["prompt"],
                "response": item["response"],
                "criteria": criteria,
                "grades": grades,
                "raw_grades": raw_grades,
                "judge_score": judge_score,
                "human_score": item["human_score"],
            }
            for key in ["task_id", "domain", "response_model"]:
                if key in item:
                    result_entry[key] = item[key]
            results.append(result_entry)

        if len(results) % checkpoint_every == 0 or batch_start + batch_size >= len(remaining_items):
            save_checkpoint(output_file, results)

    save_cache(reference_file, reference_cache)
    save_cache(rubric_file, rubric_cache)
    metrics = compute_pointwise_metrics(results)
    logger.info("[%s] Done -- %d results -> %s", judge_name, len(results), output_file)
    return output_file, metrics


# ---------------------------------------------------------------------------
# Pairwise metrics
# ---------------------------------------------------------------------------

def compute_pairwise_metrics(results: list[dict]) -> dict:
    """Compute human agreement for pairwise RubricHub results."""
    valid = [r for r in results if r["preference"] is not None]
    n_valid = len(valid)
    n_total = len(results)

    if n_valid == 0:
        return {
            "human_agreement_pct": 0.0,
            "n_valid": 0,
            "n_total": n_total,
            "n_parse_errors": n_total,
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
        description="Evaluate LLM judges using RubricHub (coarse-to-fine rubric generation)",
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
        default=10,
        help="Save progress every N items (default: 10)",
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
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel threads for Bedrock API grading calls (default: 4)",
    )
    # vLLM backend arguments
    from vllm_judge import add_vllm_args
    add_vllm_args(parser)
    args = parser.parse_args()

    if args.backend == "vllm" and not args.model:
        parser.error("--model is required when using --backend vllm")

    benchmark = args.benchmark
    is_pairwise = benchmark in PAIRWISE_BENCHMARKS

    # Output: outputs/rubrichub/{benchmark}/
    output_dir = Path(args.output_dir) / "rubrichub" / benchmark
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

    # 2. Initialize backend and run judges
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
            if is_pairwise:
                _, metrics = run_pairwise_judge_vllm(
                    vllm_judge, judge_name, data, output_dir,
                    max_examples=args.max_examples,
                    checkpoint_every=args.checkpoint_every,
                    batch_size=args.batch_size,
                )
            else:
                _, metrics = run_pointwise_judge_vllm(
                    vllm_judge, judge_name, data, output_dir,
                    max_examples=args.max_examples,
                    checkpoint_every=args.checkpoint_every,
                    batch_size=args.batch_size,
                )
            all_metrics[judge_name] = metrics

    else:
        client = BedrockClient(region=args.region)

        for judge_name in args.judges:
            judge_config = JUDGES[judge_name]

            if is_pairwise:
                _, metrics = run_pairwise_judge(
                    client,
                    judge_name,
                    judge_config,
                    data,
                    output_dir,
                    max_examples=args.max_examples,
                    checkpoint_every=args.checkpoint_every,
                    max_workers=args.max_workers,
                )
            else:
                _, metrics = run_pointwise_judge(
                    client,
                    judge_name,
                    judge_config,
                    data,
                    output_dir,
                    max_examples=args.max_examples,
                    checkpoint_every=args.checkpoint_every,
                    max_workers=args.max_workers,
                )

            all_metrics[judge_name] = metrics

    # 4. Save and display results
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("=" * 70)
    logger.info("RubricHub Results -- %s (%s)", benchmark, bm_type)

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
