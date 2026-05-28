#!/usr/bin/env python3
"""
Evaluate LLM judges on pointwise benchmarks: HelpSteer2, ProfBench, HealthBench.

Each judge model scores individual AI responses on a 1-5 scale, and we measure
how well these scores correlate with human scores using Spearman and Pearson
correlation coefficients.

Judge models (via AWS Bedrock):
    - Claude Sonnet 4
    - Llama 3.1 8B Instruct
    - Llama 3.1 70B Instruct

Rubric modes:
    - fixed:           Hardcoded benchmark-specific rubric (same for all instances)
    - fixed_instance:  Per-instance human-authored rubrics from the benchmark
    - generated_fixed: LLM generates one general-purpose rubric for all instances
    - dynamic:         LLM generates a per-instance rubric from the prompt

Authentication (environment variables):
    AWS_BEARER_TOKEN_BEDROCK  -- Bearer token for Bedrock API
    AWS_REGION                -- AWS region (default: us-east-1)

Usage:
    # Fixed rubric on HelpSteer2
    python src/eval_pointwise.py --benchmark helpsteer2 --rubric-mode fixed

    # Dynamic rubrics on HealthBench
    python src/eval_pointwise.py --benchmark healthbench --rubric-mode dynamic

    # Instance-specific rubrics on ProfBench with one judge
    python src/eval_pointwise.py --benchmark profbench --rubric-mode fixed_instance \\
        --judges claude-sonnet-4

    # Quick test run
    python src/eval_pointwise.py --benchmark helpsteer2 --rubric-mode fixed \\
        --judges llama-3.1-8b --max-examples 5
"""

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

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
# Fixed rubrics (benchmark-specific, used in "fixed" mode)
# ---------------------------------------------------------------------------
HELPSTEER2_FIXED_RUBRIC = """\
Evaluate the response on these five dimensions:
1. Helpfulness: How well does the response address the user's request and \
provide useful information or assistance?
2. Correctness: Are all stated facts accurate and free of errors? For tasks \
with a definitive answer, is it correct?
3. Coherence: Is the response logically organized, consistent, and clearly \
expressed?
4. Complexity: Does the response demonstrate appropriate intellectual depth \
and sophistication for the task?
5. Verbosity: Is the level of detail appropriate -- neither too terse nor \
excessively verbose?"""

PROFBENCH_FIXED_RUBRIC = """\
Evaluate the response on these five dimensions for professional-grade quality:
1. Technical Accuracy: Are all technical claims, code, analysis, or \
domain-specific content correct and reliable?
2. Completeness: Does the response fully address all aspects of the task \
without significant omissions?
3. Professional Quality: Is the response at a level expected of a domain \
expert, with appropriate tone and standards?
4. Clarity and Organization: Is the response well-structured, easy to follow, \
and clearly communicated?
5. Practical Applicability: Can the response be directly used or applied in \
a real-world professional context?"""

HEALTHBENCH_FIXED_RUBRIC = """\
Evaluate the response on these five dimensions for health-related quality:
1. Medical Accuracy: Is the health/medical information factually correct and \
consistent with current clinical evidence?
2. Safety: Does the response avoid harmful, dangerous, or misleading health \
recommendations?
3. Completeness: Does the response cover all relevant health considerations, \
including important caveats and when to seek professional care?
4. Communication: Is the information communicated clearly and at an \
appropriate level for the intended audience?
5. Appropriate Scope: Does the response appropriately acknowledge limitations, \
avoid overstepping its role, and recommend professional consultation when \
warranted?"""

BENCHMARK_FIXED_RUBRICS = {
    "helpsteer2": HELPSTEER2_FIXED_RUBRIC,
    "profbench": PROFBENCH_FIXED_RUBRIC,
    "healthbench": HEALTHBENCH_FIXED_RUBRIC,
}

# ---------------------------------------------------------------------------
# Pointwise judge prompts
# ---------------------------------------------------------------------------
POINTWISE_JUDGE_SYSTEM_PROMPT = """\
You are a helpful assistant that evaluates the quality of AI responses. \
You will be given a prompt, an evaluation rubric, and an AI response. \
Use the rubric criteria to assess the response and assign a score from 1 to 5.

Scoring scale:
  1 = Poor -- The response fails to address the prompt or has critical issues
  2 = Below Average -- The response partially addresses the prompt but has \
significant shortcomings
  3 = Average -- The response adequately addresses the prompt with some room \
for improvement
  4 = Good -- The response effectively addresses the prompt with minor issues
  5 = Excellent -- The response thoroughly and skillfully addresses the prompt

You must output ONLY a single JSON object with two keys:
- "score": an integer from 1 to 5
- "reason": a brief one-sentence explanation referencing the rubric criteria

Example output:
{"score": 4, "reason": "The response is accurate and well-organized but \
could provide more depth."}"""

POINTWISE_JUDGE_USER_TEMPLATE = """\
[Prompt]
{prompt}

[Evaluation Rubric]
{rubric}

[Response]
{response}

Based on the rubric, score this response from 1 to 5. Output only the JSON \
object."""

# ---------------------------------------------------------------------------
# Rubric generation prompts -- generated_fixed mode
# ---------------------------------------------------------------------------
POINTWISE_GEN_FIXED_RUBRIC_SYSTEM_PROMPT = """\
You are an expert evaluator designing assessment criteria for scoring AI \
assistant responses. Your task is to create a general-purpose evaluation rubric \
that can be applied to ANY type of prompt-response pair, where each response \
is scored individually on a 1-5 scale.

The rubric should cover the key dimensions that distinguish a good AI response \
from a poor one, regardless of the specific task. Think about what universally \
matters when a human judges the quality of an AI response.

Output ONLY a numbered list of 3-5 criteria. Each criterion should be one \
clear sentence describing what to evaluate."""

POINTWISE_GEN_FIXED_RUBRIC_USER_PROMPT = """\
Create a general-purpose evaluation rubric for scoring individual AI assistant \
responses on a 1-5 scale. The rubric should be applicable to any type of task \
(factual Q&A, creative writing, coding, reasoning, summarization, advice, etc.) \
without any task-specific or instance-specific information.

Write the evaluation rubric."""

# ---------------------------------------------------------------------------
# Rubric generation prompts -- dynamic mode
# ---------------------------------------------------------------------------
POINTWISE_DYN_RUBRIC_SYSTEM_PROMPT = """\
You are an expert evaluator designing assessment criteria. Given a task \
prompt, create a concise evaluation rubric with 3-5 specific criteria \
for scoring the quality of responses to this prompt on a 1-5 scale.

Focus on what makes a response good or bad FOR THIS SPECIFIC TASK. Consider \
the task type (factual Q&A, creative writing, coding, reasoning, \
summarization, etc.) and what a high-quality answer requires.

Output ONLY a numbered list of criteria. Each criterion should be one clear \
sentence."""

POINTWISE_DYN_RUBRIC_USER_TEMPLATE = """\
[Prompt]
{prompt}

Write the evaluation rubric for scoring responses to the above prompt."""

# ---------------------------------------------------------------------------
# Criterion-level rubric & judge prompts (ProfBench / HealthBench dynamic)
# ---------------------------------------------------------------------------
CRITERION_DYN_RUBRIC_SYSTEM_PROMPT = """\
You are an expert evaluator designing assessment criteria for AI responses. \
Given a task prompt, generate specific evaluation criteria that can each be \
answered with a clear YES or NO when checking a response.

Guidelines:
- Each criterion must describe a concrete, observable requirement -- not an \
abstract quality dimension like "clarity" or "accuracy."
- Criteria should be specific to THIS prompt, capturing what a domain expert \
would check for in a high-quality answer.
- Include both content requirements (what must be covered) and safety/quality \
guards (what must be avoided).
- Assign each criterion a weight reflecting its importance:
  Critical -- essential; failure indicates a serious flaw
  Major    -- important for a good response but not essential
  Minor    -- desirable but lower impact

Output ONLY a JSON array of 5-10 objects with keys "criterion" and "weight":
[
  {"criterion": "Identifies the described symptoms as potential anaphylaxis", "weight": "Critical"},
  {"criterion": "Advises calling emergency services immediately", "weight": "Critical"},
  {"criterion": "Mentions that shellfish is a common allergen", "weight": "Major"},
  {"criterion": "Warns against self-medicating without professional guidance", "weight": "Major"},
  {"criterion": "Suggests monitoring for worsening symptoms", "weight": "Minor"}
]"""

CRITERION_DYN_RUBRIC_USER_TEMPLATE = """\
[Prompt]
{prompt}

Generate 5-10 specific, binary-testable evaluation criteria for responses to \
the above prompt. Output only the JSON array."""

CRITERION_JUDGE_SYSTEM_PROMPT = """\
You are a helpful assistant that evaluates AI responses against specific \
criteria. For each criterion provided, determine whether the response \
satisfies it.

You must output ONLY a JSON array. For each criterion, include:
- "criterion": the criterion text (copied exactly)
- "fulfilled": true if the response clearly satisfies this criterion, \
false otherwise

Do not add extra keys or commentary."""

CRITERION_JUDGE_USER_TEMPLATE = """\
[Prompt]
{prompt}

[Response]
{response}

[Evaluation Criteria]
{criteria_json}

For each criterion above, determine whether the response fulfills it. \
Output only the JSON array."""

CRITERION_WEIGHT_MAP = {"Critical": 4, "Major": 3, "Minor": 2}

# ---------------------------------------------------------------------------
# Bedrock client (bearer-token auth)
# ---------------------------------------------------------------------------

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

    # Discover parquet file URLs via HF Datasets Server API
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

    # Filter to the requested split
    split_files = [f for f in parquet_files if f["split"] == split]

    if not split_files:
        available = sorted(
            set(
                (f.get("config", "?"), f["split"])
                for f in parquet_files
            )
        )
        raise ValueError(
            f"Split '{split}' not found for {dataset_id}. "
            f"Available (config, split) pairs: {available}"
        )

    # Download and concatenate parquet files
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
# Data loaders
# ---------------------------------------------------------------------------

SEVERITY_WEIGHTS = {"Critical": 4, "Major": 3, "Minor": 2, "Additional": 1}


def load_helpsteer2(cache_dir: Path) -> list[dict]:
    """Load HelpSteer2 validation split.

    Returns list of dicts with keys:
        prompt, response, human_score, rubric_criteria (None)
    """
    bm_dir = cache_dir / "helpsteer2"
    bm_dir.mkdir(parents=True, exist_ok=True)
    df = _download_hf_dataset("nvidia/HelpSteer2", "validation", bm_dir)

    logger.info("HelpSteer2 columns: %s", list(df.columns))
    logger.info("HelpSteer2 shape: %s", df.shape)

    items = []
    for _, row in df.iterrows():
        avg_score = (float(row["helpfulness"]) + float(row["correctness"]) + float(row["coherence"])) / 3.0
        items.append(
            {
                "prompt": str(row["prompt"]),
                "response": str(row["response"]),
                "human_score": avg_score,
                "rubric_criteria": None,
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
            "criterion_weight", c.get("severity", c.get("weight", "Minor")),
        )
        if isinstance(severity, str):
            weight = SEVERITY_WEIGHTS.get(severity, 1)
        else:
            weight = float(severity)

        # Try several possible field names for the fulfilment flag
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


def _format_profbench_criteria(rubric_criteria: list[dict]) -> str:
    """Format ProfBench expert criteria as a numbered list (no labels)."""
    lines = []
    for i, c in enumerate(rubric_criteria, 1):
        severity = c.get(
            "criterion_weight", c.get("severity", c.get("weight", "")),
        )
        criterion = c.get(
            "criterion_description",
            c.get("criterion", c.get("description", c.get("criteria", ""))),
        )
        if severity:
            lines.append(f"{i}. [{severity}] {criterion}")
        else:
            lines.append(f"{i}. {criterion}")
    return "\n".join(lines)


def load_profbench(cache_dir: Path) -> list[dict]:
    """Load ProfBench dataset.

    Returns list of dicts with keys:
        prompt, response, human_score, rubric_criteria,
        task_id, domain, response_model
    """
    bm_dir = cache_dir / "profbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple possible split names
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
    logger.info("ProfBench shape: %s", df.shape)

    # Log sample row for debugging
    if len(df) > 0:
        sample = df.iloc[0]
        for col in df.columns:
            val = sample[col]
            val_str = repr(val)
            if len(val_str) > 300:
                val_str = val_str[:300] + "..."
            logger.info(
                "  ProfBench sample[%s] type=%s: %s",
                col, type(val).__name__, val_str,
            )

    # Detect response columns
    response_cols = [
        c for c in df.columns if c.endswith("_response")
    ]
    if not response_cols:
        response_cols = [
            c for c in df.columns
            if "response" in c.lower() and c.lower() != "response"
        ]
    if not response_cols and "response" in df.columns:
        response_cols = ["response"]
    logger.info("ProfBench response columns: %s", response_cols)

    # Detect rubric column
    rubric_col = None
    for candidate in ["rubric_criteria", "rubrics", "criteria", "rubric"]:
        if candidate in df.columns:
            rubric_col = candidate
            break
    logger.info("ProfBench rubric column: %s", rubric_col)

    items = []
    for _, row in df.iterrows():
        prompt = str(
            row.get("prompt", row.get("task", row.get("instruction", "")))
        )
        task_id = row.get("task_id", row.get("id", None))
        domain = row.get("domain", None)

        # Parse rubric criteria (may be numpy ndarray from Parquet)
        raw_rubric = row.get(rubric_col) if rubric_col else None
        if isinstance(raw_rubric, str):
            try:
                rubric_criteria = json.loads(raw_rubric)
            except json.JSONDecodeError:
                rubric_criteria = []
        elif isinstance(raw_rubric, (list, np.ndarray)):
            # Convert numpy types to native Python for JSON serialization
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

            model_key = (
                col.replace("_response", "").replace("_output", "")
            )
            human_score = _compute_profbench_score(rubric_criteria, model_key)

            items.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "human_score": human_score,
                    "rubric_criteria": rubric_criteria,
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
    else:
        logger.warning("ProfBench: 0 items loaded!")
    return items


def _format_healthbench_criteria(rubric_criteria: list[dict]) -> str:
    """Format HealthBench physician criteria as a numbered list (no labels)."""
    lines = []
    for i, c in enumerate(rubric_criteria, 1):
        criterion = c.get(
            "criterion", c.get("criteria", c.get("description", "")),
        )
        points = c.get("points", 0)
        tags = c.get("tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        sign = "+" if points > 0 else ""
        lines.append(f"{i}. ({sign}{points} pts{tag_str}) {criterion}")
    return "\n".join(lines)


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

    Uses two files from the HuggingFace repo:
    - oss_eval.jsonl:      prompts + structured rubric criteria (with points)
    - oss_meta_eval.jsonl:  completions + physician binary labels per rubric

    Returns list of dicts with keys:
        prompt, response, human_score, rubric_criteria
    """
    bm_dir = cache_dir / "healthbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load structured rubric criteria from oss_eval
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

    # 2. Load completions + physician labels from meta_eval
    meta_data = _download_healthbench_jsonl(
        "2025-05-07-06-14-12_oss_meta_eval.jsonl", bm_dir,
    )
    logger.info("HealthBench meta_eval: %d rows", len(meta_data))

    # 3. Group by (prompt_id, completion_id) and aggregate physician labels
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
        # Majority of physician annotations for this rubric criterion
        labels = row["binary_labels"]
        majority = sum(labels) > len(labels) / 2
        pair_data[key]["labels"].append(majority)

    logger.info(
        "HealthBench: %d unique (prompt, completion) pairs from %d prompts",
        len(pair_data),
        len(set(v["pid"] for v in pair_data.values())),
    )

    # 4. Build items
    items = []
    for (pid, cid), data in pair_data.items():
        prompt_text = _format_messages(data["prompt"])
        response = data["completion"]
        labels = data["labels"]

        # human_score = fraction of rubric criteria passed (by physician majority)
        human_score = sum(labels) / len(labels) if labels else 0.0

        # Get structured rubric criteria from oss_eval (for fixed_instance mode)
        rubric_criteria = prompt_rubrics.get(pid, [])

        items.append(
            {
                "prompt": prompt_text,
                "response": response,
                "human_score": human_score,
                "rubric_criteria": rubric_criteria,
            }
        )

    if items:
        logger.info(
            "HealthBench: %d items, score range [%.3f, %.3f]",
            len(items),
            min(it["human_score"] for it in items),
            max(it["human_score"] for it in items),
        )
    else:
        logger.warning("HealthBench: 0 items loaded!")
    return items


BENCHMARK_LOADERS = {
    "helpsteer2": load_helpsteer2,
    "profbench": load_profbench,
    "healthbench": load_healthbench,
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
# Score parsing
# ---------------------------------------------------------------------------

def parse_score(raw: str) -> int | None:
    """Extract integer score (1-5) from the judge's JSON output."""
    # Try JSON key first
    m = re.search(r'"score"\s*:\s*(\d+)', raw)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score

    # Fallback: standalone digit 1-5
    m = re.search(r'\b([1-5])\b', raw)
    if m:
        return int(m.group(1))

    return None


# ---------------------------------------------------------------------------
# Criterion-level parsing helpers
# ---------------------------------------------------------------------------

def parse_criteria_json(raw: str) -> list[dict]:
    """Parse a JSON array of criterion objects from LLM output."""
    # Try direct parse first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from surrounding text
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    return []


def compute_criterion_score(
    criteria: list[dict],
    evaluation: list[dict],
    benchmark: str,
) -> float | None:
    """Compute aggregated score from criterion-level pass/fail evaluation.

    ProfBench: weighted fraction * 100  (matches 0-100 human scale)
    HealthBench: unweighted fraction    (matches 0-1 human scale)
    """
    if not evaluation:
        return None

    fulfilled_flags = []
    weights = []
    for item in evaluation:
        f = item.get("fulfilled")
        if f is None:
            continue
        fulfilled_flags.append(bool(f))
        # Find the matching weight from the original criteria
        criterion_text = item.get("criterion", "")
        weight = "Major"  # default
        for c in criteria:
            if c.get("criterion", "") == criterion_text:
                weight = c.get("weight", "Major")
                break
        weights.append(CRITERION_WEIGHT_MAP.get(weight, 3))

    if not fulfilled_flags:
        return None

    if benchmark == "healthbench":
        # Unweighted fraction to match human annotation scheme
        return sum(fulfilled_flags) / len(fulfilled_flags)
    else:
        # Weighted fraction * 100 to match ProfBench 0-100 scale
        total = sum(weights)
        scored = sum(w for w, f in zip(weights, fulfilled_flags) if f)
        return scored / total * 100 if total > 0 else None


# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------

def get_rubric_for_item(
    benchmark: str,
    rubric_mode: str,
    item: dict,
    *,
    generated_rubric: str | None = None,
    dynamic_rubrics: dict[str, str] | None = None,
) -> str:
    """Return the rubric text for a specific item based on the rubric mode."""
    if rubric_mode == "fixed":
        return BENCHMARK_FIXED_RUBRICS[benchmark]

    elif rubric_mode == "fixed_instance":
        criteria = item.get("rubric_criteria")
        if benchmark == "helpsteer2" or not criteria:
            # HelpSteer2 has no per-instance rubrics; fall back to fixed
            return BENCHMARK_FIXED_RUBRICS[benchmark]
        elif benchmark == "profbench":
            return _format_profbench_criteria(criteria)
        elif benchmark == "healthbench":
            return _format_healthbench_criteria(criteria)
        return BENCHMARK_FIXED_RUBRICS[benchmark]

    elif rubric_mode == "generated_fixed":
        return generated_rubric

    elif rubric_mode == "dynamic":
        return dynamic_rubrics.get(
            item["prompt"],
            "Evaluate overall quality, accuracy, helpfulness, and clarity.",
        )

    return BENCHMARK_FIXED_RUBRICS.get(benchmark, "Evaluate overall quality.")


# ---------------------------------------------------------------------------
# Rubric generation
# ---------------------------------------------------------------------------

def generate_fixed_rubric(
    client: BedrockClient,
    judge_name: str,
    model_id: str,
    output_dir: Path,
    *,
    temperature: float = 0.0,
) -> str:
    """Generate a single general-purpose pointwise rubric, cached to disk."""
    rubric_file = output_dir / f"fixed_rubric_{judge_name}.txt"

    if rubric_file.exists():
        rubric = rubric_file.read_text().strip()
        logger.info(
            "[%s] Loaded cached generated fixed rubric from %s",
            judge_name, rubric_file,
        )
        return rubric

    logger.info(
        "[%s] Generating fixed rubric (model_id=%s)", judge_name, model_id,
    )
    rubric = call_judge(
        client,
        model_id,
        system=POINTWISE_GEN_FIXED_RUBRIC_SYSTEM_PROMPT,
        user_message=POINTWISE_GEN_FIXED_RUBRIC_USER_PROMPT,
        max_tokens=512,
        temperature=temperature,
    )

    rubric_file.write_text(rubric)
    logger.info(
        "[%s] Saved generated fixed rubric -> %s", judge_name, rubric_file,
    )
    return rubric


def generate_dynamic_rubrics(
    client: BedrockClient,
    judge_name: str,
    model_id: str,
    prompts: list[str],
    output_dir: Path,
    *,
    temperature: float = 0.0,
) -> dict[str, str]:
    """Generate per-prompt rubrics for dynamic mode, with checkpointing."""
    rubric_file = output_dir / f"rubrics_{judge_name}.json"

    rubrics: dict[str, str] = {}
    if rubric_file.exists():
        with open(rubric_file) as f:
            rubrics = json.load(f)
        logger.info(
            "[%s] Loaded %d cached rubrics from %s",
            judge_name, len(rubrics), rubric_file,
        )

    remaining = [p for p in prompts if p not in rubrics]
    if not remaining:
        logger.info(
            "[%s] All %d rubrics already generated",
            judge_name, len(prompts),
        )
        return rubrics

    logger.info(
        "[%s] Generating rubrics for %d prompts (model_id=%s)",
        judge_name, len(remaining), model_id,
    )

    for i, prompt in enumerate(
        tqdm(remaining, desc=f"{judge_name} rubrics"),
    ):
        user_msg = POINTWISE_DYN_RUBRIC_USER_TEMPLATE.format(prompt=prompt)
        rubric = call_judge(
            client,
            model_id,
            system=POINTWISE_DYN_RUBRIC_SYSTEM_PROMPT,
            user_message=user_msg,
            max_tokens=512,
            temperature=temperature,
        )
        rubrics[prompt] = rubric

        # Checkpoint every 50 rubrics
        if (i + 1) % 50 == 0:
            with open(rubric_file, "w") as f:
                json.dump(rubrics, f, indent=2, ensure_ascii=False)

    # Final save
    with open(rubric_file, "w") as f:
        json.dump(rubrics, f, indent=2, ensure_ascii=False)
    logger.info(
        "[%s] Saved %d rubrics -> %s", judge_name, len(rubrics), rubric_file,
    )
    return rubrics


def generate_dynamic_criteria(
    client: BedrockClient,
    judge_name: str,
    model_id: str,
    prompts: list[str],
    output_dir: Path,
    *,
    temperature: float = 0.0,
) -> dict[str, list[dict]]:
    """Generate per-prompt criterion-level rubrics (structured JSON), with checkpointing."""
    criteria_file = output_dir / f"criteria_{judge_name}.json"

    all_criteria: dict[str, list[dict]] = {}
    if criteria_file.exists():
        with open(criteria_file) as f:
            all_criteria = json.load(f)
        logger.info(
            "[%s] Loaded %d cached criteria sets from %s",
            judge_name, len(all_criteria), criteria_file,
        )

    remaining = [p for p in prompts if p not in all_criteria]
    if not remaining:
        logger.info(
            "[%s] All %d criteria sets already generated",
            judge_name, len(prompts),
        )
        return all_criteria

    logger.info(
        "[%s] Generating criterion-level rubrics for %d prompts (model_id=%s)",
        judge_name, len(remaining), model_id,
    )

    for i, prompt in enumerate(
        tqdm(remaining, desc=f"{judge_name} criteria"),
    ):
        user_msg = CRITERION_DYN_RUBRIC_USER_TEMPLATE.format(prompt=prompt)
        raw = call_judge(
            client,
            model_id,
            system=CRITERION_DYN_RUBRIC_SYSTEM_PROMPT,
            user_message=user_msg,
            max_tokens=1024,
            temperature=temperature,
        )
        criteria = parse_criteria_json(raw)
        if not criteria:
            logger.warning(
                "[%s] Failed to parse criteria for prompt (first 80 chars): %s",
                judge_name, prompt[:80],
            )
        all_criteria[prompt] = criteria

        if (i + 1) % 50 == 0:
            with open(criteria_file, "w") as f:
                json.dump(all_criteria, f, indent=2, ensure_ascii=False)

    with open(criteria_file, "w") as f:
        json.dump(all_criteria, f, indent=2, ensure_ascii=False)
    logger.info(
        "[%s] Saved %d criteria sets -> %s",
        judge_name, len(all_criteria), criteria_file,
    )
    return all_criteria


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_judge(
    client: BedrockClient,
    judge_name: str,
    judge_config: dict,
    items: list[dict],
    benchmark: str,
    output_dir: Path,
    *,
    rubric_mode: str = "fixed",
    max_examples: int | None = None,
    checkpoint_every: int = 20,
) -> tuple[Path, dict]:
    """Run a single judge over all items; return (output_path, metrics)."""
    if max_examples is not None:
        items = items[:max_examples]

    # Criterion-level evaluation for ProfBench / HealthBench in dynamic mode
    use_criteria_eval = (
        rubric_mode == "dynamic"
        and benchmark in ("profbench", "healthbench")
    )

    # --- Step 1: Generate rubrics if needed ---
    generated_rubric: str | None = None
    dynamic_rubrics: dict[str, str] | None = None
    dynamic_criteria: dict[str, list[dict]] | None = None

    if rubric_mode == "generated_fixed":
        generated_rubric = generate_fixed_rubric(
            client,
            judge_name,
            judge_config["model_id"],
            output_dir,
            temperature=judge_config["temperature"],
        )
    elif rubric_mode == "dynamic":
        unique_prompts = list(dict.fromkeys(it["prompt"] for it in items))
        if use_criteria_eval:
            dynamic_criteria = generate_dynamic_criteria(
                client,
                judge_name,
                judge_config["model_id"],
                unique_prompts,
                output_dir,
                temperature=judge_config["temperature"],
            )
        else:
            dynamic_rubrics = generate_dynamic_rubrics(
                client,
                judge_name,
                judge_config["model_id"],
                unique_prompts,
                output_dir,
                temperature=judge_config["temperature"],
            )

    # --- Step 2: Score each item ---
    output_file = output_dir / f"judge_{judge_name}.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(items):
        logger.info(
            "[%s] Already complete (%d results)", judge_name, start_idx,
        )
        return output_file, compute_metrics(results)

    logger.info(
        "[%s] Scoring items %d -> %d  (mode=%s, model=%s, criteria_eval=%s)",
        judge_name, start_idx, len(items),
        rubric_mode, judge_config["model_id"], use_criteria_eval,
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
        if use_criteria_eval:
            # --- Criterion-level pass/fail evaluation ---
            criteria = dynamic_criteria.get(item["prompt"], [])
            if not criteria:
                # Fallback: no criteria generated, score is None
                raw = ""
                score = None
            else:
                criteria_json = json.dumps(criteria, ensure_ascii=False)
                user_msg = CRITERION_JUDGE_USER_TEMPLATE.format(
                    prompt=item["prompt"],
                    response=item["response"],
                    criteria_json=criteria_json,
                )
                raw = call_judge(
                    client,
                    judge_config["model_id"],
                    system=CRITERION_JUDGE_SYSTEM_PROMPT,
                    user_message=user_msg,
                    max_tokens=1024,
                    temperature=judge_config["temperature"],
                )
                evaluation = parse_criteria_json(raw)
                score = compute_criterion_score(
                    criteria, evaluation, benchmark,
                )

            result_entry = {
                "prompt": item["prompt"],
                "response": item["response"],
                "raw_judge_output": raw,
                "judge_score": score,
                "human_score": item["human_score"],
                "rubric": criteria,
            }
        else:
            # --- Holistic 1-5 evaluation ---
            rubric = get_rubric_for_item(
                benchmark,
                rubric_mode,
                item,
                generated_rubric=generated_rubric,
                dynamic_rubrics=dynamic_rubrics,
            )

            user_msg = POINTWISE_JUDGE_USER_TEMPLATE.format(
                prompt=item["prompt"],
                rubric=rubric,
                response=item["response"],
            )

            raw = call_judge(
                client,
                judge_config["model_id"],
                system=POINTWISE_JUDGE_SYSTEM_PROMPT,
                user_message=user_msg,
                max_tokens=judge_config["max_tokens"],
                temperature=judge_config["temperature"],
            )

            score = parse_score(raw)

            result_entry = {
                "prompt": item["prompt"],
                "response": item["response"],
                "raw_judge_output": raw,
                "judge_score": score,
                "human_score": item["human_score"],
                "rubric": rubric,
            }

        # Include optional metadata
        for key in ["task_id", "domain", "response_model"]:
            if key in item:
                result_entry[key] = item[key]

        results.append(result_entry)

        if (i + 1) % checkpoint_every == 0:
            save_checkpoint(output_file, results)
            interim = compute_metrics(results)
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
    metrics = compute_metrics(results)
    logger.info(
        "[%s] Done -- %d results -> %s", judge_name, len(results), output_file,
    )
    return output_file, metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """Compute Spearman and Pearson correlations between judge and human scores."""
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
# vLLM-based judge loop
# ---------------------------------------------------------------------------

def run_judge_vllm(
    vllm_judge,
    items: list[dict],
    benchmark: str,
    output_dir: Path,
    *,
    rubric_mode: str = "fixed",
    dynamic_rubrics: dict[str, str] | None = None,
    generated_rubric: str | None = None,
    max_examples: int | None = None,
    batch_size: int = 32,
    checkpoint_every: int = 100,
    judge_name: str = "vllm",
) -> tuple[Path, dict]:
    """Run vLLM judge over all items in batches (holistic 1-5 scoring only)."""
    if max_examples is not None:
        items = items[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(items):
        logger.info("[%s] Already complete (%d results)", judge_name, start_idx)
        return output_file, compute_metrics(results)

    logger.info(
        "[%s] Scoring items %d -> %d (mode=%s, batch_size=%d)",
        judge_name, start_idx, len(items), rubric_mode, batch_size,
    )

    remaining = items[start_idx:]
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        prompts_for_batch = []

        for item in batch:
            rubric = get_rubric_for_item(
                benchmark,
                rubric_mode,
                item,
                generated_rubric=generated_rubric,
                dynamic_rubrics=dynamic_rubrics,
            )
            user_msg = POINTWISE_JUDGE_USER_TEMPLATE.format(
                prompt=item["prompt"],
                rubric=rubric,
                response=item["response"],
            )
            prompts_for_batch.append((POINTWISE_JUDGE_SYSTEM_PROMPT, user_msg))

        raw_outputs = vllm_judge.generate_batch(prompts_for_batch, batch_size=batch_size)

        for item, raw in zip(batch, raw_outputs):
            if raw is None:
                raw = ""
            score = parse_score(raw)

            result_entry = {
                "prompt": item["prompt"],
                "response": item["response"],
                "raw_judge_output": raw,
                "judge_score": score,
                "human_score": item["human_score"],
            }
            for key in ["task_id", "domain", "response_model"]:
                if key in item:
                    result_entry[key] = item[key]
            results.append(result_entry)

        global_idx_end = start_idx + batch_start + len(batch)
        if global_idx_end % checkpoint_every < batch_size or batch_start + batch_size >= len(remaining):
            save_checkpoint(output_file, results)
            interim = compute_metrics(results)
            logger.info(
                "[%s] Checkpoint %d -- spearman=%s  pearson=%s  errors=%d",
                judge_name, global_idx_end,
                f'{interim["spearman_corr"]:.4f}' if interim["spearman_corr"] is not None else "N/A",
                f'{interim["pearson_corr"]:.4f}' if interim["pearson_corr"] is not None else "N/A",
                interim["n_parse_errors"],
            )

    save_checkpoint(output_file, results)
    metrics = compute_metrics(results)
    logger.info("[%s] Done -- %d results -> %s", judge_name, len(results), output_file)
    return output_file, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judges on pointwise benchmarks",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["helpsteer2", "profbench", "healthbench"],
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--rubric-mode",
        type=str,
        default="fixed",
        choices=["fixed", "fixed_instance", "generated_fixed", "dynamic"],
        help=(
            "'fixed' uses a hardcoded benchmark rubric; "
            "'fixed_instance' uses per-instance human-authored rubrics; "
            "'generated_fixed' uses an LLM-generated general rubric; "
            "'dynamic' generates per-instance rubrics (default: fixed)"
        ),
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        default=list(JUDGES.keys()),
        help="Which judge models to use (default: all API judges). "
             "For vLLM backend, use a custom name (e.g. 'qwen3-14b').",
    )
    parser.add_argument(
        "--rubric-file",
        type=str,
        default=None,
        help="Path to pre-generated rubrics JSON (for dynamic mode). "
             "Skips rubric generation and loads from this file instead.",
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
    parser.add_argument(
        "--backend", type=str, default="bedrock",
        choices=["bedrock", "vllm"],
        help="Judge backend: 'bedrock' (API) or 'vllm' (local) (default: bedrock)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name/path for vLLM backend (e.g. Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Tensor parallelism for vLLM (default: 1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for vLLM inference (default: 32)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=8192,
        help="Max model sequence length for vLLM (default: 8192)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="GPU memory utilization for vLLM (default: 0.90)",
    )

    args = parser.parse_args()

    # Warn about fixed_instance on HelpSteer2
    if args.rubric_mode == "fixed_instance" and args.benchmark == "helpsteer2":
        logger.warning(
            "HelpSteer2 has no per-instance rubrics; "
            "fixed_instance will fall back to fixed mode."
        )

    if args.backend == "vllm" and not args.model:
        parser.error("--model is required when using --backend vllm")

    # Output: outputs/pointwise/{benchmark}/{rubric_mode}/
    output_dir = (
        Path(args.output_dir) / "pointwise" / args.benchmark / args.rubric_mode
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Benchmark: %s | Rubric mode: %s | Backend: %s | Output: %s",
        args.benchmark, args.rubric_mode, args.backend, output_dir,
    )

    # 1. Load data
    items = BENCHMARK_LOADERS[args.benchmark](cache_dir)

    # 2. Load pre-generated rubrics if provided
    dynamic_rubrics: dict[str, str] | None = None
    generated_rubric: str | None = None
    if args.rubric_file:
        with open(args.rubric_file) as f:
            dynamic_rubrics = json.load(f)
        logger.info("Loaded %d rubrics from %s", len(dynamic_rubrics), args.rubric_file)

    # 3. Run judges
    all_metrics = {}

    if args.backend == "vllm":
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from vllm_judge import VLLMJudge

        vllm_judge = VLLMJudge(
            model_path=args.model,
            max_new_tokens=512,
            temperature=0.0,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

        # Generate rubrics via vLLM if dynamic mode and no rubric file
        if args.rubric_mode == "dynamic" and not dynamic_rubrics:
            unique_prompts = list(dict.fromkeys(it["prompt"] for it in items))
            dynamic_rubrics = vllm_judge.generate_rubrics(
                system_prompt=POINTWISE_DYN_RUBRIC_SYSTEM_PROMPT,
                user_template=POINTWISE_DYN_RUBRIC_USER_TEMPLATE,
                keys=unique_prompts,
                output_dir=output_dir,
                judge_name=args.judges[0],
                batch_size=args.batch_size,
                template_key="prompt",
            )
        elif args.rubric_mode == "generated_fixed" and not generated_rubric:
            generated_rubric = vllm_judge.generate(
                POINTWISE_GEN_FIXED_RUBRIC_SYSTEM_PROMPT,
                POINTWISE_GEN_FIXED_RUBRIC_USER_PROMPT,
            )
            logger.info("Generated fixed rubric via vLLM")

        for judge_name in args.judges:
            _, metrics = run_judge_vllm(
                vllm_judge,
                items,
                args.benchmark,
                output_dir,
                rubric_mode=args.rubric_mode,
                dynamic_rubrics=dynamic_rubrics,
                generated_rubric=generated_rubric,
                max_examples=args.max_examples,
                batch_size=args.batch_size,
                checkpoint_every=args.checkpoint_every,
                judge_name=judge_name,
            )
            all_metrics[judge_name] = metrics
    else:
        client = BedrockClient(region=args.region)

        for judge_name in args.judges:
            _, metrics = run_judge(
                client,
                judge_name,
                JUDGES[judge_name],
                items,
                args.benchmark,
                output_dir,
                rubric_mode=args.rubric_mode,
                max_examples=args.max_examples,
                checkpoint_every=args.checkpoint_every,
            )
            all_metrics[judge_name] = metrics

    # 4. Save and display results
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("=" * 70)
    logger.info(
        "Results -- %s / %s", args.benchmark, args.rubric_mode,
    )
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
            if m["spearman_corr"] is not None
            else "N/A",
            f'{m["pearson_corr"]:.4f}'
            if m["pearson_corr"] is not None
            else "N/A",
            m["n_valid"],
            m["n_parse_errors"],
        )
    logger.info("=" * 70)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
