#!/usr/bin/env python3
"""
Evaluate Prometheus 2 as an LLM judge on five benchmarks.

Prometheus 2 (prometheus-eval/prometheus-7b-v2.0 or prometheus-8x7b-v2.0)
is hosted locally via transformers and used as the judge model directly.

Supports:
    Pairwise benchmarks: alpacaeval, mtbench, biggen
    Pointwise benchmarks: helpsteer2, healthbench

Usage:
    # Pairwise on AlpacaEval with 7B model
    python src/eval_prometheus2.py \
        --benchmark alpacaeval \
        --model prometheus-eval/prometheus-7b-v2.0

    # Pointwise on HelpSteer2 with 8x7B model
    python src/eval_prometheus2.py \
        --benchmark helpsteer2 \
        --model prometheus-eval/prometheus-8x7b-v2.0

    # BiGGen with dataset rubrics
    python src/eval_prometheus2.py \
        --benchmark biggen \
        --model prometheus-eval/prometheus-7b-v2.0 \
        --use-dataset-rubric

    # Quick test
    python src/eval_prometheus2.py \
        --benchmark alpacaeval \
        --model prometheus-eval/prometheus-7b-v2.0 \
        --max-examples 10
"""

import argparse
import io
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus 2 prompt templates
# ---------------------------------------------------------------------------

ABS_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective "
    "feedback based on specific criteria, ensuring each assessment reflects "
    "the absolute standards set for performance."
)

REL_SYSTEM_PROMPT = (
    "You are a fair judge assistant assigned to deliver insightful feedback "
    "that compares individual performances, highlighting how each stands "
    "relative to others within the same cohort."
)

ABSOLUTE_PROMPT_WO_REF = """\
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Score Rubrics:
{rubric}

###Feedback: """

RELATIVE_PROMPT_WO_REF = """\
###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Response A and Response B), and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
2. Make comparisons between Response A, Response B. Instead of examining Response A and Response B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
{instruction}

###Response A:
{response_A}

###Response B:
{response_B}

###Score Rubric:
{rubric}

###Feedback: """

# Default rubrics when no dataset-specific rubric is available
DEFAULT_PAIRWISE_RUBRIC = (
    "Which response better follows the instruction and is more helpful, "
    "accurate, relevant, and well-written? Consider correctness, completeness, "
    "clarity, and overall quality."
)

DEFAULT_POINTWISE_RUBRIC = """\
[Helpfulness and Quality]
Score 1: The response completely fails to address the instruction, is largely irrelevant, or contains critical errors that make it harmful or misleading.
Score 2: The response partially addresses the instruction but has significant shortcomings in accuracy, completeness, or clarity.
Score 3: The response adequately addresses the instruction with acceptable accuracy and clarity, but has notable room for improvement.
Score 4: The response effectively addresses the instruction with good accuracy, completeness, and clarity, with only minor issues.
Score 5: The response thoroughly and skillfully addresses the instruction with excellent accuracy, completeness, clarity, and helpfulness."""

# HelpSteer2 specific rubric in Prometheus format
HELPSTEER2_RUBRIC = """\
[Helpfulness]
Score 1: The response is completely unhelpful, irrelevant, or harmful. It fails to address the user's request and provides no useful information.
Score 2: The response is minimally helpful. It partially addresses the request but has major issues with accuracy, completeness, or relevance.
Score 3: The response is moderately helpful. It addresses the main request but may lack detail, have minor inaccuracies, or miss important aspects.
Score 4: The response is very helpful. It thoroughly addresses the request with good accuracy, appropriate detail, and clear organization.
Score 5: The response is exceptionally helpful. It provides comprehensive, accurate, well-organized information that fully satisfies the user's needs."""

HEALTHBENCH_RUBRIC = """\
[Medical Response Quality]
Score 1: The response contains dangerous medical misinformation, fails to recommend professional care when needed, or could lead to serious harm.
Score 2: The response has significant medical inaccuracies or omissions that could mislead the user about their health situation.
Score 3: The response provides generally acceptable health information but lacks important nuances, caveats, or recommendations to seek professional care.
Score 4: The response provides accurate, helpful health information with appropriate caveats and recommendations, with only minor omissions.
Score 5: The response provides comprehensive, medically accurate information with clear communication, appropriate scope, and proper recommendations for professional consultation."""


# ---------------------------------------------------------------------------
# Model hosting
# ---------------------------------------------------------------------------

class Prometheus2Judge:
    """Host Prometheus 2 locally for inference (transformers backend)."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        device_map: str | dict = "auto",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info("Loading Prometheus 2 model: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info("Model loaded successfully on device: %s", self.model.device)

    def _build_prompt(self, system: str, user_message: str) -> str:
        combined_user = f"{system}\n\n{user_message}" if system else user_message
        messages = [
            {"role": "user", "content": combined_user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def generate(self, system: str, user_message: str) -> str:
        import torch

        input_text = self._build_prompt(system, user_message)
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=False,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_batch(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Generate responses for a batch of (system, user_message) pairs."""
        import torch

        if len(prompts) == 1:
            return [self.generate(prompts[0][0], prompts[0][1])]

        input_texts = [self._build_prompt(sys, usr) for sys, usr in prompts]

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=False,
        ).to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        results = []
        for idx in range(len(prompts)):
            input_len = inputs["attention_mask"][idx].sum().item()
            new_tokens = output_ids[idx][input_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(text)
        return results


class Prometheus2JudgeVLLM:
    """Host Prometheus 2 locally via vLLM for fast inference."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        logger.info(
            "Loading Prometheus 2 via vLLM: %s (tp=%d, max_model_len=%d)",
            model_path, tensor_parallel_size, max_model_len,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0,
        )
        logger.info("vLLM model loaded successfully (tp=%d)", tensor_parallel_size)

    def _build_prompt(self, system: str, user_message: str) -> str:
        combined_user = f"{system}\n\n{user_message}" if system else user_message
        messages = [
            {"role": "user", "content": combined_user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def generate(self, system: str, user_message: str) -> str:
        input_text = self._build_prompt(system, user_message)
        outputs = self.llm.generate([input_text], self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(self, prompts: list[tuple[str, str]]) -> list[str]:
        """Generate responses for a batch of (system, user_message) pairs.

        vLLM handles batching internally with continuous batching and paged
        attention, so we pass all prompts at once for maximum throughput.
        Prompts exceeding max_model_len are skipped (return None).
        """
        input_texts = [self._build_prompt(sys, usr) for sys, usr in prompts]
        max_input_len = self.llm.llm_engine.model_config.max_model_len - self.max_new_tokens

        valid_indices = []
        valid_texts = []
        for i, text in enumerate(input_texts):
            token_len = len(self.tokenizer.encode(text))
            if token_len <= max_input_len:
                valid_indices.append(i)
                valid_texts.append(text)
            else:
                logger.warning(
                    "Skipping prompt %d: %d tokens exceeds max_model_len (%d)",
                    i, token_len, self.llm.llm_engine.model_config.max_model_len,
                )

        results = [None] * len(prompts)
        if valid_texts:
            outputs = self.llm.generate(valid_texts, self.sampling_params)
            for idx, out in zip(valid_indices, outputs):
                results[idx] = out.outputs[0].text
        return results


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_pairwise_result(raw: str) -> str | None:
    """Extract 'A' or 'B' from Prometheus 2 pairwise output."""
    m = re.search(r'\[RESULT\]\s*([AB])', raw)
    if m:
        return m.group(1)
    m = re.search(r'\[RESULT\]\s*"?([AB])"?', raw)
    if m:
        return m.group(1)
    m = re.search(r'(?:Response\s+)?([AB])\s*(?:is better|wins)', raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def parse_pointwise_result(raw: str) -> int | None:
    """Extract score 1-5 from Prometheus 2 pointwise output."""
    m = re.search(r'\[RESULT\]\s*(\d)', raw)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score
    m = re.search(r'\[SCORE\]\s*(\d)', raw)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score
    m = re.search(r'Score:\s*(\d)', raw)
    if m:
        score = int(m.group(1))
        if 1 <= score <= 5:
            return score
    return None


def winner_to_preference(winner: str | None, output1_is_a: bool) -> int | None:
    if winner is None:
        return None
    if output1_is_a:
        return 1 if winner == "A" else 2
    else:
        return 2 if winner == "A" else 1


# ---------------------------------------------------------------------------
# Dataset loading -- AlpacaEval
# ---------------------------------------------------------------------------

HF_BASE = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main"


def _download_json(filename: str, cache_dir: Path) -> list[dict]:
    cache_path = cache_dir / filename
    if cache_path.exists():
        logger.info("Loading cached %s", cache_path)
        with open(cache_path) as f:
            return json.load(f)
    url = f"{HF_BASE}/{filename}"
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
    ae_dir = cache_dir / "alpacaeval"
    ae_dir.mkdir(parents=True, exist_ok=True)

    cross_raw = _download_json("alpaca_farm_human_crossannotations.json", ae_dir)
    human_ann_raw = _download_json("alpaca_farm_human_annotations.json", ae_dir)

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

    logger.info("AlpacaEval: %d pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Dataset loading -- MTBench
# ---------------------------------------------------------------------------

def _download_hf_dataset(
    dataset_id: str, split: str, cache_dir: Path,
) -> pd.DataFrame:
    safe_name = dataset_id.replace("/", "_")
    cache_path = cache_dir / f"{safe_name}_{split}.parquet"

    if cache_path.exists():
        logger.info("Loading cached %s", cache_path)
        return pd.read_parquet(cache_path)

    api_url = (
        f"https://datasets-server.huggingface.co/parquet?dataset={dataset_id}"
    )
    logger.info("Fetching parquet info for %s ...", dataset_id)
    resp = requests.get(api_url, timeout=60)
    resp.raise_for_status()
    info = resp.json()

    if "error" in info:
        raise ValueError(f"HF API error for {dataset_id}: {info['error']}")

    parquet_files = info.get("parquet_files", [])
    split_files = [f for f in parquet_files if f["split"] == split]

    if not split_files:
        available = sorted(
            set((f.get("config", "?"), f["split"]) for f in parquet_files)
        )
        raise ValueError(
            f"Split '{split}' not found for {dataset_id}. "
            f"Available: {available}"
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


def load_mtbench(cache_dir: Path) -> list[dict]:
    bm_dir = cache_dir / "mtbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset("lmsys/mt_bench_human_judgments", "human", bm_dir)

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
                "model_a": model_a,
                "model_b": model_b,
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
        entry["human_preferences"] = winners
        entry["generator"] = entry["model_b"]
        pairs.append(entry)

    logger.info("MTBench: %d pairs (%d ties excluded)", len(pairs), n_ties)
    return pairs


# ---------------------------------------------------------------------------
# Dataset loading -- BiGGen Bench
# ---------------------------------------------------------------------------

def load_biggen(cache_dir: Path) -> list[dict]:
    bm_dir = cache_dir / "biggen"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "prometheus-eval/BiGGen-Bench-Results", "human_eval", bm_dir,
    )

    prompt_data: dict[str, list[dict]] = defaultdict(list)
    prompt_rubrics: dict[str, dict] = {}

    for _, row in df.iterrows():
        prompt = str(row.get("input", row.get("prompt", "")))
        response = str(row.get("response", row.get("output", "")))
        human_score = float(
            row.get("human_score", row.get("score", row.get("human_eval", 0)))
        )
        model = str(
            row.get("model_name", row.get("model", row.get("generator", "unknown")))
        )

        raw_rubric = row.get("score_rubric")
        if raw_rubric is not None and prompt not in prompt_rubrics:
            if isinstance(raw_rubric, str):
                try:
                    prompt_rubrics[prompt] = json.loads(raw_rubric)
                except json.JSONDecodeError:
                    pass
            elif isinstance(raw_rubric, dict):
                prompt_rubrics[prompt] = dict(raw_rubric)

        prompt_data[prompt].append({
            "model": model,
            "response": response,
            "human_score": human_score,
        })

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
            entry = {
                "instruction": prompt,
                "output_1": models[m1]["response"],
                "output_2": models[m2]["response"],
                "model_1": models[m1]["model"],
                "model_2": models[m2]["model"],
                "human_majority": human_majority,
                "generator": models[m2]["model"],
            }
            if prompt in prompt_rubrics:
                entry["score_rubric"] = prompt_rubrics[prompt]
            pairs.append(entry)

    logger.info("BiGGen Bench: %d pairs (%d ties excluded)", len(pairs), n_ties)
    return pairs


def format_biggen_rubric(score_rubric: dict) -> str:
    parts = []
    criteria = score_rubric.get("criteria", "")
    if criteria:
        parts.append(f"[{criteria}]")
    for i in range(1, 6):
        desc = score_rubric.get(f"score{i}_description", "")
        if desc:
            parts.append(f"Score {i}: {desc}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dataset loading -- HelpSteer2
# ---------------------------------------------------------------------------

def load_helpsteer2(cache_dir: Path) -> list[dict]:
    bm_dir = cache_dir / "helpsteer2"
    bm_dir.mkdir(parents=True, exist_ok=True)
    df = _download_hf_dataset("nvidia/HelpSteer2", "validation", bm_dir)

    items = []
    for _, row in df.iterrows():
        avg_score = (float(row["helpfulness"]) + float(row["correctness"]) + float(row["coherence"])) / 3.0
        items.append({
            "prompt": str(row["prompt"]),
            "response": str(row["response"]),
            "human_score": avg_score,
            "rubric_criteria": None,
        })

    logger.info(
        "HelpSteer2: %d items, score range [%.1f, %.1f]",
        len(items),
        min(it["human_score"] for it in items),
        max(it["human_score"] for it in items),
    )
    return items


# ---------------------------------------------------------------------------
# Dataset loading -- HealthBench
# ---------------------------------------------------------------------------

HF_HEALTHBENCH_BASE = (
    "https://huggingface.co/datasets/openai/healthbench/resolve/main"
)


def _download_healthbench_jsonl(filename: str, cache_dir: Path) -> list[dict]:
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


def _format_messages(messages) -> str:
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


def _format_healthbench_criteria(rubric_criteria: list[dict]) -> str:
    lines = []
    for i, c in enumerate(rubric_criteria, 1):
        criterion = c.get("criterion", c.get("criteria", c.get("description", "")))
        points = c.get("points", 0)
        tags = c.get("tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        sign = "+" if points > 0 else ""
        lines.append(f"{i}. ({sign}{points} pts{tag_str}) {criterion}")
    return "\n".join(lines)


def load_healthbench(cache_dir: Path) -> list[dict]:
    bm_dir = cache_dir / "healthbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    eval_data = _download_healthbench_jsonl(
        "2025-05-07-06-14-12_oss_eval.jsonl", bm_dir,
    )
    prompt_rubrics: dict[str, list[dict]] = {}
    for item in eval_data:
        prompt_rubrics[item["prompt_id"]] = item.get("rubrics", [])
    logger.info("HealthBench oss_eval: %d prompts", len(prompt_rubrics))

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
        rubric_criteria = prompt_rubrics.get(pid, [])

        items.append({
            "prompt": prompt_text,
            "response": response,
            "human_score": human_score,
            "rubric_criteria": rubric_criteria,
        })

    if items:
        logger.info(
            "HealthBench: %d items, score range [%.3f, %.3f]",
            len(items),
            min(it["human_score"] for it in items),
            max(it["human_score"] for it in items),
        )
    return items


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

PAIRWISE_BENCHMARKS = {"alpacaeval", "mtbench", "biggen"}
POINTWISE_BENCHMARKS = {"helpsteer2", "healthbench", "biggen_pointwise"}


def load_biggen_pointwise(cache_dir: Path) -> list[dict]:
    """Load BiGGen Bench as pointwise items with human 1-5 scores."""
    bm_dir = cache_dir / "biggen"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "prometheus-eval/BiGGen-Bench-Results", "human_eval", bm_dir,
    )

    df = df[df["human_score"] > 0].copy()

    items = []
    for _, row in df.iterrows():
        prompt = str(row["input"])
        response = str(row["response"])
        human_score = float(row["human_score"])

        score_rubric = row.get("score_rubric")
        if isinstance(score_rubric, str):
            try:
                score_rubric = json.loads(score_rubric)
            except json.JSONDecodeError:
                score_rubric = None

        items.append({
            "prompt": prompt,
            "response": response,
            "human_score": human_score,
            "rubric_criteria": score_rubric,
        })

    logger.info(
        "BiGGen Pointwise: %d items, score range [%.1f, %.1f]",
        len(items),
        min(it["human_score"] for it in items),
        max(it["human_score"] for it in items),
    )
    return items


BENCHMARK_LOADERS = {
    "alpacaeval": load_alpacaeval,
    "mtbench": load_mtbench,
    "biggen": load_biggen,
    "helpsteer2": load_helpsteer2,
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
# Pairwise evaluation
# ---------------------------------------------------------------------------

def compute_pairwise_metrics(results: list[dict]) -> dict:
    valid = [r for r in results if r["preference"] is not None]
    n_valid = len(valid)
    n_total = len(results)

    if n_valid == 0:
        return {
            "human_agreement_pct": 0.0,
            "spearman_corr": None,
            "pearson_corr": None,
            "n_valid": 0,
            "n_total": n_total,
            "n_parse_errors": n_total,
        }

    n_agree = sum(1 for r in valid if r["preference"] == r["human_majority"])
    human_agreement = n_agree / n_valid * 100

    judge_by_gen: dict[str, list[float]] = {}
    gold_by_gen: dict[str, list[float]] = {}

    for r in valid:
        gen = r.get("generator")
        if gen is None:
            continue
        judge_by_gen.setdefault(gen, []).append(r["preference"] - 1)
        gold_by_gen.setdefault(gen, []).append(r["human_majority"] - 1)

    min_annotations = 5
    judge_wr = {
        g: sum(v) / len(v) * 100
        for g, v in judge_by_gen.items()
        if len(v) >= min_annotations
    }
    gold_wr = {
        g: sum(v) / len(v) * 100
        for g, v in gold_by_gen.items()
        if len(v) >= min_annotations
    }
    common_gens = sorted(set(judge_wr) & set(gold_wr))

    spearman_val = None
    pearson_val = None
    if len(common_gens) >= 3:
        j = [judge_wr[g] for g in common_gens]
        g = [gold_wr[g] for g in common_gens]
        spearman_val = round(spearmanr(j, g).statistic, 4)
        pearson_val = round(pearsonr(j, g).statistic, 4)

    return {
        "human_agreement_pct": round(human_agreement, 2),
        "spearman_corr": spearman_val,
        "pearson_corr": pearson_val,
        "n_generators": len(common_gens),
        "n_valid": n_valid,
        "n_total": n_total,
        "n_parse_errors": n_total - n_valid,
    }


def run_pairwise_eval(
    judge: Prometheus2Judge,
    pairs: list[dict],
    output_dir: Path,
    *,
    use_dataset_rubric: bool = False,
    max_examples: int | None = None,
    checkpoint_every: int = 20,
    seed: int = 42,
    batch_size: int = 1,
) -> tuple[Path, dict]:
    if max_examples is not None:
        pairs = pairs[:max_examples]

    output_file = output_dir / "judge_results.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(pairs):
        logger.info("Already complete (%d judgments)", start_idx)
        return output_file, compute_pairwise_metrics(results)

    logger.info("Judging pairwise examples %d -> %d (batch_size=%d)", start_idx, len(pairs), batch_size)

    rng = random.Random(seed)
    position_assignments = [rng.choice([True, False]) for _ in range(len(pairs))]

    remaining = pairs[start_idx:]
    pbar = tqdm(total=len(pairs), desc="pairwise", initial=start_idx)

    for batch_start in range(0, len(remaining), batch_size):
        batch_pairs = remaining[batch_start:batch_start + batch_size]
        batch_global_start = start_idx + batch_start

        batch_prompts = []
        batch_meta = []
        for j, pair in enumerate(batch_pairs):
            idx = batch_global_start + j
            output1_is_a = position_assignments[idx]
            if output1_is_a:
                resp_a, resp_b = pair["output_1"], pair["output_2"]
            else:
                resp_a, resp_b = pair["output_2"], pair["output_1"]

            if use_dataset_rubric and "score_rubric" in pair:
                rubric = format_biggen_rubric(pair["score_rubric"])
            else:
                rubric = DEFAULT_PAIRWISE_RUBRIC

            user_msg = RELATIVE_PROMPT_WO_REF.format(
                instruction=pair["instruction"],
                response_A=resp_a,
                response_B=resp_b,
                rubric=rubric,
            )
            batch_prompts.append((REL_SYSTEM_PROMPT, user_msg))
            batch_meta.append((pair, output1_is_a))

        raw_outputs = judge.generate_batch(batch_prompts)

        for j, raw in enumerate(raw_outputs):
            pair, output1_is_a = batch_meta[j]
            if raw is None:
                winner = None
                raw = "[SKIPPED: prompt too long]"
            else:
                winner = parse_pairwise_result(raw)
            pref = winner_to_preference(winner, output1_is_a)

            result_entry = {
                "instruction": pair["instruction"][:200],
                "generator": pair.get("generator"),
                "output1_is_a": output1_is_a,
                "raw_judge_output": raw,
                "preference": pref,
                "human_majority": pair["human_majority"],
            }
            results.append(result_entry)

        pbar.update(len(batch_pairs))
        current_count = start_idx + batch_start + len(batch_pairs)

        if current_count % checkpoint_every < batch_size:
            save_checkpoint(output_file, results)
            interim = compute_pairwise_metrics(results)
            logger.info(
                "Checkpoint %d -- agreement=%.1f%%  spearman=%s",
                current_count, interim["human_agreement_pct"], interim["spearman_corr"],
            )

    pbar.close()
    save_checkpoint(output_file, results)
    metrics = compute_pairwise_metrics(results)
    return output_file, metrics


# ---------------------------------------------------------------------------
# Pointwise evaluation
# ---------------------------------------------------------------------------

def compute_pointwise_metrics(results: list[dict]) -> dict:
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


def run_pointwise_eval(
    judge: Prometheus2Judge,
    items: list[dict],
    benchmark: str,
    output_dir: Path,
    *,
    use_dataset_rubric: bool = False,
    max_examples: int | None = None,
    checkpoint_every: int = 20,
    batch_size: int = 1,
) -> tuple[Path, dict]:
    if max_examples is not None:
        items = items[:max_examples]

    output_file = output_dir / "judge_results.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(items):
        logger.info("Already complete (%d results)", start_idx)
        return output_file, compute_pointwise_metrics(results)

    logger.info("Judging pointwise examples %d -> %d (batch_size=%d)", start_idx, len(items), batch_size)

    remaining = items[start_idx:]
    pbar = tqdm(total=len(items), desc="pointwise", initial=start_idx)

    for batch_start in range(0, len(remaining), batch_size):
        batch_items = remaining[batch_start:batch_start + batch_size]

        batch_prompts = []
        for item in batch_items:
            if use_dataset_rubric and item.get("rubric_criteria"):
                if benchmark == "healthbench":
                    rubric = _format_healthbench_criteria(item["rubric_criteria"])
                elif benchmark == "biggen_pointwise":
                    rubric = format_biggen_rubric(item["rubric_criteria"])
                else:
                    rubric = DEFAULT_POINTWISE_RUBRIC
            elif benchmark == "helpsteer2":
                rubric = HELPSTEER2_RUBRIC
            elif benchmark == "healthbench":
                rubric = HEALTHBENCH_RUBRIC
            else:
                rubric = DEFAULT_POINTWISE_RUBRIC

            user_msg = ABSOLUTE_PROMPT_WO_REF.format(
                instruction=item["prompt"],
                response=item["response"],
                rubric=rubric,
            )
            batch_prompts.append((ABS_SYSTEM_PROMPT, user_msg))

        raw_outputs = judge.generate_batch(batch_prompts)

        for j, raw in enumerate(raw_outputs):
            item = batch_items[j]
            if raw is None:
                score = None
                raw = "[SKIPPED: prompt too long]"
            else:
                score = parse_pointwise_result(raw)

            result_entry = {
                "prompt": item["prompt"][:200],
                "raw_judge_output": raw,
                "judge_score": score,
                "human_score": item["human_score"],
            }
            results.append(result_entry)

        pbar.update(len(batch_items))
        current_count = start_idx + batch_start + len(batch_items)

        if current_count % checkpoint_every < batch_size:
            save_checkpoint(output_file, results)
            interim = compute_pointwise_metrics(results)
            logger.info(
                "Checkpoint %d -- spearman=%s  pearson=%s",
                current_count, interim["spearman_corr"], interim["pearson_corr"],
            )

    pbar.close()
    save_checkpoint(output_file, results)
    metrics = compute_pointwise_metrics(results)
    return output_file, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Prometheus 2 as LLM judge on multiple benchmarks",
    )
    parser.add_argument(
        "--benchmark", type=str, required=True,
        choices=list(BENCHMARK_LOADERS.keys()),
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--model", type=str, default="prometheus-eval/prometheus-7b-v2.0",
        help="Prometheus 2 model name or path (default: prometheus-eval/prometheus-7b-v2.0)",
    )
    parser.add_argument(
        "--use-dataset-rubric", action="store_true",
        help="Use per-instance rubrics from the dataset when available "
             "(BiGGen score_rubric, HealthBench criteria)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Limit examples to evaluate (useful for testing)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Base directory for output files (default: outputs/)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default="data",
        help="Cache dir for dataset files (default: data/)",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=20,
        help="Save progress every N examples (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for position randomization (default: 42)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Max new tokens for generation (default: 1024)",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (default: 1.0, as per Prometheus 2 paper)",
    )
    parser.add_argument(
        "--device-map", type=str, default="auto",
        help="Device map for model loading (default: auto)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--backend", type=str, choices=["transformers", "vllm"], default="transformers",
        help="Inference backend: 'transformers' (default) or 'vllm' (much faster)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism with vLLM (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90,
        help="GPU memory utilization for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Maximum sequence length for vLLM KV cache (default: 4096)",
    )
    args = parser.parse_args()

    model_short_name = Path(args.model).name
    rubric_suffix = "instance_rubric" if args.use_dataset_rubric else "fixed_rubric"
    output_dir = (
        Path(args.output_dir) / "prometheus2" / args.benchmark
        / model_short_name / rubric_suffix
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Benchmark: %s | Model: %s | Output: %s",
        args.benchmark, args.model, output_dir,
    )

    # Load data
    data = BENCHMARK_LOADERS[args.benchmark](cache_dir)

    # Load model
    if args.backend == "vllm":
        judge = Prometheus2JudgeVLLM(
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        judge = Prometheus2Judge(
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device_map=args.device_map,
        )

    # Run evaluation
    if args.benchmark in PAIRWISE_BENCHMARKS:
        output_file, metrics = run_pairwise_eval(
            judge,
            data,
            output_dir,
            use_dataset_rubric=args.use_dataset_rubric,
            max_examples=args.max_examples,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    else:
        output_file, metrics = run_pointwise_eval(
            judge,
            data,
            args.benchmark,
            output_dir,
            use_dataset_rubric=args.use_dataset_rubric,
            max_examples=args.max_examples,
            checkpoint_every=args.checkpoint_every,
            batch_size=args.batch_size,
        )

    # Save summary
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"model": args.model, "benchmark": args.benchmark, "metrics": metrics},
            f, indent=2,
        )

    # Display results
    logger.info("=" * 70)
    logger.info("Results -- %s / %s", args.benchmark, model_short_name)
    logger.info("-" * 70)
    for k, v in metrics.items():
        logger.info("  %-25s %s", k, v)
    logger.info("=" * 70)
    logger.info("Results saved to %s", output_file)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
