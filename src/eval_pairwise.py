#!/usr/bin/env python3
"""
Unified pairwise evaluation of LLM judges against human annotations.

Supports three benchmarks:
    - alpacaeval: AlpacaFarm human cross-annotations (321 pairs, 5 annotators)
    - mtbench:    MT-Bench human judgments (pairwise)
    - biggen:     BiGGen Bench (synthetic pairwise from pointwise scores)

Rubric generation backends:
    - none:          No rubric generation (use fixed mode or pre-generated file)
    - api:           Rubric generated via Bedrock API (same model as judge)
    - transformers:  Local rubric generation via transformers/peft
    - vllm:          Local rubric generation via vLLM (fast batched inference)

Judge backends:
    - bedrock:  AWS Bedrock API (Claude, Llama, etc.)
    - vllm:     Local model via vLLM

Rubric modes:
    - fixed:                    Hardcoded generic pairwise rubric (no generation)
    - dynamic:                  Per-instance rubric (generated or loaded from file)
    - generated_fixed:          LLM-generated general-purpose rubric (one for all)
    - existing_fixed_instance:  Per-instance rubrics from dataset (BiGGen only)
    - local:                    Alias for dynamic with transformers/vllm backend

Usage examples:
    # API mode: MTBench with dynamic rubrics via Bedrock
    python src/eval_pairwise.py --benchmark mtbench --rubric-mode dynamic \\
        --judge claude-sonnet-4

    # Local rubric + API judge
    python src/eval_pairwise.py --benchmark alpacaeval --rubric-mode dynamic \\
        --rubric-backend vllm --rubric-model path/to/model \\
        --judge claude-sonnet-4

    # Fully local: vLLM rubric + vLLM judge
    python src/eval_pairwise.py --benchmark alpacaeval \\
        --rubric-backend vllm --rubric-model path/to/rubric_model \\
        --judge-backend vllm --judge-model path/to/judge_model \\
        --judge my-judge-name

    # BiGGen Bench with existing per-instance rubrics
    python src/eval_pairwise.py --benchmark biggen \\
        --rubric-mode existing_fixed_instance --judge claude-sonnet-4

    # Generate rubrics only (no judging)
    python src/eval_pairwise.py --benchmark mtbench --rubric-mode dynamic \\
        --rubric-backend transformers --rubric-model path/to/model --rubric-only

    # Quick test
    python src/eval_pairwise.py --benchmark mtbench --rubric-mode fixed \\
        --judge claude-sonnet-4 --max-examples 10
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

# ---------------------------------------------------------------------------
# Path setup and imports from shared modules
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from client import BedrockClient, call_judge, JUDGES  # noqa: E402
from vllm_judge import VLLMJudge  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts: Generated fixed rubric (used in "generated_fixed" mode)
# ---------------------------------------------------------------------------
GENERATED_FIXED_RUBRIC_SYSTEM_PROMPT = """\
You are an expert evaluator designing assessment criteria for comparing AI \
assistant responses. Your task is to create a general-purpose evaluation rubric \
that can be applied to ANY type of instruction-response pair.

The rubric should cover the key dimensions that distinguish a good AI response \
from a poor one, regardless of the specific task. Think about what universally \
matters when a human judges which of two AI responses is better.

Output ONLY a numbered list of 3-5 criteria. Each criterion should be one clear \
sentence describing what to evaluate."""

GENERATED_FIXED_RUBRIC_USER_PROMPT = """\
Create a general-purpose evaluation rubric for judging pairwise comparisons \
of AI assistant responses. The rubric should be applicable to any type of task \
(factual Q&A, creative writing, coding, reasoning, summarization, advice, etc.) \
without any task-specific or instance-specific information.

Write the evaluation rubric."""

# ---------------------------------------------------------------------------
# Prompts: Dynamic rubric generation (per-instance)
# ---------------------------------------------------------------------------
RUBRIC_SYSTEM_PROMPT = """\
You are an expert evaluator designing assessment criteria. Given a task \
instruction, create a concise evaluation rubric with 3-5 specific criteria \
for judging the quality of responses to this instruction.

Focus on what makes a response good or bad FOR THIS SPECIFIC TASK. Consider \
the task type (factual Q&A, creative writing, coding, reasoning, summarization, \
etc.) and what a high-quality answer requires.

Output ONLY a numbered list of criteria. Each criterion should be one clear sentence."""

RUBRIC_USER_TEMPLATE = """\
[Instruction]
{instruction}

Write the evaluation rubric for judging responses to the above instruction."""

# ---------------------------------------------------------------------------
# Prompts: Judging -- fixed rubric (no per-instance rubric)
# ---------------------------------------------------------------------------
FIXED_JUDGE_SYSTEM_PROMPT = """\
You are a helpful assistant that evaluates the quality of AI responses. \
You will be given an instruction and two responses (Response A and Response B). \
Compare them and decide which response better follows the instruction and is more \
helpful, accurate, and well-written.

You must output ONLY a single JSON object with two keys:
- "winner": either "A" or "B" (the letter of the better response)
- "reason": a brief one-sentence explanation

Example output:
{"winner": "A", "reason": "Response A is more detailed and directly addresses the question."}
"""

FIXED_JUDGE_USER_TEMPLATE = """\
[Instruction]
{instruction}

[Response A]
{response_a}

[Response B]
{response_b}

Which response is better? Output only the JSON object."""

# ---------------------------------------------------------------------------
# Prompts: Judging -- dynamic rubric (rubric injected per instance)
# ---------------------------------------------------------------------------
DYNAMIC_JUDGE_SYSTEM_PROMPT = """\
You are a helpful assistant that evaluates the quality of AI responses. \
You will be given an instruction, an evaluation rubric tailored to that \
instruction, and two responses (Response A and Response B). \
Use the rubric criteria to compare the responses and decide which one is better.

You must output ONLY a single JSON object with two keys:
- "winner": either "A" or "B" (the letter of the better response)
- "reason": a brief one-sentence explanation referencing the rubric criteria

Example output:
{"winner": "A", "reason": "Response A better satisfies the accuracy and completeness criteria."}
"""

DYNAMIC_JUDGE_USER_TEMPLATE = """\
[Instruction]
{instruction}

[Evaluation Rubric]
{rubric}

[Response A]
{response_a}

[Response B]
{response_b}

Based on the rubric, which response is better? Output only the JSON object."""


# ===========================================================================
# LOCAL RUBRIC GENERATORS (transformers / vLLM)
# ===========================================================================

class LocalRubricGenerator:
    """Load a local model (optionally with LoRA) for rubric generation.

    Supports Llama-3.1 and Qwen3 model families. For Qwen3 models,
    thinking mode is disabled to produce direct output.
    """

    def __init__(
        self,
        model_path: str,
        base_model: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device_map: str | dict = "cuda:0",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        adapter_config_path = Path(model_path) / "adapter_config.json"
        is_lora = adapter_config_path.exists()

        if is_lora:
            from peft import PeftModel

            actual_base = base_model
            if actual_base is None:
                with open(adapter_config_path) as f:
                    adapter_cfg = json.load(f)
                actual_base = adapter_cfg.get("base_model_name_or_path", None)
            if actual_base is None:
                raise ValueError(
                    "Cannot determine base model. Pass --base-model explicitly."
                )

            logger.info("Loading base model: %s", actual_base)
            base = AutoModelForCausalLM.from_pretrained(
                actual_base,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
            logger.info("Loading LoRA adapter: %s", model_path)
            self.model = PeftModel.from_pretrained(base, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(actual_base)
        else:
            logger.info("Loading model: %s", model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self._is_qwen3 = "qwen3" in self.tokenizer.name_or_path.lower()
        self.model.eval()

    def _build_messages(self, system: str, user_message: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    def _apply_chat_template(self, messages: list[dict]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._is_qwen3:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, system: str, user_message: str) -> str:
        import torch

        messages = self._build_messages(system, user_message)
        input_text = self._apply_chat_template(messages)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(
            self.model.device
        )

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

    def generate_batch(self, system: str, user_messages: list[str]) -> list[str]:
        """Generate completions for multiple inputs in one batched forward pass."""
        import torch

        texts = []
        for user_message in user_messages:
            messages = self._build_messages(system, user_message)
            texts.append(self._apply_chat_template(messages))

        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
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

        input_len = inputs["input_ids"].shape[1]
        results = []
        for seq in output_ids:
            new_tokens = seq[input_len:]
            results.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

        del inputs, output_ids
        torch.cuda.empty_cache()

        return results


class VLLMRubricGenerator:
    """Load a model (optionally with LoRA) via vLLM for rubric generation."""

    def __init__(
        self,
        model_path: str,
        base_model: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
    ):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        adapter_config_path = Path(model_path) / "adapter_config.json"
        is_lora = adapter_config_path.exists()

        if is_lora:
            actual_base = base_model
            if actual_base is None:
                with open(adapter_config_path) as f:
                    adapter_cfg = json.load(f)
                actual_base = adapter_cfg.get("base_model_name_or_path", None)
            if actual_base is None:
                raise ValueError(
                    "Cannot determine base model. Pass --base-model explicitly."
                )

            logger.info(
                "Loading vLLM with LoRA: base=%s, adapter=%s (tp=%d)",
                actual_base, model_path, tensor_parallel_size,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(actual_base)
            self.llm = LLM(
                model=actual_base,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype="bfloat16",
                trust_remote_code=True,
                enable_lora=True,
                max_lora_rank=128,
            )
            self.lora_request = LoRARequest("rubric_adapter", 1, model_path)
        else:
            logger.info(
                "Loading vLLM model: %s (tp=%d)", model_path, tensor_parallel_size,
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
            self.lora_request = None

        self._is_qwen3 = "qwen3" in self.tokenizer.name_or_path.lower()
        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 0,
        )
        logger.info("vLLM rubric generator loaded (qwen3=%s)", self._is_qwen3)

    def _build_messages(self, system: str, user_message: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    def _apply_chat_template(self, messages: list[dict]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._is_qwen3:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, system: str, user_message: str) -> str:
        messages = self._build_messages(system, user_message)
        input_text = self._apply_chat_template(messages)
        outputs = self.llm.generate(
            [input_text], self.sampling_params, lora_request=self.lora_request,
        )
        return outputs[0].outputs[0].text

    def generate_batch(self, system: str, user_messages: list[str]) -> list[str]:
        texts = []
        for user_message in user_messages:
            messages = self._build_messages(system, user_message)
            texts.append(self._apply_chat_template(messages))

        outputs = self.llm.generate(
            texts, self.sampling_params, lora_request=self.lora_request,
        )
        return [out.outputs[0].text for out in outputs]


# ===========================================================================
# HUGGINGFACE DATASET DOWNLOAD HELPER
# ===========================================================================

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
            set((f.get("config", "?"), f["split"]) for f in parquet_files)
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


# ===========================================================================
# DATASET LOADERS
# ===========================================================================

# ---------------------------------------------------------------------------
# AlpacaEval
# ---------------------------------------------------------------------------

HF_BASE = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main"


def _download_json(filename: str, cache_dir: Path) -> list[dict]:
    """Download a JSON file from HuggingFace or load from local cache."""
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
    """Load AlpacaFarm cross-annotations as pairwise evaluation pairs.

    Uses the eval split of alpaca_farm_human_crossannotations.json.
    Each unique (instruction, output_1, output_2) triple has 5 annotators.
    We compute the majority vote (random tie-break, seeded) as gold label.

    Returns list of dicts with keys:
        instruction, output_1, output_2, generator,
        human_majority (1 or 2), human_preferences (list[int])
    """
    ae_dir = cache_dir / "alpacaeval"
    ae_dir.mkdir(parents=True, exist_ok=True)

    # Try loading from local data directory first
    local_path = Path("data/alpacaeval/alpaca_farm_human_crossannotations.json")
    if local_path.exists():
        logger.info("Loading local AlpacaEval data from %s", local_path)
        with open(local_path) as f:
            cross_raw = json.load(f)
    else:
        cross_raw = _download_json(
            "alpaca_farm_human_crossannotations.json", ae_dir,
        )

    human_ann_raw = _download_json(
        "alpaca_farm_human_annotations.json", ae_dir,
    )

    # Build (instruction, output_1, output_2) -> generator from human annotations
    key_to_gen: dict[tuple[str, str, str], str] = {}
    for item in human_ann_raw:
        key = (item["instruction"], item["output_1"], item["output_2"])
        key_to_gen.setdefault(key, item["generator"])

    # Filter to eval split
    cross_eval = [a for a in cross_raw if a.get("datasplit") == "eval"]
    logger.info(
        "AlpacaEval cross-annotations: %d total, %d in eval split",
        len(cross_raw), len(cross_eval),
    )

    # Group preferences by unique pair
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

    # Compute majority vote per pair (random tie-break, seeded)
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

    generators = {p["generator"] for p in pairs}
    logger.info(
        "AlpacaEval: %d pairs | %d generators (%s)",
        len(pairs), len(generators), ", ".join(sorted(generators)),
    )
    return pairs


# ---------------------------------------------------------------------------
# MTBench
# ---------------------------------------------------------------------------

def load_mtbench(cache_dir: Path) -> list[dict]:
    """Load MTBench human judgments as pairwise evaluation pairs.

    Returns list of dicts with keys:
        instruction, output_1, output_2, generator (model_b),
        human_majority (1 or 2), human_preferences
    """
    bm_dir = cache_dir / "mtbench"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "lmsys/mt_bench_human_judgments", "human", bm_dir,
    )
    logger.info("MTBench columns: %s", list(df.columns))
    logger.info("MTBench shape: %s", df.shape)

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

        human_majority = 1 if majority_winner == "model_a" else 2

        entry = dict(group_data[key])
        entry["human_majority"] = human_majority
        entry["human_preferences"] = winners
        entry["generator"] = entry["model_b"]
        pairs.append(entry)

    generators = {p["generator"] for p in pairs}
    logger.info(
        "MTBench: %d pairs (%d ties excluded) | %d generators (%s)",
        len(pairs), n_ties, len(generators),
        ", ".join(sorted(generators)),
    )
    return pairs


# ---------------------------------------------------------------------------
# BiGGen Bench
# ---------------------------------------------------------------------------

def load_biggen(cache_dir: Path) -> list[dict]:
    """Load BiGGen Bench as synthetic pairwise evaluation pairs.

    BiGGen Bench is pointwise (1-5 human scores per model per prompt).
    We construct synthetic pairwise pairs by comparing models on the same
    prompt and using the higher human score as the preferred response.

    Returns list of dicts with keys:
        instruction, output_1, output_2, generator (model of output_2),
        human_majority (1 or 2), score_rubric (dict, per-instance rubric)
    """
    bm_dir = cache_dir / "biggen"
    bm_dir.mkdir(parents=True, exist_ok=True)

    df = _download_hf_dataset(
        "prometheus-eval/BiGGen-Bench-Results", "human_eval", bm_dir,
    )
    logger.info("BiGGen Bench columns: %s", list(df.columns))
    logger.info("BiGGen Bench shape: %s", df.shape)

    # Group by prompt -- collect (model, response, score, rubric)
    prompt_data: dict[str, list[dict]] = defaultdict(list)
    prompt_rubrics: dict[str, dict] = {}

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

        # Store the per-instance score_rubric
        raw_rubric = row.get("score_rubric")
        if raw_rubric is not None and prompt not in prompt_rubrics:
            if isinstance(raw_rubric, str):
                try:
                    prompt_rubrics[prompt] = json.loads(raw_rubric)
                except json.JSONDecodeError:
                    pass
            elif isinstance(raw_rubric, dict):
                prompt_rubrics[prompt] = dict(raw_rubric)

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

            # Attach per-instance rubric if available
            if prompt in prompt_rubrics:
                entry["score_rubric"] = prompt_rubrics[prompt]

            pairs.append(entry)

    generators = {p["generator"] for p in pairs}
    logger.info(
        "BiGGen Bench: %d pairs (%d ties excluded) | %d generators (%s)",
        len(pairs), n_ties, len(generators),
        ", ".join(sorted(generators)),
    )
    return pairs


# ---------------------------------------------------------------------------
# Benchmark loader registry
# ---------------------------------------------------------------------------

BENCHMARK_LOADERS = {
    "alpacaeval": load_alpacaeval,
    "mtbench": load_mtbench,
    "biggen": load_biggen,
}

RUBRIC_MODES = [
    "fixed", "dynamic", "generated_fixed",
    "existing_fixed_instance", "local",
]


# ===========================================================================
# FORMAT HELPERS
# ===========================================================================

def format_biggen_rubric(score_rubric: dict) -> str:
    """Format a BiGGen Bench score_rubric dict into readable rubric text."""
    parts = []
    criteria = score_rubric.get("criteria", "")
    if criteria:
        parts.append(f"Evaluation Criteria: {criteria}")
        parts.append("")

    for i in range(1, 6):
        key = f"score{i}_description"
        desc = score_rubric.get(key, "")
        if desc:
            parts.append(f"Score {i}: {desc}")

    return "\n".join(parts)


# ===========================================================================
# CHECKPOINT HELPERS
# ===========================================================================

def load_checkpoint(path: Path) -> list[dict]:
    """Load existing judgments from checkpoint file."""
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        logger.info("Resuming from checkpoint -- %d judgments", len(data))
        return data
    return []


def save_checkpoint(path: Path, results: list[dict]) -> None:
    """Save judgments to checkpoint file."""
    with open(path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ===========================================================================
# RUBRIC GENERATION
# ===========================================================================

def generate_rubrics_api(
    client: BedrockClient,
    judge_name: str,
    model_id: str,
    instructions: list[str],
    output_dir: Path,
    *,
    temperature: float = 0.0,
) -> dict[str, str]:
    """Generate per-instruction rubrics via Bedrock API and cache to disk."""
    rubric_file = output_dir / f"rubrics_{judge_name}.json"

    rubrics: dict[str, str] = {}
    if rubric_file.exists():
        with open(rubric_file) as f:
            rubrics = json.load(f)
        logger.info(
            "[%s] Loaded %d cached rubrics from %s",
            judge_name, len(rubrics), rubric_file,
        )

    remaining = [inst for inst in instructions if inst not in rubrics]
    if not remaining:
        logger.info("[%s] All %d rubrics already generated", judge_name, len(instructions))
        return rubrics

    logger.info(
        "[%s] Generating rubrics for %d instructions (model_id=%s)",
        judge_name, len(remaining), model_id,
    )

    for i, instruction in enumerate(
        tqdm(remaining, desc=f"{judge_name} rubrics"),
    ):
        user_msg = RUBRIC_USER_TEMPLATE.format(instruction=instruction)
        rubric = call_judge(
            client,
            model_id,
            system=RUBRIC_SYSTEM_PROMPT,
            user_message=user_msg,
            max_tokens=512,
            temperature=temperature,
        )
        rubrics[instruction] = rubric

        if (i + 1) % 50 == 0:
            with open(rubric_file, "w") as f:
                json.dump(rubrics, f, indent=2, ensure_ascii=False)

    with open(rubric_file, "w") as f:
        json.dump(rubrics, f, indent=2, ensure_ascii=False)
    logger.info("[%s] Saved %d rubrics -> %s", judge_name, len(rubrics), rubric_file)
    return rubrics


def generate_fixed_rubric_api(
    client: BedrockClient,
    judge_name: str,
    model_id: str,
    output_dir: Path,
    *,
    temperature: float = 0.0,
) -> str:
    """Generate a single general-purpose rubric via Bedrock API."""
    rubric_file = output_dir / f"fixed_rubric_{judge_name}.txt"

    if rubric_file.exists():
        rubric = rubric_file.read_text().strip()
        logger.info("[%s] Loaded cached generated fixed rubric from %s", judge_name, rubric_file)
        return rubric

    logger.info(
        "[%s] Generating fixed rubric (model_id=%s)", judge_name, model_id,
    )
    rubric = call_judge(
        client,
        model_id,
        system=GENERATED_FIXED_RUBRIC_SYSTEM_PROMPT,
        user_message=GENERATED_FIXED_RUBRIC_USER_PROMPT,
        max_tokens=512,
        temperature=temperature,
    )

    rubric_file.write_text(rubric)
    logger.info("[%s] Saved generated fixed rubric -> %s", judge_name, rubric_file)
    return rubric


def generate_rubrics_local(
    generator,
    instructions: list[str],
    output_dir: Path,
    rubric_label: str,
    batch_size: int = 8,
) -> dict[str, str]:
    """Generate per-instruction rubrics using a local model (transformers or vLLM).

    The generator must implement .generate_batch(system, user_messages) -> list[str].
    """
    rubric_file = output_dir / f"rubrics_{rubric_label}.json"

    rubrics: dict[str, str] = {}
    if rubric_file.exists():
        with open(rubric_file) as f:
            rubrics = json.load(f)
        logger.info("Loaded %d cached rubrics from %s", len(rubrics), rubric_file)

    remaining = [inst for inst in instructions if inst not in rubrics]
    if not remaining:
        logger.info("All %d rubrics already generated", len(instructions))
        return rubrics

    logger.info(
        "Generating rubrics for %d instructions (batch_size=%d)",
        len(remaining), batch_size,
    )

    for batch_start in tqdm(
        range(0, len(remaining), batch_size), desc="rubric generation",
    ):
        batch_instructions = remaining[batch_start:batch_start + batch_size]
        user_messages = [
            RUBRIC_USER_TEMPLATE.format(instruction=inst)
            for inst in batch_instructions
        ]

        batch_results = generator.generate_batch(RUBRIC_SYSTEM_PROMPT, user_messages)

        for instruction, rubric in zip(batch_instructions, batch_results):
            rubrics[instruction] = rubric

        if (batch_start // batch_size + 1) % 10 == 0:
            with open(rubric_file, "w") as f:
                json.dump(rubrics, f, indent=2, ensure_ascii=False)

    with open(rubric_file, "w") as f:
        json.dump(rubrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d rubrics -> %s", len(rubrics), rubric_file)
    return rubrics


# ===========================================================================
# PARSE JUDGE OUTPUT
# ===========================================================================

def parse_winner(raw: str) -> str | None:
    """Extract 'A' or 'B' from the judge's JSON output."""
    m = re.search(r'"winner"\s*:\s*"([AB])"', raw)
    if m:
        return m.group(1)
    m = re.search(r'\b([AB])\b', raw)
    return m.group(1) if m else None


def winner_to_preference(winner: str | None, output1_is_a: bool) -> int | None:
    """Convert A/B winner to preference 1 or 2.

    preference=1 means output_1 preferred, preference=2 means output_2 preferred.
    """
    if winner is None:
        return None
    if output1_is_a:
        return 1 if winner == "A" else 2
    else:
        return 2 if winner == "A" else 1


# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(results: list[dict]) -> dict:
    """Compute human agreement, Spearman corr., and Pearson corr.

    Correlation is computed over per-generator win rates (judge vs gold).
    """
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
            "n_generators": 0,
        }

    # --- 1. Human agreement (per-example accuracy vs majority vote) ---
    n_agree = sum(1 for r in valid if r["preference"] == r["human_majority"])
    human_agreement = n_agree / n_valid * 100

    # --- 2 & 3. Spearman / Pearson over per-generator win rates ---
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


# ===========================================================================
# JUDGING LOOPS
# ===========================================================================

def _build_judge_prompt(
    pair: dict,
    output1_is_a: bool,
    rubric_mode: str,
    rubrics: dict[str, str],
    generated_fixed_rubric: str | None,
) -> tuple[str, str, str | None]:
    """Build the system/user prompt for a single judgment.

    Returns (system_prompt, user_message, rubric_text_or_None).
    """
    if output1_is_a:
        resp_a, resp_b = pair["output_1"], pair["output_2"]
    else:
        resp_a, resp_b = pair["output_2"], pair["output_1"]

    if rubric_mode in ("dynamic", "local"):
        rubric = rubrics.get(
            pair["instruction"],
            "Evaluate overall helpfulness, accuracy, and clarity.",
        )
        system_prompt = DYNAMIC_JUDGE_SYSTEM_PROMPT
        user_msg = DYNAMIC_JUDGE_USER_TEMPLATE.format(
            instruction=pair["instruction"],
            rubric=rubric,
            response_a=resp_a,
            response_b=resp_b,
        )
    elif rubric_mode == "generated_fixed":
        rubric = generated_fixed_rubric
        system_prompt = DYNAMIC_JUDGE_SYSTEM_PROMPT
        user_msg = DYNAMIC_JUDGE_USER_TEMPLATE.format(
            instruction=pair["instruction"],
            rubric=rubric,
            response_a=resp_a,
            response_b=resp_b,
        )
    elif rubric_mode == "existing_fixed_instance":
        score_rubric = pair.get("score_rubric")
        if score_rubric:
            rubric = format_biggen_rubric(score_rubric)
        else:
            rubric = "Evaluate overall helpfulness, accuracy, and clarity."
        system_prompt = DYNAMIC_JUDGE_SYSTEM_PROMPT
        user_msg = DYNAMIC_JUDGE_USER_TEMPLATE.format(
            instruction=pair["instruction"],
            rubric=rubric,
            response_a=resp_a,
            response_b=resp_b,
        )
    else:
        # fixed mode
        rubric = None
        system_prompt = FIXED_JUDGE_SYSTEM_PROMPT
        user_msg = FIXED_JUDGE_USER_TEMPLATE.format(
            instruction=pair["instruction"],
            response_a=resp_a,
            response_b=resp_b,
        )

    return system_prompt, user_msg, rubric


# ---------------------------------------------------------------------------
# Bedrock API judge loop (with ThreadPoolExecutor for parallelism)
# ---------------------------------------------------------------------------

def _judge_one_pair_api(
    client: BedrockClient,
    judge_config: dict,
    pair: dict,
    output1_is_a: bool,
    rubric_mode: str,
    rubrics: dict[str, str],
    generated_fixed_rubric: str | None,
) -> dict:
    """Evaluate a single pair via Bedrock API (thread-safe). Returns result dict."""
    system_prompt, user_msg, rubric = _build_judge_prompt(
        pair, output1_is_a, rubric_mode, rubrics, generated_fixed_rubric,
    )

    raw = call_judge(
        client,
        judge_config["model_id"],
        system=system_prompt,
        user_message=user_msg,
        max_tokens=judge_config["max_tokens"],
        temperature=judge_config["temperature"],
    )

    winner = parse_winner(raw)
    pref = winner_to_preference(winner, output1_is_a)

    result_entry = {
        "instruction": pair["instruction"],
        "generator": pair.get("generator"),
        "output1_is_a": output1_is_a,
        "raw_judge_output": raw,
        "preference": pref,
        "human_majority": pair["human_majority"],
    }
    if "human_preferences" in pair:
        result_entry["human_preferences"] = pair["human_preferences"]
    if rubric is not None:
        result_entry["rubric"] = rubric
    return result_entry


def run_judge_bedrock(
    client: BedrockClient,
    judge_name: str,
    judge_config: dict,
    pairs: list[dict],
    rubrics: dict[str, str],
    generated_fixed_rubric: str | None,
    output_dir: Path,
    *,
    rubric_mode: str = "dynamic",
    max_examples: int | None = None,
    checkpoint_every: int = 20,
    seed: int = 42,
    max_workers: int = 4,
) -> tuple[Path, dict]:
    """Run Bedrock API judge with parallel execution via ThreadPoolExecutor."""
    if max_examples is not None:
        pairs = pairs[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(pairs):
        logger.info("[%s] Already complete (%d judgments)", judge_name, start_idx)
        return output_file, compute_metrics(results)

    logger.info(
        "[%s] Judging examples %d -> %d  (mode=%s, model_id=%s, workers=%d)",
        judge_name, start_idx, len(pairs), rubric_mode,
        judge_config["model_id"], max_workers,
    )

    # Pre-generate position assignments (seeded) so checkpoint-resume is stable
    rng = random.Random(seed)
    position_assignments = [rng.choice([True, False]) for _ in range(len(pairs))]

    remaining_pairs = pairs[start_idx:]
    remaining_positions = position_assignments[start_idx:]
    batch_size = max_workers * 2
    pbar = tqdm(total=len(pairs), desc=judge_name, initial=start_idx)

    for batch_start in range(0, len(remaining_pairs), batch_size):
        batch = remaining_pairs[batch_start:batch_start + batch_size]
        batch_positions = remaining_positions[batch_start:batch_start + batch_size]
        batch_results = [None] * len(batch)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for j, (pair, pos) in enumerate(zip(batch, batch_positions)):
                future = executor.submit(
                    _judge_one_pair_api,
                    client, judge_config, pair, pos,
                    rubric_mode, rubrics, generated_fixed_rubric,
                )
                future_to_idx[future] = j

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                batch_results[idx] = future.result()
                pbar.update(1)

        results.extend(batch_results)

        global_idx = start_idx + batch_start + len(batch)
        if global_idx % checkpoint_every < batch_size or batch_start + batch_size >= len(remaining_pairs):
            save_checkpoint(output_file, results)
            interim = compute_metrics(results)
            logger.info(
                "[%s] Checkpoint %d -- agreement=%.1f%%  spearman=%s  pearson=%s",
                judge_name, global_idx,
                interim["human_agreement_pct"],
                interim["spearman_corr"],
                interim["pearson_corr"],
            )

    pbar.close()
    save_checkpoint(output_file, results)
    metrics = compute_metrics(results)
    logger.info("[%s] Done -- %d judgments -> %s", judge_name, len(results), output_file)
    return output_file, metrics


# ---------------------------------------------------------------------------
# vLLM judge loop (batched local inference)
# ---------------------------------------------------------------------------

def run_judge_vllm(
    vllm_judge: VLLMJudge,
    judge_name: str,
    pairs: list[dict],
    rubrics: dict[str, str],
    generated_fixed_rubric: str | None,
    output_dir: Path,
    *,
    rubric_mode: str = "dynamic",
    max_examples: int | None = None,
    batch_size: int = 32,
    checkpoint_every: int = 100,
    seed: int = 42,
) -> tuple[Path, dict]:
    """Run vLLM judge over all pairs in batches."""
    if max_examples is not None:
        pairs = pairs[:max_examples]

    output_file = output_dir / f"judge_{judge_name}.json"
    results = load_checkpoint(output_file)
    start_idx = len(results)

    if start_idx >= len(pairs):
        logger.info("[%s] Already complete (%d judgments)", judge_name, start_idx)
        return output_file, compute_metrics(results)

    logger.info(
        "[%s] Judging %d -> %d (vLLM, mode=%s, batch_size=%d)",
        judge_name, start_idx, len(pairs), rubric_mode, batch_size,
    )

    rng = random.Random(seed)
    position_assignments = [rng.choice([True, False]) for _ in range(len(pairs))]

    remaining = pairs[start_idx:]
    for batch_start in tqdm(
        range(0, len(remaining), batch_size),
        desc=judge_name,
        total=(len(remaining) + batch_size - 1) // batch_size,
    ):
        batch = remaining[batch_start:batch_start + batch_size]
        prompts_for_batch = []

        for j, pair in enumerate(batch):
            global_idx = start_idx + batch_start + j
            output1_is_a = position_assignments[global_idx]

            system_prompt, user_msg, _ = _build_judge_prompt(
                pair, output1_is_a, rubric_mode, rubrics, generated_fixed_rubric,
            )
            prompts_for_batch.append((system_prompt, user_msg))

        raw_outputs = vllm_judge.generate_batch(prompts_for_batch, batch_size=batch_size)

        for j, (pair, raw) in enumerate(zip(batch, raw_outputs)):
            global_idx = start_idx + batch_start + j
            output1_is_a = position_assignments[global_idx]

            if raw is None:
                raw = ""
            winner = parse_winner(raw)
            pref = winner_to_preference(winner, output1_is_a)

            # Determine rubric text for output
            rubric_text = None
            if rubric_mode in ("dynamic", "local"):
                rubric_text = rubrics.get(pair["instruction"])
            elif rubric_mode == "existing_fixed_instance":
                score_rubric = pair.get("score_rubric")
                if score_rubric:
                    rubric_text = format_biggen_rubric(score_rubric)
            elif rubric_mode == "generated_fixed":
                rubric_text = generated_fixed_rubric

            result_entry = {
                "instruction": pair["instruction"],
                "generator": pair.get("generator"),
                "output1_is_a": output1_is_a,
                "raw_judge_output": raw,
                "preference": pref,
                "human_majority": pair["human_majority"],
            }
            if "human_preferences" in pair:
                result_entry["human_preferences"] = pair["human_preferences"]
            if rubric_text is not None:
                result_entry["rubric"] = rubric_text
            results.append(result_entry)

        global_idx_end = start_idx + batch_start + len(batch)
        if global_idx_end % checkpoint_every < batch_size or batch_start + batch_size >= len(remaining):
            save_checkpoint(output_file, results)
            interim = compute_metrics(results)
            logger.info(
                "[%s] Checkpoint %d -- agreement=%.1f%%  spearman=%s  pearson=%s",
                judge_name, global_idx_end,
                interim["human_agreement_pct"],
                interim["spearman_corr"],
                interim["pearson_corr"],
            )

    save_checkpoint(output_file, results)
    metrics = compute_metrics(results)
    logger.info("[%s] Done -- %d judgments -> %s", judge_name, len(results), output_file)
    return output_file, metrics


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified pairwise evaluation of LLM judges",
    )

    # --- Core arguments ---
    parser.add_argument(
        "--benchmark", type=str, required=True,
        choices=list(BENCHMARK_LOADERS.keys()),
        help="Benchmark to evaluate on",
    )
    parser.add_argument(
        "--rubric-mode", type=str, choices=RUBRIC_MODES, default="dynamic",
        help="Rubric mode: 'fixed' (hardcoded), 'dynamic' (per-instance), "
             "'generated_fixed' (LLM-generated general), "
             "'existing_fixed_instance' (dataset rubrics, BiGGen only), "
             "'local' (alias for dynamic with local backend) "
             "(default: dynamic)",
    )
    parser.add_argument(
        "--judge", type=str, nargs="+", required=True,
        help="Judge model name(s). For bedrock: claude-sonnet-4, llama-3.1-8b, "
             "llama-3.1-70b. For vllm: any custom name.",
    )

    # --- Rubric backend arguments ---
    parser.add_argument(
        "--rubric-backend", type=str, default="none",
        choices=["none", "api", "transformers", "vllm"],
        help="Rubric generation backend: 'none' (use fixed/pre-generated), "
             "'api' (Bedrock), 'transformers' (local HF model), "
             "'vllm' (local vLLM) (default: none)",
    )
    parser.add_argument(
        "--rubric-model", type=str, default=None,
        help="Path to rubric model (for transformers/vllm rubric backends). "
             "Can be a LoRA adapter dir or full model path.",
    )
    parser.add_argument(
        "--base-model", type=str, default=None,
        help="Base model for LoRA rubric adapters. Auto-detected from "
             "adapter_config.json if not specified.",
    )
    parser.add_argument(
        "--rubric-file", type=str, default=None,
        help="Path to pre-generated rubrics JSON. Skips rubric generation.",
    )
    parser.add_argument(
        "--rubric-temperature", type=float, default=0.0,
        help="Temperature for rubric generation (default: 0.0)",
    )
    parser.add_argument(
        "--rubric-max-tokens", type=int, default=512,
        help="Max new tokens for rubric generation (default: 512)",
    )
    parser.add_argument(
        "--rubric-batch-size", type=int, default=8,
        help="Batch size for local rubric generation (default: 8)",
    )
    parser.add_argument(
        "--rubric-tensor-parallel-size", type=int, default=1,
        help="Tensor parallelism for vLLM rubric backend (default: 1)",
    )
    parser.add_argument(
        "--rubric-max-model-len", type=int, default=8192,
        help="Max model length for vLLM rubric backend (default: 8192)",
    )
    parser.add_argument(
        "--rubric-gpu-memory-utilization", type=float, default=0.90,
        help="GPU memory utilization for vLLM rubric backend (default: 0.90)",
    )
    parser.add_argument(
        "--rubric-only", action="store_true",
        help="Only generate rubrics and exit (skip judge evaluation)",
    )

    # --- Judge backend arguments ---
    parser.add_argument(
        "--judge-backend", type=str, default="bedrock",
        choices=["bedrock", "vllm"],
        help="Judge backend: 'bedrock' (API) or 'vllm' (local) (default: bedrock)",
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
        help="Model path for vLLM judge backend (e.g. Qwen/Qwen3-14B). "
             "Required when --judge-backend=vllm.",
    )
    parser.add_argument(
        "--judge-tensor-parallel-size", type=int, default=1,
        help="Tensor parallelism for vLLM judge (default: 1)",
    )
    parser.add_argument(
        "--judge-max-model-len", type=int, default=8192,
        help="Max model length for vLLM judge (default: 8192)",
    )
    parser.add_argument(
        "--judge-gpu-memory-utilization", type=float, default=0.90,
        help="GPU memory utilization for vLLM judge (default: 0.90)",
    )
    parser.add_argument(
        "--judge-batch-size", type=int, default=32,
        help="Batch size for vLLM judge inference (default: 32)",
    )

    # --- General arguments ---
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Limit pairs to evaluate (useful for testing)",
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
        "--region", type=str, default=None,
        help="AWS region for Bedrock (default: us-east-1 or $AWS_REGION)",
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
        "--max-workers", type=int, default=4,
        help="Parallel threads for Bedrock API judge calls (default: 4)",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------
    # Normalize "local" rubric mode to "dynamic" with local backend
    effective_rubric_mode = args.rubric_mode
    if effective_rubric_mode == "local":
        effective_rubric_mode = "dynamic"
        if args.rubric_backend == "none":
            # Infer rubric backend from rubric-model availability
            if args.rubric_model:
                args.rubric_backend = "transformers"
            elif args.rubric_file:
                args.rubric_backend = "none"
            else:
                parser.error(
                    "--rubric-mode=local requires --rubric-model or --rubric-file"
                )

    if effective_rubric_mode == "existing_fixed_instance" and args.benchmark != "biggen":
        parser.error(
            "existing_fixed_instance rubric mode is only available for biggen "
            "(other benchmarks have no per-instance rubrics)"
        )

    if args.judge_backend == "vllm" and not args.judge_model:
        parser.error("--judge-model is required when using --judge-backend=vllm")

    if args.judge_backend == "bedrock":
        for jname in args.judge:
            if jname not in JUDGES:
                parser.error(
                    f"Unknown Bedrock judge '{jname}'. "
                    f"Available: {list(JUDGES.keys())}"
                )

    if effective_rubric_mode == "dynamic" and args.rubric_backend in ("transformers", "vllm"):
        if not args.rubric_model and not args.rubric_file:
            parser.error(
                "--rubric-model or --rubric-file is required for dynamic rubric "
                "generation with transformers/vllm backend"
            )

    # Auto-set rubric-backend to "api" when dynamic mode + bedrock judge + no local model
    if (effective_rubric_mode == "dynamic"
            and args.rubric_backend == "none"
            and not args.rubric_file
            and args.rubric_model is None):
        args.rubric_backend = "api"

    # -----------------------------------------------------------------------
    # Output directory
    # -----------------------------------------------------------------------
    # Derive rubric label for output path
    if args.rubric_file:
        rubric_label = Path(args.rubric_file).stem
    elif args.rubric_model:
        rubric_label = Path(args.rubric_model).name
    else:
        rubric_label = effective_rubric_mode

    output_dir = Path(args.output_dir) / "pairwise" / args.benchmark / rubric_label
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Benchmark: %s | Rubric mode: %s | Rubric backend: %s | "
        "Judge backend: %s | Judge(s): %s | Output: %s",
        args.benchmark, effective_rubric_mode, args.rubric_backend,
        args.judge_backend, ", ".join(args.judge), output_dir,
    )

    # -----------------------------------------------------------------------
    # 1. Load benchmark data
    # -----------------------------------------------------------------------
    pairs = BENCHMARK_LOADERS[args.benchmark](cache_dir)

    # -----------------------------------------------------------------------
    # 2. Rubric generation / loading
    # -----------------------------------------------------------------------
    rubrics: dict[str, str] = {}
    generated_fixed_rubric: str | None = None

    if effective_rubric_mode == "dynamic":
        if args.rubric_file:
            # Load pre-generated rubrics from file
            logger.info("Loading pre-generated rubrics from: %s", args.rubric_file)
            with open(args.rubric_file) as f:
                rubrics = json.load(f)
            logger.info("Loaded %d rubrics", len(rubrics))

        elif args.rubric_backend == "api":
            # Generate rubrics via Bedrock API
            client = BedrockClient(region=args.region)
            effective_pairs = pairs[:args.max_examples] if args.max_examples else pairs
            unique_instructions = list(
                dict.fromkeys(p["instruction"] for p in effective_pairs)
            )
            # Use first judge model for rubric generation
            judge_for_rubric = args.judge[0]
            rubrics = generate_rubrics_api(
                client,
                judge_for_rubric,
                JUDGES[judge_for_rubric]["model_id"],
                unique_instructions,
                output_dir,
                temperature=JUDGES[judge_for_rubric]["temperature"],
            )

        elif args.rubric_backend == "transformers":
            # Generate rubrics via local transformers model
            logger.info("Loading transformers rubric generator: %s", args.rubric_model)
            generator = LocalRubricGenerator(
                model_path=args.rubric_model,
                base_model=args.base_model,
                max_new_tokens=args.rubric_max_tokens,
                temperature=args.rubric_temperature,
            )

            effective_pairs = pairs[:args.max_examples] if args.max_examples else pairs
            unique_instructions = list(
                dict.fromkeys(p["instruction"] for p in effective_pairs)
            )
            rubrics = generate_rubrics_local(
                generator, unique_instructions, output_dir, rubric_label,
                batch_size=args.rubric_batch_size,
            )

            # Free GPU memory
            del generator
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Rubric generation complete. GPU memory freed.")

        elif args.rubric_backend == "vllm":
            # Generate rubrics via vLLM
            logger.info("Loading vLLM rubric generator: %s", args.rubric_model)
            generator = VLLMRubricGenerator(
                model_path=args.rubric_model,
                base_model=args.base_model,
                max_new_tokens=args.rubric_max_tokens,
                temperature=args.rubric_temperature,
                tensor_parallel_size=args.rubric_tensor_parallel_size,
                gpu_memory_utilization=args.rubric_gpu_memory_utilization,
                max_model_len=args.rubric_max_model_len,
            )

            effective_pairs = pairs[:args.max_examples] if args.max_examples else pairs
            unique_instructions = list(
                dict.fromkeys(p["instruction"] for p in effective_pairs)
            )
            rubrics = generate_rubrics_local(
                generator, unique_instructions, output_dir, rubric_label,
                batch_size=args.rubric_batch_size,
            )

            # Free GPU memory
            if hasattr(generator, "llm"):
                del generator.llm
            del generator
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Rubric generation complete. GPU memory freed.")

    elif effective_rubric_mode == "generated_fixed":
        if args.rubric_backend == "api" or args.rubric_backend == "none":
            client = BedrockClient(region=args.region)
            judge_for_rubric = args.judge[0]
            generated_fixed_rubric = generate_fixed_rubric_api(
                client,
                judge_for_rubric,
                JUDGES[judge_for_rubric]["model_id"],
                output_dir,
                temperature=JUDGES[judge_for_rubric]["temperature"],
            )
        elif args.rubric_backend == "vllm":
            logger.info("Loading vLLM rubric generator for fixed rubric: %s", args.rubric_model)
            generator = VLLMRubricGenerator(
                model_path=args.rubric_model,
                base_model=args.base_model,
                max_new_tokens=args.rubric_max_tokens,
                temperature=args.rubric_temperature,
                tensor_parallel_size=args.rubric_tensor_parallel_size,
                gpu_memory_utilization=args.rubric_gpu_memory_utilization,
                max_model_len=args.rubric_max_model_len,
            )
            generated_fixed_rubric = generator.generate(
                GENERATED_FIXED_RUBRIC_SYSTEM_PROMPT,
                GENERATED_FIXED_RUBRIC_USER_PROMPT,
            )
            del generator
            import gc
            gc.collect()
            logger.info("Generated fixed rubric via vLLM")
        elif args.rubric_backend == "transformers":
            logger.info("Loading transformers rubric generator for fixed rubric: %s", args.rubric_model)
            generator = LocalRubricGenerator(
                model_path=args.rubric_model,
                base_model=args.base_model,
                max_new_tokens=args.rubric_max_tokens,
                temperature=args.rubric_temperature,
            )
            generated_fixed_rubric = generator.generate(
                GENERATED_FIXED_RUBRIC_SYSTEM_PROMPT,
                GENERATED_FIXED_RUBRIC_USER_PROMPT,
            )
            del generator
            import gc
            gc.collect()
            logger.info("Generated fixed rubric via transformers")

    # -----------------------------------------------------------------------
    # 2b. Rubric-only mode: exit after generation
    # -----------------------------------------------------------------------
    if args.rubric_only:
        logger.info("Rubric-only mode: skipping judge evaluation.")
        logger.info("Rubrics saved to: %s", output_dir)
        return

    # -----------------------------------------------------------------------
    # 3. Run judge(s)
    # -----------------------------------------------------------------------
    all_metrics = {}

    if args.judge_backend == "vllm":
        vllm_judge = VLLMJudge(
            model_path=args.judge_model,
            max_new_tokens=256,
            temperature=0.0,
            tensor_parallel_size=args.judge_tensor_parallel_size,
            gpu_memory_utilization=args.judge_gpu_memory_utilization,
            max_model_len=args.judge_max_model_len,
        )

        for judge_name in args.judge:
            _, metrics = run_judge_vllm(
                vllm_judge,
                judge_name,
                pairs,
                rubrics,
                generated_fixed_rubric,
                output_dir,
                rubric_mode=effective_rubric_mode,
                max_examples=args.max_examples,
                batch_size=args.judge_batch_size,
                checkpoint_every=args.checkpoint_every,
                seed=args.seed,
            )
            all_metrics[judge_name] = metrics

        # Cleanup vLLM judge
        del vllm_judge.llm
        del vllm_judge
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    else:
        # Bedrock API judge
        client = BedrockClient(region=args.region)

        for judge_name in args.judge:
            _, metrics = run_judge_bedrock(
                client,
                judge_name,
                JUDGES[judge_name],
                pairs,
                rubrics,
                generated_fixed_rubric,
                output_dir,
                rubric_mode=effective_rubric_mode,
                max_examples=args.max_examples,
                checkpoint_every=args.checkpoint_every,
                seed=args.seed,
                max_workers=args.max_workers,
            )
            all_metrics[judge_name] = metrics

    # -----------------------------------------------------------------------
    # 4. Save and display results
    # -----------------------------------------------------------------------
    summary_path = output_dir / "metrics_summary.json"
    existing_metrics = {}
    if summary_path.exists():
        with open(summary_path) as f:
            existing_metrics = json.load(f)
    existing_metrics.update(all_metrics)
    with open(summary_path, "w") as f:
        json.dump(existing_metrics, f, indent=2)

    logger.info("=" * 75)
    logger.info(
        "Results -- %s / %s / rubric_backend=%s",
        args.benchmark, effective_rubric_mode, args.rubric_backend,
    )
    logger.info(
        "%-20s  %10s  %10s  %10s  %5s  %6s",
        "Judge", "Agreement", "Spearman", "Pearson", "Gens", "Errors",
    )
    logger.info("-" * 75)
    for judge_name, metrics in all_metrics.items():
        logger.info(
            "%-20s  %9.2f%%  %10s  %10s  %5d  %6d",
            judge_name,
            metrics["human_agreement_pct"],
            f'{metrics["spearman_corr"]:.4f}' if metrics["spearman_corr"] is not None else "N/A",
            f'{metrics["pearson_corr"]:.4f}' if metrics["pearson_corr"] is not None else "N/A",
            metrics["n_generators"],
            metrics["n_parse_errors"],
        )
    logger.info("=" * 75)
    logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
