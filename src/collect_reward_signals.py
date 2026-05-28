#!/usr/bin/env python3
"""
Collect reward signals for DPO fine-tuning of rubric generators.

Uses eval-type-aware prompts (pairwise vs pointwise) that match inference
so the DPO training distribution aligns with test-time usage.

Two reward strategies:
  1. meta-judge:     A strong LLM compares rubric pairs and picks the better one.
  2. outcome-based:  Rubrics are used to judge responses; the rubric yielding
                     higher correlation with human labels is preferred.
  3. combined:       Both meta-judge and outcome-based scoring.

Rubric candidate generation backends:
  - api:            Generate candidates via Bedrock API (temperature sampling)
  - transformers:   Generate candidates locally via transformers + LoRA
  - vllm:           Generate candidates locally via vLLM

Pipeline:
  For each prompt in the benchmark data:
    a) Generate K rubric candidates (temperature sampling)
    b) Score each rubric (meta-judge pairwise or outcome-based correlation)
    c) Select chosen / rejected rubric pairs for DPO

Output: JSONL file with {prompt, chosen, rejected} suitable for trl DPOTrainer.

Usage:
    # Meta-judge reward on AlpacaEval (API rubric generation)
    python src/collect_reward_signals.py \\
        --benchmark alpacaeval \\
        --reward-mode meta-judge \\
        --rubric-backend api \\
        --generator-model llama-3.1-8b \\
        --meta-judge-model claude-sonnet-4 \\
        --num-candidates 8

    # Local rubric generation with DPO-finetuned model
    python src/collect_reward_signals.py \\
        --benchmark helpsteer2 \\
        --reward-mode meta-judge \\
        --rubric-backend transformers \\
        --rubric-model models/rubric-gen-v1 \\
        --meta-judge-model claude-sonnet-4 \\
        --num-candidates 8

    # vLLM rubric generation
    python src/collect_reward_signals.py \\
        --benchmark alpacaeval \\
        --reward-mode combined \\
        --rubric-backend vllm \\
        --rubric-model Qwen/Qwen3-14B \\
        --meta-judge-model claude-sonnet-4

    # Outcome-based reward on HealthBench
    python src/collect_reward_signals.py \\
        --benchmark healthbench \\
        --reward-mode outcome \\
        --rubric-backend api \\
        --generator-model llama-3.1-8b \\
        --judge-model claude-sonnet-4
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import requests
from scipy.stats import spearmanr
from tqdm import tqdm

from client import BedrockClient, call_judge, JUDGES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry (for API-based rubric generation)
# ---------------------------------------------------------------------------
MODELS = {
    "claude-sonnet-4": {
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "llama-3.1-8b": {
        "model_id": "us.meta.llama3-1-8b-instruct-v1:0",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "llama-3.1-70b": {
        "model_id": "us.meta.llama3-1-70b-instruct-v1:0",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
}

# ---------------------------------------------------------------------------
# Rubric generation prompts (aligned with eval_pairwise/eval_pointwise)
# ---------------------------------------------------------------------------
PAIRWISE_RUBRIC_SYSTEM = """\
You are an expert evaluator designing assessment criteria. Given a task \
instruction, create a concise evaluation rubric with 3-5 specific criteria \
for judging the quality of responses to this instruction.

Focus on what makes a response good or bad FOR THIS SPECIFIC TASK. Consider \
the task type (factual Q&A, creative writing, coding, reasoning, summarization, \
etc.) and what a high-quality answer requires.

Output ONLY a numbered list of criteria. Each criterion should be one clear sentence."""

PAIRWISE_RUBRIC_USER = """\
[Instruction]
{instruction}

Write the evaluation rubric for judging responses to the above instruction."""

POINTWISE_RUBRIC_SYSTEM = """\
You are an expert evaluator designing assessment criteria. Given a task \
prompt, create a concise evaluation rubric with 3-5 specific criteria \
for scoring the quality of responses to this prompt on a 1-5 scale.

Focus on what makes a response good or bad FOR THIS SPECIFIC TASK. Consider \
the task type (factual Q&A, creative writing, coding, reasoning, summarization, \
etc.) and what a high-quality answer requires.

Output ONLY a numbered list of criteria. Each criterion should be one clear \
sentence."""

POINTWISE_RUBRIC_USER = """\
[Prompt]
{prompt}

Write the evaluation rubric for scoring responses to the above prompt."""

# ---------------------------------------------------------------------------
# Meta-judge prompts
# ---------------------------------------------------------------------------
META_JUDGE_SYSTEM = """\
You are a meta-evaluator that assesses the quality of evaluation rubrics. \
You will be given a task prompt and two candidate rubrics (Rubric A and \
Rubric B). Determine which rubric would be more effective for evaluating \
AI responses to the given prompt.

Consider these dimensions:
1. Specificity: Are criteria concrete and testable, or vague and subjective?
2. Coverage: Do criteria address the key aspects an expert would check?
3. Discriminability: Can these criteria distinguish genuinely good responses \
from superficially good ones?
4. Domain-appropriateness: Do criteria reflect the expertise level the \
task requires?

Output ONLY a JSON object:
{"winner": "A" or "B", "reason": "brief explanation"}"""

META_JUDGE_USER = """\
[Task Prompt]
{prompt}

[Rubric A]
{rubric_a}

[Rubric B]
{rubric_b}

Which rubric is better for evaluating responses to this task? Output only \
the JSON object."""

# ---------------------------------------------------------------------------
# Outcome-based judge prompts
# ---------------------------------------------------------------------------
OUTCOME_JUDGE_SYSTEM = """\
You are a helpful assistant that evaluates AI responses against specific \
criteria. For each criterion provided, determine whether the response \
satisfies it.

You must output ONLY a JSON array. For each criterion, include:
- "criterion": the criterion text (copied exactly)
- "fulfilled": true if the response clearly satisfies this criterion, \
false otherwise"""

OUTCOME_JUDGE_USER = """\
[Prompt]
{prompt}

[Response]
{response}

[Evaluation Criteria]
{criteria_text}

For each criterion above, determine whether the response fulfills it. \
Output only the JSON array."""


# ---------------------------------------------------------------------------
# API helper (wraps call_judge with different defaults)
# ---------------------------------------------------------------------------
def call_model(client, model_id, *, system, user_message,
               max_tokens=1024, temperature=0.0, max_retries=5):
    return call_judge(
        client, model_id,
        system=system, user_message=user_message,
        max_tokens=max_tokens, temperature=temperature,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Local rubric generator (transformers + LoRA)
# ---------------------------------------------------------------------------
class LocalRubricGenerator:
    """Load a local model (with optional LoRA adapter) for rubric generation."""

    def __init__(
        self,
        model_path: str,
        base_model: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
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
        logger.info("Local generator loaded on device: %s", self.model.device)

    def _apply_chat_template(self, messages: list[dict]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._is_qwen3:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate_multiple(self, system: str, user_message: str, n: int) -> list[str]:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        input_text = self._apply_chat_template(messages)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(
            self.model.device
        )

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": True,
            "temperature": self.temperature,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": n,
        }

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


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
def parse_json_array(raw):
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return []


def parse_json_object(raw):
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_benchmark_prompts(benchmark, cache_dir):
    """Load prompts (and optionally responses + human scores) for a benchmark."""
    from eval_pointwise import load_healthbench, load_helpsteer2
    from eval_checkeval import load_alpacaeval, load_mtbench, load_biggen

    pointwise_loaders = {
        "helpsteer2": lambda: load_helpsteer2(cache_dir),
        "healthbench": lambda: load_healthbench(cache_dir),
    }

    pairwise_loaders = {
        "alpacaeval": lambda: load_alpacaeval(cache_dir),
        "mtbench": lambda: load_mtbench(cache_dir),
        "biggen": lambda: load_biggen(cache_dir),
    }

    if benchmark in pointwise_loaders:
        items = pointwise_loaders[benchmark]()
        prompt_groups = {}
        for item in items:
            p = item["prompt"]
            if p not in prompt_groups:
                prompt_groups[p] = []
            prompt_groups[p].append(item)
        return prompt_groups, "pointwise"
    elif benchmark in pairwise_loaders:
        items = pairwise_loaders[benchmark]()
        prompt_groups = {}
        for item in items:
            p = item.get("instruction", item.get("prompt", ""))
            if p not in prompt_groups:
                prompt_groups[p] = []
            prompt_groups[p].append(item)
        return prompt_groups, "pairwise"
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


# ---------------------------------------------------------------------------
# Prompt selection helper
# ---------------------------------------------------------------------------
def _get_rubric_prompts(eval_type: str):
    """Return (system_prompt, user_template, format_key) based on eval type."""
    if eval_type == "pairwise":
        return PAIRWISE_RUBRIC_SYSTEM, PAIRWISE_RUBRIC_USER, "instruction"
    else:
        return POINTWISE_RUBRIC_SYSTEM, POINTWISE_RUBRIC_USER, "prompt"


# ---------------------------------------------------------------------------
# Step 1: Generate rubric candidates
# ---------------------------------------------------------------------------
def generate_rubric_candidates_api(
    client, model_id, prompts, eval_type, num_candidates, output_dir,
    max_workers=16,
):
    """Generate K rubric candidates per prompt via API with temperature sampling."""
    cache_file = output_dir / "rubric_candidates.json"
    all_candidates = {}
    if cache_file.exists():
        with open(cache_file) as f:
            all_candidates = json.load(f)
        logger.info("Loaded %d cached prompts from %s", len(all_candidates), cache_file)

    remaining = [p for p in prompts if p not in all_candidates]
    if not remaining:
        logger.info("All %d prompts already have candidates", len(prompts))
        return all_candidates

    logger.info(
        "Generating %d candidates each for %d prompts (workers=%d)",
        num_candidates, len(remaining), max_workers,
    )

    system_prompt, user_template, key = _get_rubric_prompts(eval_type)

    def _generate_one(prompt, candidate_idx):
        user_msg = user_template.format(**{key: prompt})
        raw = call_model(
            client, model_id,
            system=system_prompt,
            user_message=user_msg,
            max_tokens=1024,
            temperature=0.8,
        )
        return prompt, candidate_idx, raw

    pending = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for prompt in remaining:
            pending[prompt] = [None] * num_candidates
            for k in range(num_candidates):
                future = executor.submit(_generate_one, prompt, k)
                future._prompt = prompt
                future._idx = k
                pending[prompt][k] = future

        completed_prompts = 0
        pbar = tqdm(total=len(remaining), desc="Generating rubrics")
        done_prompts = set()

        for future in as_completed(
            [f for futures in pending.values() for f in futures]
        ):
            prompt = future._prompt
            if prompt in done_prompts:
                continue
            if all(f.done() for f in pending[prompt]):
                candidates = []
                for f in pending[prompt]:
                    _, _, raw = f.result()
                    if raw.strip():
                        candidates.append({"rubric": raw})
                all_candidates[prompt] = candidates
                done_prompts.add(prompt)
                completed_prompts += 1
                pbar.update(1)

                if completed_prompts % 50 == 0:
                    with open(cache_file, "w") as f:
                        json.dump(all_candidates, f, indent=2, ensure_ascii=False)

        pbar.close()

    with open(cache_file, "w") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)
    logger.info("Saved candidates for %d prompts -> %s", len(all_candidates), cache_file)
    return all_candidates


def generate_rubric_candidates_local(
    generator, prompts, eval_type, num_candidates, output_dir,
):
    """Generate K rubric candidates per prompt using a local model."""
    cache_file = output_dir / "rubric_candidates.json"
    all_candidates = {}
    if cache_file.exists():
        with open(cache_file) as f:
            all_candidates = json.load(f)
        logger.info("Loaded %d cached prompts from %s", len(all_candidates), cache_file)

    remaining = [p for p in prompts if p not in all_candidates]
    if not remaining:
        logger.info("All %d prompts already have candidates", len(prompts))
        return all_candidates

    logger.info(
        "Generating %d candidates each for %d prompts (local)",
        num_candidates, len(remaining),
    )

    system_prompt, user_template, key = _get_rubric_prompts(eval_type)

    for prompt in tqdm(remaining, desc="Generating rubrics (local)"):
        user_msg = user_template.format(**{key: prompt})
        completions = generator.generate_multiple(system_prompt, user_msg, num_candidates)
        candidates = [{"rubric": c} for c in completions if c.strip()]
        all_candidates[prompt] = candidates

        if len(all_candidates) % 50 == 0:
            with open(cache_file, "w") as f:
                json.dump(all_candidates, f, indent=2, ensure_ascii=False)

    with open(cache_file, "w") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)
    logger.info("Saved candidates for %d prompts -> %s", len(all_candidates), cache_file)
    return all_candidates


def generate_rubric_candidates_vllm(
    vllm_judge, prompts, eval_type, num_candidates, output_dir,
    batch_size=32,
):
    """Generate K rubric candidates per prompt using vLLM."""
    from vllm import SamplingParams

    cache_file = output_dir / "rubric_candidates.json"
    all_candidates = {}
    if cache_file.exists():
        with open(cache_file) as f:
            all_candidates = json.load(f)
        logger.info("Loaded %d cached prompts from %s", len(all_candidates), cache_file)

    remaining = [p for p in prompts if p not in all_candidates]
    if not remaining:
        logger.info("All %d prompts already have candidates", len(prompts))
        return all_candidates

    logger.info(
        "Generating %d candidates each for %d prompts (vLLM, batch=%d)",
        num_candidates, len(remaining), batch_size,
    )

    system_prompt, user_template, key = _get_rubric_prompts(eval_type)
    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.8,
        n=num_candidates,
    )

    for batch_start in tqdm(
        range(0, len(remaining), batch_size), desc="Generating rubrics (vLLM)"
    ):
        batch = remaining[batch_start:batch_start + batch_size]
        input_texts = []
        for prompt in batch:
            user_msg = user_template.format(**{key: prompt})
            input_texts.append(vllm_judge._build_prompt(system_prompt, user_msg))

        outputs = vllm_judge.llm.generate(input_texts, sampling_params)
        for prompt, out in zip(batch, outputs):
            candidates = []
            for completion in out.outputs:
                text = completion.text.strip()
                if text:
                    candidates.append({"rubric": text})
            all_candidates[prompt] = candidates

        if (batch_start + batch_size) % 100 < batch_size:
            with open(cache_file, "w") as f:
                json.dump(all_candidates, f, indent=2, ensure_ascii=False)

    with open(cache_file, "w") as f:
        json.dump(all_candidates, f, indent=2, ensure_ascii=False)
    logger.info("Saved candidates for %d prompts -> %s", len(all_candidates), cache_file)
    return all_candidates


# ---------------------------------------------------------------------------
# Step 2a: Meta-judge scoring
# ---------------------------------------------------------------------------
def run_meta_judge(
    client, meta_judge_model_id, prompts, all_candidates, output_dir,
    max_pairs_per_prompt=10, max_workers=16,
):
    """Run meta-judge pairwise comparisons between rubric candidates."""
    cache_file = output_dir / "meta_judge_results.json"
    results = {}
    if cache_file.exists():
        with open(cache_file) as f:
            results = json.load(f)
        logger.info("Loaded %d cached meta-judge results", len(results))

    remaining = [p for p in prompts if p not in results]
    if not remaining:
        return results

    logger.info("Running meta-judge on %d prompts (workers=%d)", len(remaining), max_workers)

    def _judge_pair(prompt, i, j, rubric_a, rubric_b):
        user_msg = META_JUDGE_USER.format(
            prompt=prompt, rubric_a=rubric_a, rubric_b=rubric_b,
        )
        raw_fwd = call_model(
            client, meta_judge_model_id,
            system=META_JUDGE_SYSTEM,
            user_message=user_msg,
            max_tokens=256,
            temperature=0.0,
        )
        result_fwd = parse_json_object(raw_fwd)

        user_msg_rev = META_JUDGE_USER.format(
            prompt=prompt, rubric_a=rubric_b, rubric_b=rubric_a,
        )
        raw_rev = call_model(
            client, meta_judge_model_id,
            system=META_JUDGE_SYSTEM,
            user_message=user_msg_rev,
            max_tokens=256,
            temperature=0.0,
        )
        result_rev = parse_json_object(raw_rev)

        fwd_winner = result_fwd.get("winner", "")
        rev_winner = result_rev.get("winner", "")

        w_ij = 0.0
        w_ji = 0.0
        if fwd_winner == "A":
            w_ij += 0.5
        elif fwd_winner == "B":
            w_ji += 0.5
        if rev_winner == "A":
            w_ji += 0.5
        elif rev_winner == "B":
            w_ij += 0.5

        return prompt, i, j, w_ij, w_ji

    tasks = []
    for prompt in remaining:
        candidates = all_candidates.get(prompt, [])
        if len(candidates) < 2:
            results[prompt] = {
                "wins": [0] * len(candidates),
                "pair_outcomes": [],
                "n_candidates": len(candidates),
            }
            continue

        all_pairs = list(combinations(range(len(candidates)), 2))
        if len(all_pairs) > max_pairs_per_prompt:
            rng = np.random.default_rng(hash(prompt) % (2**32))
            indices = rng.choice(
                len(all_pairs), size=max_pairs_per_prompt, replace=False,
            )
            pairs = [all_pairs[idx] for idx in indices]
        else:
            pairs = all_pairs

        for i, j in pairs:
            rubric_a = candidates[i]["rubric"]
            rubric_b = candidates[j]["rubric"]
            tasks.append((prompt, i, j, rubric_a, rubric_b))

    prompt_pairs = {}
    prompt_expected = {}
    for prompt in remaining:
        candidates = all_candidates.get(prompt, [])
        if len(candidates) >= 2:
            prompt_pairs[prompt] = []

    for p, i, j, ra, rb in tasks:
        prompt_expected[p] = prompt_expected.get(p, 0) + 1

    completed_prompts = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_judge_pair, p, i, j, ra, rb)
            for p, i, j, ra, rb in tasks
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Meta-judging"):
            prompt, i, j, w_ij, w_ji = future.result()
            prompt_pairs[prompt].append((i, j, w_ij, w_ji))

            if len(prompt_pairs[prompt]) == prompt_expected[prompt]:
                candidates = all_candidates.get(prompt, [])
                wins = [0.0] * len(candidates)
                pair_outcomes = []
                for pi, pj, pw_ij, pw_ji in prompt_pairs[prompt]:
                    wins[pi] += pw_ij
                    wins[pj] += pw_ji
                    pair_outcomes.append([pi, pj, pw_ij, pw_ji])

                results[prompt] = {
                    "wins": wins,
                    "pair_outcomes": pair_outcomes,
                    "n_candidates": len(candidates),
                }
                completed_prompts += 1

                if completed_prompts % 20 == 0:
                    with open(cache_file, "w") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


# ---------------------------------------------------------------------------
# Step 2b: Outcome-based scoring
# ---------------------------------------------------------------------------
def compute_criterion_score(criteria_text, evaluation, benchmark):
    fulfilled = []
    for item in evaluation:
        f = item.get("fulfilled")
        if f is None:
            continue
        fulfilled.append(bool(f))
    if not fulfilled:
        return None
    return sum(fulfilled) / len(fulfilled)


def run_outcome_scoring(
    client, judge_model_id, prompts, all_candidates,
    prompt_groups, benchmark, output_dir, max_workers=16,
):
    """Score rubrics by how well they predict human labels."""
    cache_file = output_dir / "outcome_scores.json"
    scores = {}
    if cache_file.exists():
        with open(cache_file) as f:
            scores = json.load(f)
        logger.info("Loaded %d cached outcome scores", len(scores))

    remaining = [p for p in prompts if p not in scores]
    if not remaining:
        return scores

    logger.info("Running outcome-based scoring on %d prompts (workers=%d)", len(remaining), max_workers)

    def _eval_one(prompt, response, criteria_text):
        user_msg = OUTCOME_JUDGE_USER.format(
            prompt=prompt, response=response, criteria_text=criteria_text,
        )
        raw = call_model(
            client, judge_model_id,
            system=OUTCOME_JUDGE_SYSTEM,
            user_message=user_msg,
            max_tokens=1024,
            temperature=0.0,
        )
        return raw

    for prompt in tqdm(remaining, desc="Outcome scoring"):
        candidates = all_candidates.get(prompt, [])
        items = prompt_groups.get(prompt, [])

        if not candidates or not items:
            scores[prompt] = {"correlations": [0.0] * len(candidates)}
            continue

        eval_tasks = []
        for cand_idx, cand in enumerate(candidates):
            criteria_text = cand["rubric"]
            if not criteria_text.strip():
                continue
            for item_idx, item in enumerate(items):
                response = item.get("response", "")
                human_score = item.get("human_score")
                if human_score is None or not response:
                    continue
                eval_tasks.append((cand_idx, item_idx, response, criteria_text, human_score))

        eval_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {}
            for cand_idx, item_idx, response, criteria_text, human_score in eval_tasks:
                future = executor.submit(_eval_one, prompt, response, criteria_text)
                future_to_key[future] = (cand_idx, item_idx, human_score)

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                cand_idx, item_idx, human_score = key
                eval_results[(cand_idx, item_idx)] = (future.result(), human_score)

        candidate_corrs = []
        for cand_idx, cand in enumerate(candidates):
            criteria_text = cand["rubric"]
            if not criteria_text.strip():
                candidate_corrs.append(0.0)
                continue

            judge_scores = []
            human_scores = []
            for (ci, ii), (raw, hs) in eval_results.items():
                if ci != cand_idx:
                    continue
                evaluation = parse_json_array(raw)
                score = compute_criterion_score(criteria_text, evaluation, benchmark)
                if score is not None:
                    judge_scores.append(score)
                    human_scores.append(hs)

            if len(judge_scores) >= 3:
                corr = spearmanr(judge_scores, human_scores).statistic
                candidate_corrs.append(corr if not np.isnan(corr) else 0.0)
            else:
                candidate_corrs.append(0.0)

        scores[prompt] = {"correlations": candidate_corrs}

        if len(scores) % 10 == 0:
            with open(cache_file, "w") as f:
                json.dump(scores, f, indent=2, ensure_ascii=False)

    with open(cache_file, "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    return scores


# ---------------------------------------------------------------------------
# Bradley-Terry model fitting
# ---------------------------------------------------------------------------
def bradley_terry_scores(n_candidates, pair_outcomes, max_iter=100, tol=1e-6):
    """Fit Bradley-Terry model via iterative MLE and return log-strengths."""
    if n_candidates < 2 or not pair_outcomes:
        return [0.0] * n_candidates

    gamma = np.ones(n_candidates, dtype=np.float64)

    W = np.zeros((n_candidates, n_candidates), dtype=np.float64)
    for i, j, w_ij, w_ji in pair_outcomes:
        W[int(i)][int(j)] += w_ij
        W[int(j)][int(i)] += w_ji

    N = W + W.T

    for _ in range(max_iter):
        gamma_old = gamma.copy()
        for i in range(n_candidates):
            numerator = 0.0
            denominator = 0.0
            for j in range(n_candidates):
                if i == j or N[i][j] == 0:
                    continue
                numerator += W[i][j]
                denominator += N[i][j] / (gamma[i] + gamma[j])
            if denominator > 0:
                gamma[i] = numerator / denominator

        gamma /= gamma.mean()

        if np.max(np.abs(gamma - gamma_old)) < tol:
            break

    gamma = np.clip(gamma, 1e-10, None)
    scores = (400.0 * np.log10(gamma)).tolist()
    return scores


# ---------------------------------------------------------------------------
# Step 3: Construct DPO pairs
# ---------------------------------------------------------------------------
def construct_dpo_pairs(
    prompts, all_candidates, eval_type, meta_results, outcome_scores,
    reward_mode,
):
    """Build (prompt, chosen, rejected) triples from reward signals."""
    dpo_data = []

    system_prompt, user_template, key = _get_rubric_prompts(eval_type)

    for prompt in prompts:
        candidates = all_candidates.get(prompt, [])
        if len(candidates) < 2:
            continue

        scores = [0.0] * len(candidates)

        if reward_mode in ("meta-judge", "combined") and meta_results:
            meta = meta_results.get(prompt, {})
            bt_scores = bradley_terry_scores(
                meta.get("n_candidates", len(candidates)),
                meta.get("pair_outcomes", []),
            )
            for k in range(len(candidates)):
                if k < len(bt_scores):
                    scores[k] += bt_scores[k]

        if reward_mode in ("outcome", "combined") and outcome_scores:
            outcome = outcome_scores.get(prompt, {})
            corrs = outcome.get("correlations", [0.0] * len(candidates))
            for k in range(len(candidates)):
                if k < len(corrs):
                    scores[k] += corrs[k] * 200.0

        best_idx = int(np.argmax(scores))
        worst_idx = int(np.argmin(scores))

        if best_idx == worst_idx:
            continue
        if scores[best_idx] <= scores[worst_idx]:
            continue

        gen_prompt = user_template.format(**{key: prompt})
        chosen = candidates[best_idx]["rubric"]
        rejected = candidates[worst_idx]["rubric"]

        dpo_data.append({
            "prompt": f"{system_prompt}\n\n{gen_prompt}",
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "task_prompt": prompt[:200],
                "chosen_score": scores[best_idx],
                "rejected_score": scores[worst_idx],
                "chosen_idx": best_idx,
                "rejected_idx": worst_idx,
            },
        })

    return dpo_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Collect reward signals for DPO fine-tuning",
    )
    parser.add_argument(
        "--benchmark", required=True,
        choices=["alpacaeval", "mtbench", "biggen", "helpsteer2", "healthbench"],
    )
    parser.add_argument(
        "--eval-type", default=None, choices=["pairwise", "pointwise"],
        help="Evaluation type (auto-detected from benchmark if not set)",
    )
    parser.add_argument(
        "--reward-mode", default="meta-judge",
        choices=["meta-judge", "outcome", "combined"],
        help="Reward signal strategy",
    )

    # Rubric generation backend
    parser.add_argument(
        "--rubric-backend", default="api",
        choices=["api", "transformers", "vllm"],
        help="Backend for rubric candidate generation",
    )
    parser.add_argument(
        "--generator-model", default="llama-3.1-8b",
        help="Model for generating rubric candidates (API model name or local path)",
    )
    parser.add_argument(
        "--base-model", default=None,
        help="Base model for LoRA adapters (auto-detected if not set)",
    )
    parser.add_argument(
        "--rubric-model", default=None,
        help="Alias for --generator-model (local model path)",
    )

    # Meta-judge / outcome judge
    parser.add_argument(
        "--meta-judge-model", default="claude-sonnet-4",
        choices=list(MODELS.keys()),
        help="Model for meta-judge comparisons",
    )
    parser.add_argument(
        "--judge-model", default="claude-sonnet-4",
        choices=list(MODELS.keys()),
        help="Model for outcome-based evaluation",
    )

    # Generation params
    parser.add_argument(
        "--num-candidates", type=int, default=8,
        help="Number of rubric candidates per prompt (default: 8)",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=10,
        help="Max meta-judge pairs per prompt (default: 10)",
    )
    parser.add_argument(
        "--max-prompts", type=int, default=None,
        help="Limit number of prompts (for testing)",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/reward_signals")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--region", default=None)

    # vLLM params
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)

    args = parser.parse_args()

    # Resolve rubric-model alias
    if args.rubric_model:
        args.generator_model = args.rubric_model

    # Auto-detect eval type
    pairwise_benchmarks = {"alpacaeval", "mtbench", "biggen"}
    if args.eval_type is None:
        args.eval_type = "pairwise" if args.benchmark in pairwise_benchmarks else "pointwise"

    # Output directory
    if args.reward_mode in ("meta-judge", "combined"):
        output_dir = (
            Path(args.output_dir) / args.benchmark
            / f"{args.reward_mode}_{args.meta_judge_model}"
        )
    elif args.reward_mode == "outcome":
        output_dir = (
            Path(args.output_dir) / args.benchmark
            / f"{args.reward_mode}_{args.judge_model}"
        )
    else:
        output_dir = Path(args.output_dir) / args.benchmark / args.reward_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    logger.info(
        "Benchmark: %s | Eval type: %s | Reward: %s | Backend: %s | Candidates: %d",
        args.benchmark, args.eval_type, args.reward_mode,
        args.rubric_backend, args.num_candidates,
    )

    # 1. Load data
    prompt_groups, eval_type = load_benchmark_prompts(args.benchmark, cache_dir)
    prompts = list(prompt_groups.keys())
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    logger.info("Loaded %d prompts (%s)", len(prompts), eval_type)

    # 2. Generate rubric candidates
    if args.rubric_backend == "api":
        client = BedrockClient(region=args.region)
        if args.generator_model in MODELS:
            gen_model_id = MODELS[args.generator_model]["model_id"]
        else:
            gen_model_id = args.generator_model
        all_candidates = generate_rubric_candidates_api(
            client, gen_model_id,
            prompts, eval_type, args.num_candidates, output_dir,
            max_workers=args.max_workers,
        )
    elif args.rubric_backend == "transformers":
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        generator = LocalRubricGenerator(
            model_path=args.generator_model,
            base_model=args.base_model,
            temperature=0.8,
        )
        all_candidates = generate_rubric_candidates_local(
            generator, prompts, eval_type, args.num_candidates, output_dir,
        )
        del generator
        import torch
        torch.cuda.empty_cache()
    elif args.rubric_backend == "vllm":
        from vllm_judge import VLLMJudge
        vllm_gen = VLLMJudge(
            model_path=args.generator_model,
            max_new_tokens=1024,
            temperature=0.8,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        all_candidates = generate_rubric_candidates_vllm(
            vllm_gen, prompts, eval_type, args.num_candidates, output_dir,
            batch_size=args.batch_size,
        )
        del vllm_gen

    # 3. Initialize Bedrock client for meta-judge / outcome scoring
    client = BedrockClient(region=args.region)

    # 4. Score rubrics
    meta_results = None
    outcome_scores = None

    if args.reward_mode in ("meta-judge", "combined"):
        mj_config = MODELS[args.meta_judge_model]
        meta_results = run_meta_judge(
            client, mj_config["model_id"],
            prompts, all_candidates, output_dir,
            max_pairs_per_prompt=args.max_pairs,
            max_workers=args.max_workers,
        )

    if args.reward_mode in ("outcome", "combined"):
        judge_config = MODELS[args.judge_model]
        outcome_scores = run_outcome_scoring(
            client, judge_config["model_id"],
            prompts, all_candidates, prompt_groups,
            args.benchmark, output_dir,
            max_workers=args.max_workers,
        )

    # 5. Construct DPO pairs
    dpo_data = construct_dpo_pairs(
        prompts, all_candidates, eval_type, meta_results, outcome_scores,
        args.reward_mode,
    )

    # 6. Save
    dpo_file = output_dir / "dpo_pairs.jsonl"
    with open(dpo_file, "w") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("=" * 70)
    logger.info("Done! %d DPO pairs saved to %s", len(dpo_data), dpo_file)
    logger.info("  Prompts processed: %d", len(prompts))
    logger.info("  Candidates generated: %d", sum(len(v) for v in all_candidates.values()))
    logger.info("  DPO pairs: %d (%.1f%% of prompts)",
                len(dpo_data), len(dpo_data) / len(prompts) * 100 if prompts else 0)
    logger.info("=" * 70)

    if dpo_data:
        chosen_scores = [d["metadata"]["chosen_score"] for d in dpo_data]
        rejected_scores = [d["metadata"]["rejected_score"] for d in dpo_data]
        logger.info(
            "Score gap: mean=%.3f, min=%.3f, max=%.3f",
            np.mean(np.array(chosen_scores) - np.array(rejected_scores)),
            np.min(np.array(chosen_scores) - np.array(rejected_scores)),
            np.max(np.array(chosen_scores) - np.array(rejected_scores)),
        )


if __name__ == "__main__":
    main()
