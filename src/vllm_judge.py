#!/usr/bin/env python3
"""Shared vLLM-based judge for local model inference.

Provides a VLLMJudge class that can be used by eval_pairwise.py,
eval_pointwise.py, and eval_biggen_pointwise.py as an alternative
to the Bedrock API client.
"""

import json
import logging
from pathlib import Path

from tqdm import tqdm

logger = logging.getLogger(__name__)


class VLLMJudge:
    """Local LLM judge powered by vLLM for fast batched inference."""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
    ):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_model_len = max_model_len

        logger.info(
            "Loading vLLM judge: %s (tp=%d, max_model_len=%d)",
            model_path, tensor_parallel_size, max_model_len,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._is_qwen3 = "qwen3" in model_path.lower()

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
        logger.info("vLLM judge loaded (tp=%d, qwen3=%s)", tensor_parallel_size, self._is_qwen3)

    def _build_prompt(self, system: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._is_qwen3:
            kwargs["enable_thinking"] = False
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(self, system: str, user_message: str) -> str:
        input_text = self._build_prompt(system, user_message)
        outputs = self.llm.generate([input_text], self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(self, prompts: list[tuple[str, str]], batch_size: int = 32) -> list[str | None]:
        """Generate responses for a batch of (system, user_message) pairs.

        Returns a list of strings (or None for prompts that exceed max_model_len).
        Processes in sub-batches of `batch_size` for memory efficiency.
        """
        input_texts = [self._build_prompt(sys, usr) for sys, usr in prompts]
        max_input_len = self.max_model_len - self.max_new_tokens

        valid_indices = []
        valid_texts = []
        for i, text in enumerate(input_texts):
            token_len = len(self.tokenizer.encode(text))
            if token_len <= max_input_len:
                valid_indices.append(i)
                valid_texts.append(text)
            else:
                logger.warning(
                    "Skipping prompt %d: %d tokens exceeds limit (%d)",
                    i, token_len, max_input_len,
                )

        results = [None] * len(prompts)
        if not valid_texts:
            return results

        for batch_start in range(0, len(valid_texts), batch_size):
            batch = valid_texts[batch_start:batch_start + batch_size]
            batch_indices = valid_indices[batch_start:batch_start + batch_size]
            outputs = self.llm.generate(batch, self.sampling_params)
            for idx, out in zip(batch_indices, outputs):
                results[idx] = out.outputs[0].text

        return results

    def generate_rubrics(
        self,
        system_prompt: str,
        user_template: str,
        keys: list[str],
        output_dir: Path,
        judge_name: str,
        *,
        batch_size: int = 32,
        template_key: str = "instruction",
    ) -> dict[str, str]:
        """Generate per-instance rubrics in batches with checkpoint support.

        Args:
            system_prompt: System prompt for rubric generation.
            user_template: User message template with a single format key.
            keys: List of instruction/prompt strings to generate rubrics for.
            output_dir: Directory to save rubric checkpoint file.
            judge_name: Name for checkpoint file.
            batch_size: Batch size for generation.
            template_key: The format key in user_template (e.g. 'instruction' or 'prompt').

        Returns:
            Dict mapping instruction/prompt -> generated rubric text.
        """
        from vllm import SamplingParams

        rubric_file = output_dir / f"rubrics_{judge_name}.json"

        rubrics: dict[str, str] = {}
        if rubric_file.exists():
            with open(rubric_file) as f:
                rubrics = json.load(f)
            logger.info("Loaded %d cached rubrics from %s", len(rubrics), rubric_file)

        remaining = [k for k in keys if k not in rubrics]
        if not remaining:
            logger.info("All %d rubrics already generated", len(keys))
            return rubrics

        logger.info("Generating rubrics for %d keys (batch_size=%d)", len(remaining), batch_size)

        rubric_sampling = SamplingParams(
            max_tokens=512,
            temperature=0,
        )

        for batch_start in tqdm(range(0, len(remaining), batch_size), desc="rubric gen"):
            batch = remaining[batch_start:batch_start + batch_size]
            input_texts = []
            for key in batch:
                user_msg = user_template.format(**{template_key: key})
                input_texts.append(self._build_prompt(system_prompt, user_msg))

            outputs = self.llm.generate(input_texts, rubric_sampling)
            for key, out in zip(batch, outputs):
                rubrics[key] = out.outputs[0].text

            if (batch_start + batch_size) % 100 < batch_size:
                with open(rubric_file, "w") as f:
                    json.dump(rubrics, f, indent=2, ensure_ascii=False)

        with open(rubric_file, "w") as f:
            json.dump(rubrics, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d rubrics -> %s", len(rubrics), rubric_file)
        return rubrics


def add_vllm_args(parser):
    """Add vLLM-related arguments to an argparse parser."""
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
