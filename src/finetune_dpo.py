#!/usr/bin/env python3
"""
Fine-tune a model as a rubric generator using DPO.

Supports Llama-3.1 and Qwen3 model families.

Trains on (prompt, chosen_rubric, rejected_rubric) pairs collected by
collect_reward_signals.py.

Requirements:
    pip install torch transformers trl peft bitsandbytes datasets accelerate

Usage:
    # Basic DPO training with Llama (base model -> v1)
    python src/finetune_dpo.py \
        --train-file outputs/reward_signals/healthbench/meta-judge/dpo_pairs.jsonl \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --output-dir models/rubric-gen-v1

    # DPO training with Qwen3-14B
    python src/finetune_dpo.py \
        --train-file outputs/reward_signals/biggen_pointwise/meta-judge_claude-sonnet-4/dpo_pairs.jsonl \
        --model-name Qwen/Qwen3-14B \
        --output-dir models/rubric-gen-qwen3-14b-v1

    # Continue DPO from existing LoRA adapter (v1 -> v2)
    python src/finetune_dpo.py \
        --train-file outputs/reward_signals/llama_3.1_8b/alpacaeval/meta-judge_claude-sonnet-4/dpo_pairs.jsonl \
        --adapter-path models/rubric-gen-v1 \
        --output-dir models/rubric-gen-v2

    # With multiple data sources
    python src/finetune_dpo.py \
        --train-file outputs/reward_signals/healthbench/combined/dpo_pairs.jsonl \
                     outputs/reward_signals/profbench/combined/dpo_pairs.jsonl \
                     outputs/reward_signals/helpsteer2/combined/dpo_pairs.jsonl \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --output-dir models/rubric-gen-v1

    # Quick test
    python src/finetune_dpo.py \
        --train-file outputs/reward_signals/healthbench/meta-judge/dpo_pairs.jsonl \
        --model-name Qwen/Qwen3-14B \
        --output-dir models/test \
        --max-steps 50
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dpo_data(train_files):
    """Load DPO pairs from one or more JSONL files."""
    all_data = []
    for fpath in train_files:
        path = Path(fpath)
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue
        with open(path) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_data.append({
                        "prompt": item["prompt"],
                        "chosen": item["chosen"],
                        "rejected": item["rejected"],
                    })
        logger.info("Loaded %d pairs from %s", len(all_data), path)

    logger.info("Total DPO pairs: %d", len(all_data))
    return all_data


QWEN3_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

LLAMA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _is_qwen3_model(model_name: str) -> bool:
    return "qwen3" in model_name.lower() or "qwen-3" in model_name.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune rubric generator with DPO",
    )
    parser.add_argument(
        "--train-file", nargs="+", required=True,
        help="Path(s) to DPO JSONL files from collect_reward_signals.py",
    )
    parser.add_argument(
        "--model-name", default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name or path (e.g., meta-llama/Llama-3.1-8B-Instruct, "
             "Qwen/Qwen3-14B)",
    )
    parser.add_argument(
        "--adapter-path", default=None,
        help="Path to an existing LoRA adapter (e.g., v1 model) to continue "
             "training from. The adapter is merged into the base model before "
             "applying a new LoRA for the next round of DPO.",
    )
    parser.add_argument(
        "--output-dir", default="models/rubric-gen-v1",
        help="Output directory for the fine-tuned model",
    )

    # LoRA config
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument(
        "--lora-target-modules", nargs="+", default=None,
        help="LoRA target modules (auto-detected from model family if not set)",
    )

    # DPO config
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Override num_epochs with max steps (-1 = use epochs)")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    # Quantization
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization (needs more VRAM)")
    parser.add_argument("--bf16", action="store_true", default=True)

    # Misc
    parser.add_argument("--eval-split", type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # --- Lazy imports (these are heavy) ---
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import DPOConfig, DPOTrainer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    raw_data = load_dpo_data(args.train_file)
    if not raw_data:
        logger.error("No training data found. Run collect_reward_signals.py first.")
        return

    dataset = Dataset.from_list(raw_data)

    # Train/eval split
    if args.eval_split > 0 and len(dataset) > 20:
        split = dataset.train_test_split(
            test_size=args.eval_split, seed=args.seed,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(
            "Split: %d train, %d eval", len(train_dataset), len(eval_dataset),
        )
    else:
        train_dataset = dataset
        eval_dataset = None

    # 2. Load model + tokenizer
    if args.adapter_path:
        adapter_config_path = Path(args.adapter_path) / "adapter_config.json"
        if adapter_config_path.exists():
            with open(adapter_config_path) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg.get(
                "base_model_name_or_path", args.model_name,
            )
        else:
            base_model_name = args.model_name
        logger.info(
            "Continuing from adapter: %s (base: %s)",
            args.adapter_path, base_model_name,
        )
    else:
        base_model_name = args.model_name
        logger.info("Loading base model: %s", base_model_name)

    is_qwen3 = _is_qwen3_model(base_model_name)
    if is_qwen3:
        logger.info("Detected Qwen3 model family")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {}
    if not args.no_quantize:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    attn_impl = "flash_attention_2"
    if is_qwen3:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
        **model_kwargs,
    )

    if args.adapter_path:
        logger.info("Loading and merging adapter from %s", args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        logger.info("Adapter merged into base model")

    # 3. LoRA config
    if args.lora_target_modules:
        target_modules = args.lora_target_modules
    elif is_qwen3:
        target_modules = QWEN3_TARGET_MODULES
    else:
        target_modules = LLAMA_TARGET_MODULES

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4. DPO training config
    training_args = DPOConfig(
        output_dir=str(output_dir),
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # 5. Train
    logger.info("Starting DPO training...")
    logger.info("  Train samples: %d", len(train_dataset))
    logger.info("  Eval samples: %d", len(eval_dataset) if eval_dataset else 0)
    if args.adapter_path:
        logger.info("  Starting from adapter: %s", args.adapter_path)
    logger.info("  LoRA rank: %d, alpha: %d", args.lora_rank, args.lora_alpha)
    logger.info("  DPO beta: %.2f, LR: %.1e", args.beta, args.learning_rate)
    logger.info(
        "  Effective batch: %d",
        args.per_device_batch_size * args.gradient_accumulation_steps,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    # 6. Save
    logger.info("Saving model to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training args for reproducibility
    with open(output_dir / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info("Done. Model saved to %s", output_dir)
    logger.info(
        "To merge LoRA weights: "
        "model = PeftModel.from_pretrained(base_model, '%s')", output_dir,
    )


if __name__ == "__main__":
    main()
