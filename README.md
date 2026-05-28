# Generating and Refining Dynamic Evaluation Rubrics for LLM-as-a-Judge

Code and data for the paper "Generating and Refining Dynamic Evaluation Rubrics for LLM-as-a-Judge" (ACL 2025).

## Setup

```bash
pip install -r requirements.txt
```

For local model inference, also install:
```bash
pip install torch transformers peft bitsandbytes accelerate datasets vllm trl
```

Set environment variables for Bedrock API access:
```bash
export AWS_BEARER_TOKEN_BEDROCK="your-token"
export AWS_REGION="us-east-1"
```

## Repository Structure

```
src/
  client.py                    # Shared Bedrock API client
  vllm_judge.py                # Shared vLLM inference engine
  eval_pairwise.py             # Pairwise evaluation (AlpacaEval, MTBench, BiGGen)
  eval_pointwise.py            # Pointwise evaluation (HelpSteer2, ProfBench, HealthBench)
  collect_reward_signals.py    # Reward signal collection for DPO training
  finetune_dpo.py              # DPO fine-tuning of rubric generator
  significance_test.py         # Statistical significance tests
  eval_checkeval.py            # Baseline: CheckEval
  eval_dna.py                  # Baseline: DnA-Eval
  eval_rubrichub.py            # Baseline: RubricHub
  eval_prometheus2.py          # Baseline: Prometheus 2
data/                          # Benchmark source data
outputs/
  reward_signals/              # Rubric candidates, meta-judge results, DPO pairs
  rubrics/                     # Generated rubrics for evaluation
```

## Usage

### Pairwise Evaluation

```bash
# Training-free dynamic rubrics (G=J)
python src/eval_pairwise.py --benchmark alpacaeval --rubric-mode dynamic --judges claude-sonnet-4

# With a fine-tuned rubric generator (local)
python src/eval_pairwise.py --benchmark mtbench \
    --rubric-backend vllm --rubric-model Qwen/Qwen3-14B \
    --judge-backend bedrock --judges claude-sonnet-4
```

### Pointwise Evaluation

```bash
python src/eval_pointwise.py --benchmark helpsteer2 --rubric-mode dynamic --judges claude-sonnet-4
```

### Reward Signal Collection

```bash
python src/collect_reward_signals.py \
    --benchmark alpacaeval \
    --reward-mode meta-judge \
    --rubric-backend api \
    --generator-model llama-3.1-8b \
    --meta-judge-model claude-sonnet-4
```

### DPO Fine-tuning

```bash
python src/finetune_dpo.py \
    --train-file outputs/reward_signals/Qwen3-14B/alpacaeval/meta-judge_claude-sonnet-4/dpo_pairs.jsonl \
    --model-name Qwen/Qwen3-14B \
    --output-dir models/rubric-gen-v1
```

### Baselines

```bash
python src/eval_checkeval.py --benchmark alpacaeval --judges claude-sonnet-4
python src/eval_dna.py --benchmark helpsteer2 --judges llama-3.1-70b
python src/eval_rubrichub.py --benchmark mtbench --judges claude-sonnet-4
python src/eval_prometheus2.py --benchmark alpacaeval --model prometheus-eval/prometheus-7b-v2.0
```

## Benchmarks

| Benchmark | Type | Source |
|-----------|------|--------|
| AlpacaEval | Pairwise | tatsu-lab/alpaca_eval |
| MT-Bench | Pairwise | lmsys/mt_bench_human_judgments |
| BiGGen Bench | Pairwise / Pointwise | prometheus-eval/BiGGen-Bench-Results |
| HelpSteer2 | Pointwise | nvidia/HelpSteer2 |
| ProfBench | Pointwise | nvidia/ProfBench |
| HealthBench | Pointwise | openai/healthbench |

## Judge Models

- Claude Sonnet 4 (via AWS Bedrock)
- Llama 3.1 8B / 70B Instruct (via AWS Bedrock or vLLM)
- Qwen3 14B (via vLLM)
