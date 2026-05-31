"""
09_sagemaker_train.py
---------------------
AWS SageMaker + Bedrock patterns for training, deploying, and calling HF models.

This file has TWO sections:
  A) The SageMaker TRAINING SCRIPT  — runs inside the SageMaker container
  B) The LAUNCH SCRIPT              — runs locally / in a notebook

NOTE: Requires AWS credentials + SageMaker/Bedrock access.
      No real training will run unless you have an AWS account configured.
      The script prints all code patterns so you can adapt them.

Run with: python scripts/09_sagemaker_train.py
"""

import sys

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║           AWS SAGEMAKER + BEDROCK PATTERNS FOR HF MODELS                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION A: SAGEMAKER TRAINING SCRIPT  (save as  train_script.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)

# SageMaker injects paths via environment variables
MODEL_DIR   = os.environ["SM_MODEL_DIR"]      # /opt/ml/model
TRAIN_DIR   = os.environ["SM_CHANNEL_TRAIN"]  # /opt/ml/input/data/train
# Optional extra channels:
# TEST_DIR  = os.environ.get("SM_CHANNEL_TEST", None)
# OUTPUT_DIR= os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

# Hyperparameters injected by the estimator
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--epochs",     type=int,   default=1)
parser.add_argument("--batch_size", type=int,   default=4)
parser.add_argument("--lr",         type=float, default=5e-5)
parser.add_argument("--model_id",   type=str,   default="distilgpt2")
args, _ = parser.parse_known_args()

# Load data uploaded to S3 (SageMaker downloads it to TRAIN_DIR)
dataset = load_from_disk(TRAIN_DIR)

tok = AutoTokenizer.from_pretrained(args.model_id)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model_id)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

train_args = TrainingArguments(
    output_dir=MODEL_DIR,                # SageMaker saves everything here
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    learning_rate=args.lr,
    save_strategy="epoch",
    report_to="none",
)
trainer = Trainer(
    model=model, args=train_args,
    train_dataset=dataset,
    data_collator=collator,
    processing_class=tok,
)
trainer.train()
model.save_pretrained(MODEL_DIR)
tok.save_pretrained(MODEL_DIR)
print("Training done. Model saved to", MODEL_DIR)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION B: LAUNCH SCRIPT  (run locally or in SageMaker Studio notebook)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import sagemaker
from sagemaker.huggingface import HuggingFace

sess   = sagemaker.Session()
role   = sagemaker.get_execution_role()
bucket = sess.default_bucket()

# ── Upload data to S3 ────────────────────────────────────────────────────────
from datasets import load_dataset
ds = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
ds.save_to_disk("/tmp/train_data")
train_s3 = sess.upload_data("/tmp/train_data", key_prefix="llm-ft/data/train")

# ── Define Estimator ─────────────────────────────────────────────────────────
estimator = HuggingFace(
    entry_point="train_script.py",          # your training script
    source_dir=".",                         # directory containing the script
    role=role,
    instance_count=1,
    instance_type="ml.g5.2xlarge",          # A10G GPU, ~$1.5/hr
    transformers_version="4.36",
    pytorch_version="2.1",
    py_version="py310",
    hyperparameters={
        "epochs": 3,
        "batch_size": 8,
        "lr": 2e-5,
        "model_id": "distilgpt2",
    },
    # Spot instance (up to 70% cheaper, can be interrupted):
    # use_spot_instances=True,
    # max_wait=7200,
    # max_run=3600,
)

estimator.fit({"train": train_s3})
print("Training job:", estimator.latest_training_job.name)

# ── Deploy: TGI endpoint for LLM ─────────────────────────────────────────────
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

llm_image = get_huggingface_llm_image_uri("huggingface", version="2.0")

env = {
    "HF_MODEL_ID": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "HF_TASK": "text-generation",
    "HUGGING_FACE_HUB_TOKEN": "<your-hf-token>",
    "SM_NUM_GPUS": "1",
    "MAX_INPUT_LENGTH": "2048",
    "MAX_TOTAL_TOKENS": "4096",
}
hf_model = HuggingFaceModel(image_uri=llm_image, env=env, role=role)
predictor = hf_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
)

# ── Invoke endpoint ───────────────────────────────────────────────────────────
response = predictor.predict({
    "inputs": "Explain LoRA in one sentence.",
    "parameters": {"max_new_tokens": 200, "temperature": 0.7},
})
print(response[0]["generated_text"])

# ── Cleanup ───────────────────────────────────────────────────────────────────
predictor.delete_endpoint()


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SECTION C: BEDROCK  — Converse API + Fine-tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

# ── Converse (chat) ───────────────────────────────────────────────────────────
response = bedrock.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": [{"text": "Explain GRPO."}]}],
    inferenceConfig={"maxTokens": 512, "temperature": 0.7},
)
print(response["output"]["message"]["content"][0]["text"])

# ── Converse stream ───────────────────────────────────────────────────────────
stream = bedrock.converse_stream(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    messages=[{"role": "user", "content": [{"text": "Explain LoRA step by step."}]}],
)
for event in stream["stream"]:
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"]["text"], end="", flush=True)

# ── Fine-tune on Bedrock ──────────────────────────────────────────────────────
bedrock_mgmt = boto3.client("bedrock", region_name="us-east-1")

bedrock_mgmt.create_model_customization_job(
    jobName="my-finetune-job",
    customModelName="my-custom-model",
    roleArn="arn:aws:iam::123456789:role/BedrockRole",
    baseModelIdentifier="amazon.titan-text-express-v1",
    trainingDataConfig={"s3Uri": "s3://my-bucket/training-data.jsonl"},
    outputDataConfig={"s3Uri": "s3://my-bucket/output/"},
    hyperParameters={"epochCount": "3", "batchSize": "32", "learningRate": "0.00001"},
)

# ── Provisioned Throughput ────────────────────────────────────────────────────
bedrock_mgmt.create_provisioned_model_throughput(
    modelUnits=1,
    provisionedModelName="my-provisioned-model",
    modelId="my-custom-model",
)
""")

# ── Live check: boto3/botocore importable ─────────────────────────────────────
print("━━━ Live import check ━━━")
try:
    import boto3
    import botocore
    print(f"  boto3=={boto3.__version__}  botocore=={botocore.__version__}  ✓")
except ImportError as e:
    print(f"  boto3/botocore not installed: {e}")
    print("  Install: uv pip install boto3")

try:
    import sagemaker
    print(f"  sagemaker=={sagemaker.__version__}  ✓")
except ImportError:
    print("  sagemaker not installed. Install: uv pip install sagemaker")

# ── Running example: pick a SageMaker instance for a model + method ───────────
# The cheatsheet shows a static instance table. SageMaker/Bedrock calls need AWS
# credentials, but instance *selection* is pure arithmetic we can run offline:
# estimate VRAM from model size + fine-tuning method, then pick the cheapest box
# that fits. The GB/B heuristics match the handbook's LoRA/QLoRA/Full-FT figures
# (QLoRA 8B≈5.5GB, LoRA≈14GB, Full≈80GB+) — see 15_lora_footprint.py.
print("\n━━━ Instance recommender (offline) ━━━")

# (instance, total VRAM GB) — smallest first
INSTANCES = [
    ("ml.p3.2xlarge  (V100 16GB)", 16),
    ("ml.g5.2xlarge  (A10G 24GB)", 24),
    ("ml.g5.12xlarge (4×A10G 96GB)", 96),
    ("ml.p3dn.24xlarge (8×V100 256GB)", 256),
]
VRAM_PER_B = {"qlora": 0.7, "lora": 2.0, "full": 16.0}   # GB per billion params


HEADROOM = 1.25   # +25% for activations, optimizer transients, fragmentation


def recommend_instance(params_b: float, method: str) -> str:
    need = params_b * VRAM_PER_B[method] * HEADROOM
    for name, vram in INSTANCES:
        if vram >= need:
            return f"{params_b:>4.0f}B {method:<5} -> ~{need:>5.1f}GB -> {name}"
    return f"{params_b:>4.0f}B {method:<5} -> ~{need:>5.1f}GB -> needs multi-node (>256GB)"


for pb in (7, 8, 13, 34, 70):
    for m in ("qlora", "lora", "full"):
        print("  " + recommend_instance(pb, m))

print("\nSageMaker/Bedrock pattern reference printed OK!")
