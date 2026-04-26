"""
13_pytorch_internals.py
-----------------------
PyTorch's module system, weight surgery, and HuggingFace head swapping.

This is the chapter that saves you from silent bugs: why nn.ModuleList
exists, how to freeze/unfreeze layers, extract and modify weights, and
what HuggingFace actually does when you call AutoModelForSequenceClassification.

Run with: python scripts/13_pytorch_internals.py
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import copy

# ═══════════════════════════════════════════════════════════════════════════════
# PART A: nn.Module — Everything Is a Module
# ═══════════════════════════════════════════════════════════════════════════════
# In PyTorch, nn.Module is the base class for EVERYTHING:
#   - A single Linear layer is a Module
#   - A full GPT-2 model is a Module
#   - Your custom network is a Module
#
# The key behaviors:
#   - .parameters() → yields all trainable parameters (recursively)
#   - .named_parameters() → same but with names ("layer1.weight", etc.)
#   - .children() → direct sub-modules
#   - .modules() → ALL sub-modules (recursive)
#   - .state_dict() → ordered dict of all parameters (for saving/loading)
#   - .to(device) → moves all parameters to device
#   - .train() / .eval() → sets training mode (affects dropout, batchnorm)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART A: nn.Module — Everything Is a Module")
print("=" * 70)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


model = SimpleNet()
print(f"  Model:\n{model}\n")

# .parameters() finds everything automatically — because we used nn attributes
params = list(model.parameters())
print(f"  Total parameter tensors: {len(params)}")
for name, p in model.named_parameters():
    print(f"    {name:20s}  shape={list(p.shape)}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART B: The ModuleList Gotcha — The #1 Silent Bug
# ═══════════════════════════════════════════════════════════════════════════════
# If you store layers in a plain Python list, PyTorch CANNOT FIND THEM.
# They won't appear in .parameters(), won't move to GPU, won't be saved.
# This is the single most common silent bug for people coming from Python.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART B: The ModuleList Gotcha — #1 Silent Bug")
print("=" * 70)

# ── WRONG: Plain Python list ────────────────────────────────────────────────


class BrokenNet(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        # BUG: plain list — PyTorch doesn't know about these layers!
        self.layers = [nn.Linear(10, 10) for _ in range(num_layers)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


broken = BrokenNet()
broken_params = list(broken.parameters())
print(f"  BrokenNet (plain list):")
print(f"    Parameters found: {len(broken_params)}")
print(f"    ^ ZERO! The layers exist but PyTorch can't see them.")
print(f"    ^ They won't train, won't move to GPU, won't be saved.")

# ── CORRECT: nn.ModuleList ──────────────────────────────────────────────────


class FixedNet(nn.Module):
    def __init__(self, num_layers=3):
        super().__init__()
        # CORRECT: nn.ModuleList registers each layer as a sub-module
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


fixed = FixedNet()
fixed_params = list(fixed.parameters())
print(f"\n  FixedNet (nn.ModuleList):")
print(f"    Parameters found: {len(fixed_params)}")
for name, p in fixed.named_parameters():
    print(f"    {name:20s}  shape={list(p.shape)}")

# ── The same applies to dicts ───────────────────────────────────────────────


class DictNet(nn.Module):
    def __init__(self):
        super().__init__()
        # WRONG: plain dict → use nn.ModuleDict instead
        # self.heads = {"cls": nn.Linear(10, 2), "reg": nn.Linear(10, 1)}
        self.heads = nn.ModuleDict({
            "cls": nn.Linear(10, 2),
            "reg": nn.Linear(10, 1),
        })

    def forward(self, x, task="cls"):
        return self.heads[task](x)


dict_net = DictNet()
print(f"\n  DictNet (nn.ModuleDict):")
print(f"    Parameters found: {len(list(dict_net.parameters()))}")
for name, _ in dict_net.named_parameters():
    print(f"    {name}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART C: Non-Linear Forward Pass — Not Just Sequential
# ═══════════════════════════════════════════════════════════════════════════════
# nn.Sequential is fine for simple stacks, but real models have:
#   - Skip connections (ResNet)
#   - Multiple inputs/outputs
#   - Conditional branches
#   - Shared layers
#
# The forward() method is just Python — you can do anything.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART C: Non-Linear Forward — Skip Connections, Branches, Sharing")
print("=" * 70)


class ResidualBlock(nn.Module):
    """Skip connection — output = input + transformed(input)"""

    def __init__(self, dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # This is the non-linear part: x goes TWO paths
        return self.norm(x + self.ff(x))  # skip connection!


class MultiHeadModel(nn.Module):
    """One backbone, multiple task heads — common in multi-task learning."""

    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        # Different heads for different tasks
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x, task="classify"):
        features = self.backbone(x)     # shared computation
        if task == "classify":
            return self.classifier(features)
        else:
            return self.regressor(features)


mhm = MultiHeadModel()
x = torch.randn(2, 10)
cls_out = mhm(x, task="classify")
reg_out = mhm(x, task="regress")
print(f"  MultiHeadModel:")
print(f"    classify output shape: {cls_out.shape}")   # [2, 3]
print(f"    regress output shape:  {reg_out.shape}")    # [2, 1]
print(f"    Total params: {sum(p.numel() for p in mhm.parameters()):,}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART D: Weight Surgery — Extract, Inspect, Modify, Freeze
# ═══════════════════════════════════════════════════════════════════════════════
# Every parameter is a tensor. You can read it, copy it, zero it out,
# replace it, or freeze it. This is how transfer learning works.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART D: Weight Surgery — Extract, Inspect, Modify, Freeze")
print("=" * 70)

model = SimpleNet()

# ── D1: Extract and inspect weights ─────────────────────────────────────────
print("\n--- D1: Extract and inspect ---")
w1 = model.layer1.weight
b1 = model.layer1.bias
print(f"  layer1.weight shape: {w1.shape}")   # [20, 10]
print(f"  layer1.bias shape:   {b1.shape}")    # [20]
print(f"  weight mean: {w1.data.mean():.4f}, std: {w1.data.std():.4f}")
print(f"  requires_grad: {w1.requires_grad}")

# ── D2: Freeze specific layers (make non-trainable) ────────────────────────
print("\n--- D2: Freeze / unfreeze layers ---")


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())


print(f"  Before freeze: {count_trainable(model):,} / {count_total(model):,} trainable")

# Freeze layer1
for param in model.layer1.parameters():
    param.requires_grad = False

print(f"  After freezing layer1: {count_trainable(model):,} / {count_total(model):,} trainable")

# Unfreeze it back
for param in model.layer1.parameters():
    param.requires_grad = True

print(f"  After unfreezing: {count_trainable(model):,} / {count_total(model):,} trainable")

# ── D3: Freeze everything except the last layer (transfer learning pattern) ─
print("\n--- D3: Transfer learning pattern ---")
for param in model.parameters():
    param.requires_grad = False      # freeze all

for param in model.layer2.parameters():
    param.requires_grad = True       # unfreeze only the head

print(f"  Frozen all, unfrozen layer2: {count_trainable(model):,} / {count_total(model):,} trainable")
print(f"  ^ This is exactly what LoRA does, but more surgically (rank decomposition)")

# ── D4: Modify weights directly ────────────────────────────────────────────
print("\n--- D4: Modify weights directly ---")
# Zero out biases
with torch.no_grad():
    model.layer2.bias.zero_()
print(f"  layer2.bias after zero_(): {model.layer2.bias.data}")

# Copy weights from another model
model2 = SimpleNet()
with torch.no_grad():
    model.layer2.weight.copy_(model2.layer2.weight)
print(f"  Copied layer2.weight from model2 → model")

# ── D5: Replace a layer entirely ────────────────────────────────────────────
print("\n--- D5: Replace a layer ---")
print(f"  Before: {model.layer2}")
model.layer2 = nn.Linear(20, 10)  # different output dim
print(f"  After:  {model.layer2}")
print(f"  ^ New layer is trainable by default, old one is garbage collected")

# ── D6: state_dict — the complete weight snapshot ──────────────────────────
print("\n--- D6: state_dict ---")
# Reset model
model = SimpleNet()
sd = model.state_dict()
print(f"  state_dict keys: {list(sd.keys())}")
print(f"  layer1.weight shape: {sd['layer1.weight'].shape}")
# You can save/load state_dict to transfer weights between architectures
# torch.save(model.state_dict(), "weights.pt")
# model.load_state_dict(torch.load("weights.pt"))

# ═══════════════════════════════════════════════════════════════════════════════
# PART E: How HuggingFace Swaps Model Heads
# ═══════════════════════════════════════════════════════════════════════════════
# When you call AutoModelForCausalLM vs AutoModelForSequenceClassification
# from the same base model, HuggingFace:
#   1. Loads the same pretrained transformer backbone
#   2. STRIPS the original output head (lm_head for causal LM)
#   3. ADDS a new task-specific head (classifier, regression head, etc.)
#   4. The new head is randomly initialized (needs training!)
#
# This is not magic — it's just the layer replacement from Part D,
# automated by HuggingFace's model classes.
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART E: HuggingFace Head Swapping — What AutoModelFor* Actually Does")
print("=" * 70)

# ── E1: Load base model vs task-specific models ────────────────────────────
print("\n--- E1: Same base, different heads ---")

base = AutoModel.from_pretrained("gpt2")
causal = AutoModelForCausalLM.from_pretrained("gpt2")
classifier = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=3)

print(f"  AutoModel (base):")
print(f"    Type:   {type(base).__name__}")
print(f"    Params: {sum(p.numel() for p in base.parameters()):,}")

print(f"\n  AutoModelForCausalLM:")
print(f"    Type:   {type(causal).__name__}")
print(f"    Params: {sum(p.numel() for p in causal.parameters()):,}")

print(f"\n  AutoModelForSequenceClassification (3 labels):")
print(f"    Type:   {type(classifier).__name__}")
print(f"    Params: {sum(p.numel() for p in classifier.parameters()):,}")

# ── E2: Inspect the architecture difference ─────────────────────────────────
print("\n--- E2: What's different? The head. ---")

# CausalLM has: base transformer + lm_head (vocab-sized linear layer)
print(f"  CausalLM final layer:       {causal.lm_head}")
print(f"    lm_head shape: {causal.lm_head.weight.shape}")
print(f"    ^ Maps hidden_dim → vocab_size (for next token prediction)")

# Classifier has: base transformer + score (num_labels linear layer)
print(f"\n  Classifier final layer:     {classifier.score}")
print(f"    score shape: {classifier.score.weight.shape}")
print(f"    ^ Maps hidden_dim → num_labels (for classification)")
print(f"    ^ This layer is RANDOMLY INITIALIZED — it needs fine-tuning!")

# ── E3: The backbone is identical ──────────────────────────────────────────
print("\n--- E3: The backbone is identical ---")
base_sd = base.state_dict()
causal_backbone_sd = {k.replace("transformer.", ""): v
                      for k, v in causal.state_dict().items()
                      if k.startswith("transformer.")}

# Compare a few weights to prove they're the same
test_key = "h.0.attn.c_attn.weight"  # first attention layer
match = torch.equal(base_sd[test_key], causal_backbone_sd[test_key])
print(f"  h.0.attn.c_attn.weight identical? {match}")
print(f"  ^ Same pretrained weights. Only the head differs.")

# ── E4: What HuggingFace actually does (simplified) ────────────────────────
print("\n--- E4: What HF does under the hood (simplified) ---")
print("""
  # Pseudocode for AutoModelForSequenceClassification:
  class GPT2ForSequenceClassification(GPT2PreTrainedModel):
      def __init__(self, config):
          super().__init__(config)
          self.transformer = GPT2Model(config)  # ← same backbone
          self.score = nn.Linear(config.n_embd, config.num_labels,
                                 bias=False)      # ← NEW random head

      def forward(self, input_ids, ...):
          hidden = self.transformer(input_ids).last_hidden_state
          # Take the last token's representation (for classification)
          logits = self.score(hidden[:, -1, :])
          return logits
""")

# ── E5: Freeze backbone, train only the head ──────────────────────────────
print("--- E5: Freeze backbone, train only head ---")
for param in classifier.transformer.parameters():
    param.requires_grad = False

trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
total = sum(p.numel() for p in classifier.parameters())
print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
print(f"  ^ Only the classification head trains (like classical transfer learning)")
print(f"  ^ LoRA is better — it adds small trainable matrices INSIDE the backbone")

# ═══════════════════════════════════════════════════════════════════════════════
# PART F: Common AutoModelFor* Classes
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART F: AutoModelFor* — Which Class For Which Task")
print("=" * 70)
print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │  AutoModelFor*                    Head Added                     │
  ├──────────────────────────────────────────────────────────────────┤
  │  AutoModel                        None (raw hidden states)      │
  │  AutoModelForCausalLM             lm_head → vocab_size          │
  │  AutoModelForSeq2SeqLM            encoder-decoder + lm_head     │
  │  AutoModelForSequenceClassification  score → num_labels         │
  │  AutoModelForTokenClassification  classifier → num_labels       │
  │  AutoModelForQuestionAnswering    qa_outputs → 2 (start/end)    │
  │  AutoModelForMultipleChoice       classifier → 1 (per choice)   │
  │  AutoModelForMaskedLM             cls → vocab_size (MLM)        │
  ├──────────────────────────────────────────────────────────────────┤
  │  KEY INSIGHT: The backbone is always the same pretrained model.  │
  │  Only the final projection layer changes per task.               │
  │  The new head is randomly initialized — you MUST fine-tune it.  │
  └──────────────────────────────────────────────────────────────────┘
""")

# ═══════════════════════════════════════════════════════════════════════════════
# PART G: Inspecting Any Model's Architecture
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART G: Inspecting Any Model's Architecture")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained("gpt2")

# Method 1: print the model (shows full tree)
print("\n--- Method 1: print(model) (truncated) ---")
model_str = str(model)
lines = model_str.split("\n")
for line in lines[:15]:
    print(f"  {line}")
print(f"  ... ({len(lines)} lines total)")

# Method 2: named_modules() for the full hierarchy
print("\n--- Method 2: Top-level children ---")
for name, child in model.named_children():
    n_params = sum(p.numel() for p in child.parameters())
    print(f"  {name:20s}  type={type(child).__name__:30s}  params={n_params:,}")

# Method 3: named_parameters() to see every weight
print("\n--- Method 3: All parameter shapes (first 10) ---")
for i, (name, p) in enumerate(model.named_parameters()):
    if i >= 10:
        print(f"  ... and {sum(1 for _ in model.parameters()) - 10} more")
        break
    print(f"  {name:45s}  {str(list(p.shape)):20s}  grad={p.requires_grad}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART H: Putting It All Together — The Mental Model
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("MENTAL MODEL: PyTorch Module System")
print("=" * 70)
print("""
  ┌─────────────────────────────────────────────────────────────────┐
  │  nn.Module                                                      │
  │  ├─ Registers sub-modules as attributes (self.layer = ...)     │
  │  ├─ .parameters() finds them recursively                       │
  │  ├─ .to(device) moves them all                                 │
  │  └─ .state_dict() serializes them all                          │
  │                                                                 │
  │  GOTCHA: plain lists/dicts of modules → INVISIBLE to PyTorch    │
  │  FIX:    nn.ModuleList, nn.ModuleDict, nn.Sequential           │
  │                                                                 │
  │  forward() = just Python                                        │
  │  ├─ Can have branches, loops, conditionals                     │
  │  ├─ Skip connections, multi-head, shared layers — all fine     │
  │  └─ nn.Sequential is a convenience, NOT a requirement          │
  │                                                                 │
  │  Weight surgery                                                 │
  │  ├─ param.requires_grad = False   → freeze                     │
  │  ├─ param.requires_grad = True    → unfreeze                   │
  │  ├─ param.data.copy_(other)       → transfer weights           │
  │  ├─ model.layer = nn.Linear(...)  → replace layer              │
  │  └─ model.state_dict()            → save/load weights          │
  │                                                                 │
  │  HuggingFace AutoModelFor*                                      │
  │  ├─ Same backbone, different output heads                      │
  │  ├─ New head = randomly initialized = needs fine-tuning        │
  │  ├─ AutoModel = no head (raw features)                         │
  │  ├─ AutoModelForCausalLM = lm_head → vocab_size                │
  │  └─ AutoModelForSequenceClassification = score → num_labels    │
  └─────────────────────────────────────────────────────────────────┘
""")

print("PyTorch internals walkthrough OK!")
