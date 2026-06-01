"""
01_pytorch_basics.py
--------------------
Working PyTorch examples: tensors, autograd, nn.Module, DataLoader, save/load.
Run with: python scripts/01_pytorch_basics.py
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── 1. Tensors ────────────────────────────────────────────────────────────────
print("=== 1. Tensors ===")
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.zeros(2, 2)
print("x:", x)
print("x + y:", x + y)
print("x @ x.T:", x @ x.T)          # matrix multiply
print("x.shape:", x.shape, "dtype:", x.dtype)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
x = x.to(device)

# ── 2. Autograd ───────────────────────────────────────────────────────────────
print("\n=== 2. Autograd ===")
a = torch.tensor(3.0, requires_grad=True)
b = a ** 2 + 2 * a + 1          # b = (a+1)^2
b.backward()
print(f"a={a.item():.1f}  b={b.item():.1f}  db/da={a.grad.item():.1f}")  # 2*(3+1)=8

# ── 3. nn.Module ──────────────────────────────────────────────────────────────
print("\n=== 3. nn.Module ===")

class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = TinyMLP(4, 8, 2).to(device)
print(model)
print("params:", sum(p.numel() for p in model.parameters()))

# ── 4. Loss + Optimizer ───────────────────────────────────────────────────────
print("\n=== 4. Loss + Optimizer ===")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# ── 5. Dataset + DataLoader ───────────────────────────────────────────────────
print("\n=== 5. DataLoader ===")

class ToyDataset(Dataset):
    def __init__(self, n: int = 100):
        self.X = torch.randn(n, 4)
        self.y = (self.X[:, 0] > 0).long()   # binary label

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

ds = ToyDataset(200)
loader = DataLoader(ds, batch_size=16, shuffle=True)
print(f"dataset size={len(ds)}  batches={len(loader)}")

# ── 6. Training Loop ─────────────────────────────────────────────────────────
print("\n=== 6. Training Loop ===")
model.train()
for epoch in range(3):
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  epoch {epoch+1}  loss={total_loss/len(loader):.4f}")

# ── 7. Save + Load ────────────────────────────────────────────────────────────
print("\n=== 7. Save + Load ===")
import tempfile, os
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model.pt")
    torch.save(model.state_dict(), path)

    model2 = TinyMLP(4, 8, 2).to(device)
    model2.load_state_dict(torch.load(path, map_location=device))
    model2.eval()
    with torch.no_grad():
        sample = torch.randn(1, 4).to(device)
        out = model2(sample)
    print("  loaded model output:", out)

# ── 8. Mixed Precision (AMP) — the running example behind "2x faster" ─────────
# The cheatsheet claims autocast + GradScaler is "2x faster on modern GPUs".
# Here we actually time it so the claim is demonstrated, not asserted. AMP only
# speeds up on a CUDA GPU with Tensor Cores; on CPU it's a no-op (shown honestly).
print("\n=== 8. Mixed Precision (AMP) ===")
import time

# Compute-bound stack (big matmuls) so fp16 Tensor Cores can actually pay off.
big = nn.Sequential(
    nn.Linear(8192, 8192), nn.ReLU(),
    nn.Linear(8192, 8192), nn.ReLU(),
    nn.Linear(8192, 8192),
).to(device)
opt = torch.optim.SGD(big.parameters(), lr=1e-3)
data = torch.randn(512, 8192, device=device)
target = torch.randn(512, 8192, device=device)
loss_fn = nn.MSELoss()


def run(use_amp, steps=30, warmup=5):
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    def step():
        opt.zero_grad()
        with torch.autocast(device_type=device, enabled=use_amp):
            loss = loss_fn(big(data), target)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):   # warmup: pay CUDA init / cudnn autotune before timing
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


fp32 = run(use_amp=False)
amp = run(use_amp=True)
print(f"  device={device}  fp32={fp32:.3f}s  amp={amp:.3f}s  speedup={fp32/amp:.2f}x")
if device != "cuda":
    print("  (CPU has no Tensor Cores — AMP shows ~1x here. The 2x claim is GPU-only.)")

print("\nAll PyTorch basics OK!")