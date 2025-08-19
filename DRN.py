"""
Dual-Region Network (DRN) Implementation
========================================

This module provides a PyTorch implementation of the Dual-Region Network (DRN)
architecture introduced in:

    Zareh et al., 2025
    "Dual-Region Deep Learning for Odor Discrimination from Local Field Potentials"

Overview
--------
The DRN is designed to classify odor-evoked local field potentials (LFPs) recorded
from two brain regions: the olfactory bulb (OB) and the piriform cortex (PCx).
It uses parallel Region-Specific Networks (RSNs) to process each region
independently, then fuses them with an attention mechanism before classification.

Main Components
---------------
1. **ChannelAttention** (CA)
   - Learns channel-wise weights using distinct MLPs for max and average pooled
     features.

2. **SpatialAttention** (SA)
   - Learns spatial importance using pooled features, convolution, and sigmoid
     gating.

3. **ResidualBlock** (RB)
   - Pre-activation residual block with channel attention; supports downsampling
     by stride=2.

4. **RegionSpecificNet** (RSN)
   - Full processing pipeline per region:
     CA → ConvBlock → MaxPool → N×RB → SA → Global pooling → Feature Projection.

5. **DualRegionNet** (DRN)
   - Combines two RSNs (OB and PCx), applies Fusion Attention (FA), and feeds
     fused features into:
       • A deep classifier block
       • A shallow skip path
     Logits from both are summed for final predictions.

Training Utilities
------------------
- **DualInputDataset**: wraps OB, PCx, and labels into a PyTorch dataset.
- **FocalLoss**: γ=2.0, reduction='sum', robust to class imbalance.
- **Mixup**: on-the-fly augmentation applied consistently across OB/PCx.
- **train_epoch / validate**: single-epoch training and validation loops.
- **train_model**: full training routine (AdamW, gradient clipping, 70 epochs by default).
- **train_ensemble**: trains with K-fold CV and assembles an ensemble prediction
  on a held-out balanced test set (50 samples/class).

Usage
-----
Typical workflow:
    • Prepare OB and PCx feature arrays shaped (N, Channels, Bands).
    • Encode class labels to integers.
    • Call `train_ensemble(...)` with data arrays, config dict, and LabelEncoder.

Dependencies
------------
- Python 3.10+
- PyTorch >= 2.0
- NumPy, SciPy
- scikit-learn
- tqdm

License
-------
Released under the MIT license (see LICENSE file).

"""


import math
import torch
import torch.nn as nn


# ---------- shape helpers ----------
def _conv1d_out_len(L_in: int, kernel: int, stride: int, padding: int = 0, dilation: int = 1) -> int:
    # PyTorch formula
    return math.floor((L_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

def _pool1d_out_len(L_in: int, kernel: int, stride: int, padding: int = 0, dilation: int = 1) -> int:
    # Same as conv formula
    return math.floor((L_in + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)

def _after_maxpool2(L_in: int) -> int:
    # MaxPool1d(kernel=2, stride=2, padding=0)
    return _pool1d_out_len(L_in, kernel=2, stride=2, padding=0)

def _after_stride2_k3p1(L_in: int) -> int:
    # Conv1d(k=3, stride=2, padding=1) -> length_out = floor((L+1)/2) == ceil(L/2)
    return _conv1d_out_len(L_in, kernel=3, stride=2, padding=1)


# ---------- attention modules ----------
class ChannelAttention(nn.Module):
    """Channel attention with DISTINCT MLPs per pooling branch (as in the paper)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        assert channels > 0, "channels must be positive"
        hidden = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.avg_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.max_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.shape
        avg = self.avg_pool(x).view(b, c)
        mx  = self.max_pool(x).view(b, c)
        out = self.avg_mlp(avg) + self.max_mlp(mx)
        att = self.sigmoid(out).view(b, c, 1)
        return x * att


class SpatialAttention(nn.Module):
    """Pool (avg/max) → conv → BN → ReLU → conv → sigmoid; multiplies input."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 8, kernel_size=kernel_size, padding=padding, bias=True)
        self.bn    = nn.BatchNorm1d(8)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(8, 1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg, mx], dim=1)
        y = self.relu(self.bn(self.conv1(y)))
        y = self.conv2(y)
        att = self.sigmoid(y)
        return x * att


# ---------- residual block ----------
class ResidualBlock(nn.Module):
    """Pre-activation residual block with channel attention; can downsample with stride=2."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.bn1  = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2  = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.ca   = ChannelAttention(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.ca(out)
        out = out + self.shortcut(identity)
        return out


# ---------- region-specific net ----------
class RegionSpecificNet(nn.Module):
    """
    RSN pipeline:
      CA -> ConvBlock(two convs) -> MaxPool2 ->
      [RB x N (stride=2; each halves length & doubles channels)] ->
      SA -> (GAP || GMP) concat ->
      FP: half then x4 (→ output_dim = 4 * last_channels)

    By default:
      base_channels=64, max_channels=512 -> with enough spectral length, N=3 and output_dim=2048 (paper exact).
    """
    def __init__(
        self,
        num_channels: int,
        num_bands: int,
        *,
        dropout_rate: float = 0.25,
        base_channels: int = 64,
        max_channels: int = 512,
        num_residual_blocks: int | None = None,
        max_residual_blocks_cap: int = 8,
        min_len_after_blocks: int = 1,  # stop if next block would drop below this
    ):
        super().__init__()
        assert num_channels > 0 and num_bands > 0

        self.num_channels = num_channels
        self.num_bands    = num_bands
        self.base_channels = base_channels
        self.max_channels  = max_channels

        # --- Channel Attention ---
        self.channel_attention = ChannelAttention(num_channels)

        # --- Initial Conv Block (keeps length) + MaxPool2 (halves length) ---
        self.conv1 = nn.Sequential(
            nn.Conv1d(num_channels, base_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # halves length
        )

        # --- Decide how many residual blocks we can/should use ---
        # Compute post-pool length to plan RBs
        L = _after_maxpool2(num_bands)
        planned_blocks = 0
        ch = base_channels
        if num_residual_blocks is None:
            # Add blocks until we hit channel or length constraints or cap
            while planned_blocks < max_residual_blocks_cap:
                # check channel limit for *next* block
                if ch * 2 > max_channels:
                    break
                L_next = _after_stride2_k3p1(L)  # stride-2 conv in the block
                if L_next < min_len_after_blocks:
                    break
                planned_blocks += 1
                L, ch = L_next, ch * 2
        else:
            # Honor explicit number, but enforce channel cap & length
            planned_blocks = int(num_residual_blocks)
            ch = base_channels
            L_tmp = L
            for i in range(planned_blocks):
                if ch * 2 > max_channels:
                    planned_blocks = i
                    ch = ch  # unchanged
                    break
                L_tmp = _after_stride2_k3p1(L_tmp)
                if L_tmp < min_len_after_blocks:
                    planned_blocks = i
                    break
                ch *= 2
            L = L_tmp

        self.num_residual_blocks = planned_blocks
        self.last_channels = base_channels * (2 ** planned_blocks)

        # --- Residual stack (each doubles channels, halves length) ---
        blocks = []
        in_c = base_channels
        for _ in range(planned_blocks):
            out_c = in_c * 2
            blocks.append(ResidualBlock(in_c, out_c, stride=2))
            in_c = out_c
        self.residual_blocks = nn.Sequential(*blocks)

        # --- Spatial Attention ---
        self.spatial_attn = SpatialAttention()

        # --- Global pooling & projection ---
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)

        # concat(GAP, GMP) → size = 2 * last_channels
        self.feature_size = 2 * self.last_channels

        # Feature Projection (FP): half then x4 → output_dim = 2 * feature_size = 4 * last_channels
        hidden_fp = self.feature_size // 2
        out_fp    = 4 * hidden_fp  # == 2 * feature_size
        self.output_dim = out_fp

        self.feature_projection = nn.Sequential(
            nn.Linear(self.feature_size, hidden_fp),
            nn.BatchNorm1d(hidden_fp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_fp, out_fp),
            nn.BatchNorm1d(out_fp),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 0.5)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, C_in, L]
        x = self.channel_attention(x)
        x = self.conv1(x)
        if self.num_residual_blocks > 0:
            x = self.residual_blocks(x)
        x = self.spatial_attn(x)

        avg = self.global_avgpool(x).view(x.size(0), -1)
        mx  = self.global_maxpool(x).view(x.size(0), -1)
        feat = torch.cat([avg, mx], dim=1)           # [B, 2*last_channels]
        out  = self.feature_projection(feat)          # [B, output_dim]
        return out


# ---------- dual-region net ----------
class DualRegionNet(nn.Module):
    """
    Dual-Region Network (adaptive):
      - Plans residual blocks jointly so both RSNs yield the SAME output_dim for fusion.
      - Defaults reproduce the paper exactly (3 RBs → 2048-D per RSN → FA 4096→64→2; CF 2048→1024→32→16→C).
      - If spectral length allows and you raise `max_channels` and/or `max_residual_blocks_cap`, it will add blocks.

    Args:
      ob_channels, ob_bands:    OB input shape (C, L)
      pcx_channels, pcx_bands:  PCx input shape (C, L)
      num_classes:              number of classes
      dropout_rate:             dropout used in RSN FP and classifier
      base_channels:            channels after the initial conv block (default 64)
      max_channels:             cap for channels after residual stack (default 512 == paper)
      max_residual_blocks_cap:  hard cap on the number of RBs (default 8)
      min_len_after_blocks:     stop adding RBs if length would drop below this
    """
    def __init__(
        self,
        ob_channels: int, ob_bands: int,
        pcx_channels: int, pcx_bands: int,
        num_classes: int,
        *,
        dropout_rate: float = 0.25,
        base_channels: int = 64,
        max_channels: int = 512,            # 512 → paper default (3 RBs)
        max_residual_blocks_cap: int = 8,
        min_len_after_blocks: int = 1
    ):
        super().__init__()

        # --- plan shared residual depth so both regions have the same output_dim ---
        def plan_blocks(num_bands: int) -> tuple[int, int]:
            L = _after_maxpool2(num_bands)
            ch = base_channels
            blocks = 0
            while blocks < max_residual_blocks_cap and (ch * 2) <= max_channels:
                L_next = _after_stride2_k3p1(L)
                if L_next < min_len_after_blocks:
                    break
                blocks += 1
                ch *= 2
                L = L_next
            return blocks, ch  # ch is last_channels

        ob_blocks, ob_last_c   = plan_blocks(ob_bands)
        pcx_blocks, pcx_last_c = plan_blocks(pcx_bands)

        # Require both RSNs to use the SAME number of blocks (take the feasible minimum)
        shared_blocks = min(ob_blocks, pcx_blocks)

        # With that shared depth, the last channels are:
        last_channels = base_channels * (2 ** shared_blocks)
        # This also fixes RSN output_dim = 4 * last_channels (by RSN design).
        rsn_output_dim = 4 * last_channels

        # --- instantiate RSNs with SHARED num_residual_blocks ---
        self.ob_net = RegionSpecificNet(
            ob_channels, ob_bands,
            dropout_rate=dropout_rate,
            base_channels=base_channels,
            max_channels=max_channels,
            num_residual_blocks=shared_blocks,
            max_residual_blocks_cap=max_residual_blocks_cap,
            min_len_after_blocks=min_len_after_blocks,
        )
        self.pcx_net = RegionSpecificNet(
            pcx_channels, pcx_bands,
            dropout_rate=dropout_rate,
            base_channels=base_channels,
            max_channels=max_channels,
            num_residual_blocks=shared_blocks,
            max_residual_blocks_cap=max_residual_blocks_cap,
            min_len_after_blocks=min_len_after_blocks,
        )

        # Sanity: both RSNs must report same output size
        assert self.ob_net.output_dim == self.pcx_net.output_dim == rsn_output_dim

        # --- Fusion Attention: concat (2 * rsn_output_dim) -> 64 -> 2 -> softmax ---
        self.feature_dim = rsn_output_dim               # per-region
        self.fusion_in   = 2 * self.feature_dim         # concat
        self.fusion_attention = nn.Sequential(
            nn.Linear(self.fusion_in, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )

        # --- Fused representation keeps feature_dim (weighted sum of 2 regions) ---
        self.combined_dim = self.feature_dim

        # --- Classifier Block (adaptive but preserves paper when feature_dim=2048) ---
        # First hidden width: scale with feature_dim but clamp to [64, 1024]
        hidden1 = int(max(64, min(1024, self.combined_dim // 2)))
        hidden2 = 32
        hidden3 = 16

        self.classifier = nn.Sequential(
            nn.Linear(self.combined_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden3, num_classes)
        )

        # Shallow skip path (linear → logits)
        self.skip = nn.Linear(self.combined_dim, num_classes)

        # Expose for inspection
        self.num_residual_blocks = shared_blocks
        self.last_channels = last_channels
        self.rsn_output_dim = rsn_output_dim  # per-region feature dim (before fusion)

    def forward(self, ob_x, pcx_x):
        # ob_x:  [B, C_ob, L_ob]
        # pcx_x: [B, C_pcx, L_pcx]
        ob_feat  = self.ob_net(ob_x)            # [B, D]
        pcx_feat = self.pcx_net(pcx_x)          # [B, D]

        concat = torch.cat([ob_feat, pcx_feat], dim=1)     # [B, 2D]
        w = self.fusion_attention(concat)                  # [B, 2], softmax

        combined = w[:, 0:1] * ob_feat + w[:, 1:2] * pcx_feat   # [B, D]

        main_logits = self.classifier(combined)   # [B, C]
        skip_logits = self.skip(combined)         # [B, C]
        return main_logits + skip_logits          # summed logits


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


# -------- Dataset --------
class DualInputDataset(Dataset):
    """(OB, PCx, label) triple"""
    def __init__(self, ob_data: torch.Tensor, pcx_data: torch.Tensor, labels: torch.Tensor):
        assert len(ob_data) == len(pcx_data) == len(labels)
        self.ob_data = ob_data
        self.pcx_data = pcx_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.ob_data[idx], self.pcx_data[idx], self.labels[idx]


# -------- Focal Loss --------
class FocalLoss(nn.Module):
    """
    FL(pt) = - (1 - pt)^gamma * log(pt)
    Paper settings: gamma=2.0, alpha=None, reduction='sum'
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = 'sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [B, C] logits; targets: [B] class indices
        ce = F.cross_entropy(inputs, targets, reduction='none')  # [B]
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            fl = alpha_t * fl
        if self.reduction == 'mean':
            return fl.mean()
        if self.reduction == 'sum':
            return fl.sum()
        return fl


# -------- Mixup --------
def mixup_data(ob_x: torch.Tensor, pcx_x: torch.Tensor, y: torch.Tensor, alpha: float):
    assert alpha > 0.0
    b = y.size(0)
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(b, device=ob_x.device)
    mixed_ob  = lam * ob_x  + (1 - lam) * ob_x[idx]
    mixed_pcx = lam * pcx_x + (1 - lam) * pcx_x[idx]
    y_a, y_b = y, y[idx]
    return mixed_ob, mixed_pcx, y_a, y_b, float(lam)


def _infer_mixup_alpha(model: nn.Module, alpha4: float = 0.20, alpha7: float = 0.40, fallback: float = 0.20) -> float:
    try:
        ncls = model.classifier[-1].out_features
    except Exception:
        ncls = None
    if ncls == 4: return alpha4
    if ncls == 7: return alpha7
    return fallback


# -------- Train / Val (NO plots, NO TTA) --------
def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
                optimizer: optim.Optimizer, device: torch.device,
                *, use_mixup: bool = True, mixup_alpha: float | None = None):
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0

    if use_mixup and (mixup_alpha is None):
        mixup_alpha = _infer_mixup_alpha(model)

    for ob_x, pcx_x, y in tqdm(train_loader, desc="Training", leave=False):
        ob_x = ob_x.to(device, non_blocking=True)
        pcx_x = pcx_x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            mo, mp, ya, yb, lam = mixup_data(ob_x, pcx_x, y, alpha=mixup_alpha)
            logits = model(mo, mp)  # summed logits from DRN
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
        else:
            logits = model(ob_x, pcx_x)
            loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bsz = y.size(0)
        loss_sum += loss.item()
        seen += bsz
        with torch.no_grad():
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()

    return loss_sum / max(1, seen), 100.0 * correct / max(1, seen)


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    loss_sum, correct, seen = 0.0, 0, 0
    for ob_x, pcx_x, y in val_loader:
        ob_x = ob_x.to(device, non_blocking=True)
        pcx_x = pcx_x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(ob_x, pcx_x)
        loss = criterion(logits, y)
        bsz = y.size(0)
        loss_sum += loss.item()
        seen += bsz
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
    return loss_sum / max(1, seen), 100.0 * correct / max(1, seen)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, config: dict, fold: int | None = None):
    """
    Paper-consistent:
      - FocalLoss(gamma=2.0, alpha=None, reduction='sum')
      - AdamW(lr=1e-4, weight_decay=1e-5), fixed LR
      - 70 epochs default, best-val checkpoint kept
    """
    n_epochs   = int(config.get('num_epochs', 70))
    lr         = float(config.get('learning_rate', 1e-4))
    wd         = float(config.get('weight_decay', 1e-5))
    use_mixup  = bool(config.get('use_mixup', True))
    mix_alpha  = config.get('mixup_alpha', None)
    use_es     = bool(config.get('use_early_stopping', False))
    patience   = int(config.get('patience', 10))

    criterion = FocalLoss(gamma=2.0, alpha=None, reduction='sum')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = -1.0
    best_state = None
    wait = 0
    tag = f"[Fold {fold}] " if fold is not None else ""

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                      use_mixup=use_mixup, mixup_alpha=mix_alpha)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        print(f"{tag}Epoch {epoch:03d}/{n_epochs}  "
              f"Train loss/sample={tr_loss:.6f}, acc={tr_acc:.2f}%   "
              f"Val loss/sample={va_loss:.6f}, acc={va_acc:.2f}%")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            print(f"{tag}↳ New best val acc: {best_val_acc:.2f}%")
        else:
            wait += 1
            if use_es and wait >= patience:
                print(f"{tag}Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return model, best_val_acc, None


# -------- Ensemble (logits averaging, no plots) --------
def train_ensemble(X_ob, X_pcx, y_encoded, config, label_encoder, device=None):
    """Train an ensemble with perfectly balanced held-out test set and balanced K-fold CV.
       Ensemble = average of fold LOGITS (pre-softmax) → softmax → argmax."""
    print("\n=== Training Ensemble Models with Perfect Balance ===")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_folds = config['num_folds']
    num_classes = len(np.unique(y_encoded))

    # Split by class
    class_indices = [np.where(y_encoded == i)[0] for i in range(num_classes)]
    min_class_size = min(len(idx) for idx in class_indices)

    test_size_per_class = 50
    train_size_per_class = min_class_size - test_size_per_class
    if train_size_per_class <= 0:
        raise ValueError(f"Not enough samples: min class has {min_class_size}, need > {test_size_per_class}")

    print(f"Creating balanced splits with {train_size_per_class} train and {test_size_per_class} test per class")

    # Make balanced test split
    np.random.seed(42)
    test_indices, remaining_indices = [], []
    for cls in range(num_classes):
        idx = class_indices[cls].copy()
        np.random.shuffle(idx)
        test_indices.extend(idx[:test_size_per_class])
        remaining_indices.extend(idx[test_size_per_class:])

    # Test arrays
    X_ob_test = X_ob[test_indices]
    X_pcx_test = X_pcx[test_indices]
    y_test = y_encoded[test_indices]

    print("\nTest set class distribution:")
    counts = np.bincount(y_test, minlength=num_classes)
    for i in range(num_classes):
        print(f"  Class {i} ({label_encoder.classes_[i]}): {counts[i]} samples")

    # Per-class remaining indices for K-fold
    class_remaining_indices = [[] for _ in range(num_classes)]
    for idx in remaining_indices:
        class_remaining_indices[y_encoded[idx]].append(idx)

    samples_per_class_per_fold = [len(idx) // num_folds for idx in class_remaining_indices]

    models = []
    val_accs = []
    test_logits = []  # store [N_test, C] per fold

    for fold_idx in range(num_folds):
        print(f"\n=== Training Fold {fold_idx+1}/{num_folds} ===")
        train_indices, val_indices = [], []
        for cls in range(num_classes):
            cls_idx = class_remaining_indices[cls]
            fold_size = samples_per_class_per_fold[cls]
            start = fold_idx * fold_size
            end = (fold_idx + 1) * fold_size if fold_idx < num_folds - 1 else len(cls_idx)
            val_idx = cls_idx[start:end]
            tr_idx  = cls_idx[:start] + cls_idx[end:]
            val_indices.extend(val_idx)
            train_indices.extend(tr_idx)

        # Raw splits
        X_ob_tr, X_pcx_tr, y_tr = X_ob[train_indices], X_pcx[train_indices], y_encoded[train_indices]
        X_ob_va, X_pcx_va, y_va = X_ob[val_indices],   X_pcx[val_indices],   y_encoded[val_indices]

        print(f"Fold {fold_idx+1} class distribution:")
        tr_counts = np.bincount(y_tr, minlength=num_classes)
        va_counts = np.bincount(y_va, minlength=num_classes)
        for i in range(num_classes):
            print(f"  Class {i} ({label_encoder.classes_[i]}): {tr_counts[i]} train, {va_counts[i]} validation")

        # Robust scaling: fit on train only, apply to val & test (per region)
        X_ob_tr_norm, X_ob_va_norm, _ = normalize_data_robust(X_ob_tr, X_ob_va)
        X_pcx_tr_norm, X_pcx_va_norm, _ = normalize_data_robust(X_pcx_tr, X_pcx_va)
        _, _, X_ob_te_norm = normalize_data_robust(X_ob_tr, None, X_ob_test)
        _, _, X_pcx_te_norm = normalize_data_robust(X_pcx_tr, None, X_pcx_test)

        # Optional feature enhancement (same transform to all splits)
        X_ob_tr_en = enhance_features_advanced(X_ob_tr_norm)
        X_ob_va_en = enhance_features_advanced(X_ob_va_norm)
        X_ob_te_en = enhance_features_advanced(X_ob_te_norm)
        X_pcx_tr_en = enhance_features_advanced(X_pcx_tr_norm)
        X_pcx_va_en = enhance_features_advanced(X_pcx_va_norm)
        X_pcx_te_en = enhance_features_advanced(X_pcx_te_norm)

        # Tensors
        tr_ds = DualInputDataset(torch.FloatTensor(X_ob_tr_en),
                                 torch.FloatTensor(X_pcx_tr_en),
                                 torch.LongTensor(y_tr))
        va_ds = DualInputDataset(torch.FloatTensor(X_ob_va_en),
                                 torch.FloatTensor(X_pcx_va_en),
                                 torch.LongTensor(y_va))
        te_ds = DualInputDataset(torch.FloatTensor(X_ob_te_en),
                                 torch.FloatTensor(X_pcx_te_en),
                                 torch.LongTensor(y_test))

        train_loader = DataLoader(tr_ds, batch_size=config['batch_size'], shuffle=True,
                                  num_workers=2, pin_memory=True)
        val_loader = DataLoader(va_ds, batch_size=config['batch_size'], shuffle=False,
                                num_workers=2, pin_memory=True)
        test_loader = DataLoader(te_ds, batch_size=config['batch_size'], shuffle=False)

        # Model
        model = DualRegionNet(
            ob_channels=X_ob.shape[1], ob_bands=X_ob.shape[2],
            pcx_channels=X_pcx.shape[1], pcx_bands=X_pcx.shape[2],
            num_classes=num_classes, dropout_rate=config['dropout_rate']
        ).to(device)

        # Train one fold
        model, val_acc, _ = train_model(model, train_loader, val_loader, device, config, fold=fold_idx+1)
        models.append(model)
        val_accs.append(val_acc)

        # Collect fold LOGITS on the (fixed) test set
        model.eval()
        fold_logits_list = []
        with torch.no_grad():
            for ob_b, pcx_b, _ in test_loader:
                ob_b = ob_b.to(device, non_blocking=True)
                pcx_b = pcx_b.to(device, non_blocking=True)
                logits_b = model(ob_b, pcx_b)  # pre-softmax
                fold_logits_list.append(logits_b.cpu().numpy())
        test_logits.append(np.vstack(fold_logits_list))  # [N_test, C]

    # Ensemble: average logits, then softmax
    ensemble_logits = np.mean(test_logits, axis=0)  # [N_test, C]
    # softmax in numpy (stable)
    ensemble_logits -= ensemble_logits.max(axis=1, keepdims=True)
    ensemble_probs = np.exp(ensemble_logits)
    ensemble_probs /= ensemble_probs.sum(axis=1, keepdims=True)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    # Metrics
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    print(f"\nEnsemble Validation Accuracies: {val_accs}")
    print(f"Average Validation Accuracy: {np.mean(val_accs):.2f}%")
    print(f"Ensemble Test Accuracy: {ensemble_acc * 100:.2f}%")

    print("\nEnsemble Classification Report:")
    print(classification_report(y_test, ensemble_preds, target_names=label_encoder.classes_))

    # Per-class accuracy
    print("\nPer-class test accuracy:")
    for i in range(num_classes):
        mask = (y_test == i)
        correct = (ensemble_preds[mask] == y_test[mask]).sum()
        total = mask.sum()
        acc_i = 100.0 * correct / total if total > 0 else 0.0
        print(f"  Class {i} ({label_encoder.classes_[i]}): {correct}/{total} = {acc_i:.2f}%")

    return models, ensemble_acc, (y_test, ensemble_preds, ensemble_probs)
