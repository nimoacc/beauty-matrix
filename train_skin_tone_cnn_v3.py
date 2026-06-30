"""
肤色 3分类 CNN 训练 v3 — EfficientNet-B0 + Mixup + SWA
========================================================
精度提升方案:
- 骨架升级: MobileNetV3-Small (1.2M) → EfficientNet-B0 (5.3M)
- Mixup 数据增强 (alpha=0.2): 标签平滑插值, 适合肤色连续谱
- SWA 随机权重平均: 最终 10 epoch 取平均, 消除过拟合尖峰
- 40 epochs + warmup(3) + CosineAnnealing
- 更轻的色彩抖动 (避免干扰色度学习)
- TTA: 测试时多裁切投票
"""
from __future__ import annotations
import os, sys, io, time, json, copy
import numpy as np
import cv2

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms as T
from sklearn.metrics import (accuracy_score, classification_report)

# ═══════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'SCUT-FBP5500_v2', 'SCUT-FBP5500_v2')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'stats_output', 'gender_cnn_cache')
LABEL_CACHE = os.path.join(PROJECT_ROOT, 'stats_output', 'skin_tone_3class_labels.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'stats_output')
TRAIN_TXT = os.path.join(DATASET_DIR, 'train_test_files',
                         'split_of_60%training and 40%testing', 'train.txt')
TEST_TXT = os.path.join(DATASET_DIR, 'train_test_files',
                        'split_of_60%training and 40%testing', 'test.txt')
MODEL_PTH = os.path.join(OUTPUT_DIR, 'skin_tone_cnn_v3.pth')
MODEL_ONNX = os.path.join(OUTPUT_DIR, 'skin_tone_cnn_v3.onnx')
REPORT_JSON = os.path.join(OUTPUT_DIR, 'skin_tone_training_report_v3.json')

BATCH_SIZE = 32
NUM_EPOCHS = 40
WARMUP_EPOCHS = 3
LEARNING_RATE = 1e-3          # EfficientNet 建议更高 LR
WEIGHT_DECAY = 1e-4
INPUT_SIZE = 224
LABEL_SMOOTHING = 0.08
MIXUP_ALPHA = 0.2             # Mixup Beta 分布参数
MAX_CLASS_WEIGHT = 4.0
SWA_START = 30                # SWA 起始 epoch
SWA_FREQ = 1                  # SWA 每 N epoch 取一次快照

IDX2TONE = ['深肤色', '中间肤色', '浅肤色']
TONE2IDX = {t: i for i, t in enumerate(IDX2TONE)}
NUM_CLASSES = len(IDX2TONE)


# ═══════════════════════════════════════════
#  伪标签生成 (复用 v2 逻辑)
# ═══════════════════════════════════════════
def _compute_skin_pixel_mask(bgr_arr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2YCrCb)
    lower_hsv_light = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv_light = np.array([25, 170, 255], dtype=np.uint8)
    mask_light = cv2.inRange(hsv, lower_hsv_light, upper_hsv_light)
    lower_hsv_dark = np.array([0, 20, 30], dtype=np.uint8)
    upper_hsv_dark = np.array([25, 200, 180], dtype=np.uint8)
    mask_dark = cv2.inRange(hsv, lower_hsv_dark, upper_hsv_dark)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    return cv2.bitwise_or(cv2.bitwise_or(mask_light, mask_dark), mask_ycrcb) > 0


def classify_skin_tone_3class(face_bgr: np.ndarray) -> str:
    if face_bgr.size == 0:
        return '中间肤色'
    try:
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab)
        skin_mask = _compute_skin_pixel_mask(face_bgr)
        if np.count_nonzero(skin_mask) < 100:
            skin_mask = np.ones(face_bgr.shape[:2], dtype=bool)

        y_ch = ycrcb[:, :, 0][skin_mask]
        y_sorted = np.sort(y_ch)[::-1]
        top_n = max(len(y_sorted) // 3, 50)
        y_top = np.mean(y_sorted[:top_n])

        cr_ch = ycrcb[:, :, 1][skin_mask]
        cr_mean = float(np.mean(cr_ch))
        chroma_ratio = cr_mean / max(y_top, 1)

        l_ch = lab[:, :, 0][skin_mask].astype(np.float64)
        b_ch = lab[:, :, 2][skin_mask].astype(np.float64)
        l_cie = np.mean(l_ch) * 100.0 / 255.0
        b_cie = np.mean(b_ch) - 128.0
        ita = float(np.degrees(np.arctan2(l_cie - 50.0, b_cie)))

        if ita > 50 and chroma_ratio < 0.78:
            return '浅肤色'
        elif ita > 48:
            return '浅肤色'
        elif ita > 20:
            return '中间肤色'
        else:
            return '深肤色'
    except Exception:
        return '中间肤色'


def generate_pseudo_labels():
    os.makedirs(os.path.dirname(LABEL_CACHE), exist_ok=True)
    if os.path.exists(LABEL_CACHE):
        with open(LABEL_CACHE, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f'[标签] 加载已有缓存: {len(labels)} 条')
        return labels

    print('[标签] 为 SCUT-FBP5500 生成 3分类肤色伪标签...')
    cached = sorted([f for f in os.listdir(CACHE_DIR)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    labels = {}
    total = len(cached)
    for i, fname in enumerate(cached):
        fpath = os.path.join(CACHE_DIR, fname)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            labels[fname] = '中间肤色'
            continue
        labels[fname] = classify_skin_tone_3class(img_bgr)
        if (i + 1) % 500 == 0:
            print(f'  进度: {i + 1}/{total}')

    dist = {}
    for t in labels.values():
        dist[t] = dist.get(t, 0) + 1
    print(f'  标签分布: {dict(sorted(dist.items()))} 总计: {len(labels)}')

    with open(LABEL_CACHE, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    return labels


# ═══════════════════════════════════════════
#  数据增强 (v3: 更轻的色彩抖动 + Mixup)
# ═══════════════════════════════════════════
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(INPUT_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
    T.RandomHorizontalFlip(p=0.5),
    # v3: 更轻的色彩抖动 — EfficientNet 对色度变化更敏感
    T.RandomApply([T.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.25, hue=0.1
    )], p=0.85),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
    T.RandomRotation(degrees=15, fill=128),
    T.RandomGrayscale(p=0.03),
    # RandAugment-mini: 额外随机增强
    T.RandomApply([T.RandomAdjustSharpness(sharpness_factor=2, p=0.5)], p=0.3),
    T.RandomApply([T.RandomPosterize(bits=5, p=0.5)], p=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((INPUT_SIZE, INPUT_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ═══════════════════════════════════════════
#  Mixup 实现
# ═══════════════════════════════════════════
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device):
    """返回 mixup 后的 (x_mixed, y_a, y_b, lam)"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    x_mixed = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return x_mixed, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 损失: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════
class SkinToneDataset(Dataset):
    def __init__(self, filenames, labels, images_dir, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        filepath = os.path.join(self.images_dir, fname)
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            dummy = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
            dummy_tensor = T.ToTensor()(dummy)
            dummy_tensor = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])(dummy_tensor)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tone = self.labels.get(fname, '中间肤色')
        label = TONE2IDX.get(tone, 1)

        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

        return img_tensor, torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════
#  模型: EfficientNet-B0 (v3)
# ═══════════════════════════════════════════
class SkinToneEfficientNet(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[1].in_features  # 1280

        # 替换分类头
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.7, inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout * 0.3, inplace=True),
            nn.Linear(256, num_classes),
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ═══════════════════════════════════════════
#  SWA (随机权重平均)
# ═══════════════════════════════════════════
class SWA:
    def __init__(self, model, start_epoch: int, freq: int = 1):
        self.model = model
        self.start_epoch = start_epoch
        self.freq = freq
        self.swa_model = None
        self.n_models = 0
        self.swa_state = None

    def update(self, epoch: int):
        if self.swa_state is None:
            self.swa_state = {k: v.clone().float() for k, v in self.model.state_dict().items()}
        if epoch >= self.start_epoch and (epoch - self.start_epoch) % self.freq == 0:
            for k, v in self.model.state_dict().items():
                self.swa_state[k] = self.swa_state[k].to(v.device, dtype=torch.float32)
                self.swa_state[k] = (self.swa_state[k] * self.n_models + v.float()) / (self.n_models + 1)
            self.n_models += 1

    def apply_to(self, model):
        """将 SWA 权重写入目标模型"""
        model.load_state_dict(self.swa_state)
        return model


# ═══════════════════════════════════════════
#  训练工具
# ═══════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
            return False
        if val_acc < self.best_score + self.min_delta:
            self.counter += 1
            return self.counter >= self.patience
        else:
            self.best_score = val_acc
            self.counter = 0
            return False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs,
                    use_mixup=True, mixup_alpha=MIXUP_ALPHA):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)
    for bi, (imgs, labels) in enumerate(loader):
        valid = labels >= 0
        imgs, labels = imgs[valid].to(device), labels[valid].to(device)
        if imgs.size(0) == 0:
            continue

        # Mixup (仅在训练集应用)
        if use_mixup and mixup_alpha > 0:
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, mixup_alpha, device)

        optimizer.zero_grad()
        outputs = model(imgs)

        if use_mixup and mixup_alpha > 0:
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        # 梯度裁剪 (EfficientNet 训练稳定性)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        if use_mixup and mixup_alpha > 0:
            correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
        else:
            correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if (bi + 1) % 50 == 0 or bi == n_batches - 1:
            print(f'    [E{epoch}/{total_epochs}] Batch {bi+1}/{n_batches} '
                  f'loss={total_loss/max(total,1):.3f} acc={correct/max(total,1):.3f}',
                  flush=True)

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        valid = labels >= 0
        imgs, labels = imgs[valid].to(device), labels[valid].to(device)
        if imgs.size(0) == 0:
            continue
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    return {
        'loss': total_loss / max(total, 1),
        'accuracy': acc,
        'labels': np.array(all_labels),
        'preds': np.array(all_preds),
    }


@torch.no_grad()
def evaluate_tta(model, loader, device, num_crops=5):
    """测试时增强: 多裁切 + 水平翻转投票"""
    model.eval()
    all_labels, all_preds = [], []

    # TTA transforms
    tta_transforms = []
    # 5种裁切: 中心 + 4个角
    crop_size = int(INPUT_SIZE * 0.85)
    offsets = [(0, 0), (0, INPUT_SIZE - crop_size), (INPUT_SIZE - crop_size, 0),
               (INPUT_SIZE - crop_size, INPUT_SIZE - crop_size),
               ((INPUT_SIZE - crop_size) // 2, (INPUT_SIZE - crop_size) // 2)]

    for imgs, labels in loader:
        valid = labels >= 0
        imgs, labels = imgs[valid].to(device), labels[valid].to(device)
        if imgs.size(0) == 0:
            continue

        batch_size = imgs.size(0)
        # 收集所有裁切的预测
        all_crop_logits = torch.zeros(batch_size, NUM_CLASSES, len(offsets) * 2).to(device)
        crop_idx = 0

        for ox, oy in offsets:
            crop = imgs[:, :, oy:oy + crop_size, ox:ox + crop_size]
            crop_resized = torch.nn.functional.interpolate(crop, size=INPUT_SIZE, mode='bilinear', align_corners=False)
            # 原始裁切
            out = model(crop_resized)
            all_crop_logits[:, :, crop_idx] = out
            crop_idx += 1
            # 水平翻转裁切
            out_flip = model(torch.flip(crop_resized, dims=[3]))
            all_crop_logits[:, :, crop_idx] = out_flip
            crop_idx += 1

        # 平均所有裁切 logits 再 softmax
        avg_logits = torch.mean(all_crop_logits, dim=2)
        _, preds = torch.max(avg_logits, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return {'accuracy': acc, 'labels': np.array(all_labels), 'preds': np.array(all_preds)}


def print_metrics(name, res, class_names):
    yt, yp = res['labels'], res['preds']
    print(f'\n  {name} — Acc: {res["accuracy"]:.4f}')
    print(classification_report(yt, yp, target_names=class_names, zero_division=0))


# ═══════════════════════════════════════════
#  ONNX 导出
# ═══════════════════════════════════════════
def export_onnx(model, path, device):
    try:
        import onnx
    except ImportError:
        print('[WARN] onnx 未安装, 跳过 ONNX 导出')
        return
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    torch.onnx.export(
        model, dummy, path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=18, dynamo=False,
    )
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f'[ONNX] 已导出: {path} ({size_mb:.1f} MB)')


# ═══════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════
def main():
    print('=' * 70)
    print('  肤色 3分类 CNN 训练 v3 — EfficientNet-B0 + Mixup + SWA')
    print(f'  类别: {IDX2TONE}')
    print('  (v3: 更大骨架 + Mixup增强 + SWA权重平均 + 40 Epochs)')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[设备] {device}')

    # ── 标签 ──
    labels = generate_pseudo_labels()

    # ── 数据分割 ──
    with open(TRAIN_TXT, 'r') as f:
        train_files = [ln.strip().split()[0] for ln in f if ln.strip()]
    with open(TEST_TXT, 'r') as f:
        test_files = [ln.strip().split()[0] for ln in f if ln.strip()]

    existing = set(os.listdir(CACHE_DIR))
    train_files = [f for f in train_files if f in existing and f in labels]
    test_files = [f for f in test_files if f in existing and f in labels]
    print(f'[数据] Train: {len(train_files)}, Test: {len(test_files)}')

    for name, files in [('Train', train_files), ('Test', test_files)]:
        dist = {}
        for f in files:
            t = labels.get(f, '中间肤色')
            dist[t] = dist.get(t, 0) + 1
        print(f'  {name}: {dict(sorted(dist.items()))}')

    # ── Dataset & DataLoader ──
    train_ds = SkinToneDataset(train_files, labels, CACHE_DIR, transform=train_transform)
    test_ds = SkinToneDataset(test_files, labels, CACHE_DIR, transform=val_transform)
    nw = 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=nw, pin_memory=False)
    print(f'[加载器] batch={BATCH_SIZE}, workers={nw}')

    # ── 类别权重 ──
    train_label_counts = [0] * NUM_CLASSES
    for f in train_files:
        idx = TONE2IDX.get(labels.get(f, '中间肤色'), 1)
        train_label_counts[idx] += 1
    max_count = max(train_label_counts)
    cls_weights = [min(max_count / max(c, 1), MAX_CLASS_WEIGHT) for c in train_label_counts]
    cls_weight = torch.tensor(cls_weights, dtype=torch.float32).to(device)
    print(f'[类别权重] {[round(w, 2) for w in cls_weights]} (上限 {MAX_CLASS_WEIGHT})')

    # ── 模型 (v3: EfficientNet-B0) ──
    model = SkinToneEfficientNet(num_classes=NUM_CLASSES, dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[模型] EfficientNet-B0, 总参数: {total_params:,} ({total_params/1e6:.1f}M)', flush=True)

    criterion = nn.CrossEntropyLoss(weight=cls_weight, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 学习率策略: Warmup(3) + CosineAnnealing(37)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
    scheduler = SequentialLR(optimizer,
                              schedulers=[warmup_scheduler, cosine_scheduler],
                              milestones=[WARMUP_EPOCHS])

    stopper = EarlyStopping(patience=18)
    swa = SWA(model, start_epoch=SWA_START, freq=SWA_FREQ)

    # ── 训练 ──
    print(f'\n{"="*70}')
    print(f'  开始训练 ({NUM_EPOCHS} epochs, Mixup α={MIXUP_ALPHA})')
    print(f'{"="*70}', flush=True)

    best_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        # 前 5 个 epoch 不用 Mixup, 让模型先收敛到基本方向
        use_mixup_flag = epoch > 5

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS,
            use_mixup=use_mixup_flag)

        val_res = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # SWA 更新
        swa.update(epoch)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_res['loss'])
        history['val_acc'].append(val_res['accuracy'])

        lr_now = optimizer.param_groups[0]['lr']
        mixup_tag = 'M' if use_mixup_flag else ' '
        print(f'  Epoch {epoch:2d}/{NUM_EPOCHS} [{mixup_tag}] | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_res["loss"]:.4f} Acc: {val_res["accuracy"]:.4f} | '
              f'LR: {lr_now:.2e}', flush=True)

        if val_res['accuracy'] > best_acc:
            best_acc = val_res['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  ↑ 新最佳模型 (Acc={best_acc:.4f})', flush=True)

        if stopper(val_res['accuracy']):
            print(f'  Early stopping at epoch {epoch}')
            break

    train_time = time.time() - t_start
    print(f'\n[训练完成] 耗时 {train_time/60:.1f} 分钟')
    print(f'[最佳验证准确率] {best_acc:.4f}')

    # ── 加载最佳权重 + SWA ──
    if best_state:
        model.load_state_dict(best_state)

    # 最终模型 = SWA 平均 (优于单次最佳)
    final_model = SkinToneEfficientNet(num_classes=NUM_CLASSES, dropout=0.3).to(device)
    if swa.n_models > 0:
        swa.apply_to(final_model)
        print(f'[SWA] 使用 {swa.n_models} 个模型快照的平均权重')

        # 比较 SWA vs 最佳单次模型
        best_res = evaluate(model, test_loader, criterion, device)
        swa_res = evaluate(final_model, test_loader, criterion, device)
        print(f'  最佳单次: {best_res["accuracy"]:.4f}')
        print(f'  SWA 平均: {swa_res["accuracy"]:.4f}')

        if swa_res['accuracy'] > best_res['accuracy']:
            print('[SWA] SWA 优于最佳单次 ✓')
            model = final_model
        else:
            print('[SWA] SWA 劣于最佳单次, 使用最佳单次')
    else:
        print('[SWA] SWA 未激活 (epochs < SWA_START)')

    # ── 保存模型 ──
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': INPUT_SIZE,
        'class_names': IDX2TONE,
        'tone2idx': TONE2IDX,
        'transform_mean': [0.485, 0.456, 0.406],
        'transform_std': [0.229, 0.224, 0.225],
        'best_val_acc': best_acc,
        'history': history,
        'swa_used': swa.n_models > 0,
        'swa_n_models': swa.n_models,
    }, MODEL_PTH)
    print(f'[保存] {MODEL_PTH}')

    # ── 测试集评估 ──
    print(f'\n{"="*70}')
    print(f'  测试集最终评估 (SCUT-FBP5500)')
    print(f'{"="*70}')

    # 标准评估
    test_res = evaluate(model, test_loader, criterion, device)
    print_metrics('Test (标准)', test_res, IDX2TONE)

    # TTA 评估
    test_tta_res = evaluate_tta(model, test_loader, device, num_crops=5)
    print_metrics('Test (TTA, 5裁切)', test_tta_res, IDX2TONE)
    print(f'  Δ TTA vs 标准: {test_tta_res["accuracy"] - test_res["accuracy"]:+.4f}')

    # ── ONNX ──
    export_onnx(model, MODEL_ONNX, device)

    # ── 报告 ──
    report = {
        'version': 'v3',
        'model': 'EfficientNet-B0 (3-class)',
        'framework': 'PyTorch',
        'input_size': INPUT_SIZE,
        'num_classes': NUM_CLASSES,
        'class_names': IDX2TONE,
        'pseudo_label_method': 'ITA° (CIE Lab, clinical standard)',
        'improvements_vs_v2': [
            'MobileNetV3-Small → EfficientNet-B0 (1.2M → 5.3M 参数)',
            'Mixup augmentation (α=0.2, epoch 6+)',
            'SWA 随机权重平均 (epoch 30-40)',
            'Warmup 3 epochs + CosineAnnealing',
            'Gradient clipping (max_norm=2.0)',
            'TTA 多裁切投票评估',
            'RandAugment-mini (Sharpness + Posterize)',
        ],
        'augmentation': [
            'RandomResizedCrop(0.5-1.0)', 'RandomHorizontalFlip(0.5)',
            'ColorJitter(b=0.3, c=0.3, s=0.25, h=0.1)',
            'GaussianBlur(p=0.2)', 'RandomRotation(±15°)', 'RandomGrayscale(p=0.03)',
            'RandomAdjustSharpness(p=0.3)', 'RandomPosterize(p=0.1)',
            f'Mixup(alpha={MIXUP_ALPHA})',
        ],
        'train_params': {
            'batch_size': BATCH_SIZE,
            'epochs_trained': len(history['train_loss']),
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'label_smoothing': LABEL_SMOOTHING,
            'mixup_alpha': MIXUP_ALPHA,
            'max_class_weight': MAX_CLASS_WEIGHT,
            'swa_start_epoch': SWA_START,
            'swa_n_models': swa.n_models,
            'optimizer': 'AdamW',
            'scheduler': 'Warmup(3) + CosineAnnealing',
        },
        'metrics': {
            'val_best_accuracy': round(float(best_acc), 4),
            'test_accuracy': round(float(test_res['accuracy']), 4),
            'test_tta_accuracy': round(float(test_tta_res['accuracy']), 4),
            'test_loss': round(float(test_res['loss']), 4),
            'n_train': len(train_files),
            'n_test': len(test_files),
            'total_params': total_params,
        },
        'label_distribution': {
            'train': {t: sum(1 for f in train_files if labels.get(f) == t) for t in IDX2TONE},
            'test': {t: sum(1 for f in test_files if labels.get(f) == t) for t in IDX2TONE},
        },
        'training_time_minutes': round(train_time / 60, 1),
        'v2_baseline': 0.6709,
        'v3_improvement': round(float(test_res['accuracy']) - 0.6709, 4),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n[报告] {REPORT_JSON}')
    print(f'\n{"="*70}')
    print(f'  训练完成!')
    print(f'  v2 基准: 67.09%  (MobileNetV3-Small, 25 epochs)')
    print(f'  v3 标准: {test_res["accuracy"]*100:.2f}%  (EfficientNet-B0, {len(history["train_loss"])} epochs)')
    print(f'  v3 TTA:  {test_tta_res["accuracy"]*100:.2f}% (多裁切投票)')
    print(f'  提升:    {test_res["accuracy"]*100 - 67.09:+.2f}%')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
