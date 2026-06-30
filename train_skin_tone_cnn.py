"""
肤色 3分类 CNN 训练 — MobileNetV3-Small 迁移学习
================================================
精度优先方案 v2:
- 3分类: 浅肤色/中间肤色/深肤色 (ITA° 临床标准)
- CNN 端到端学习，替代手工 ITA° 阈值，光照/相机色偏鲁棒
- 训练后 + R/B 比值后处理 → 5小类 (白皙/红润/自然/小麦/橄榄)
- 25 epochs + 强色彩增强 + LabelSmoothing
- 输出: stats_output/skin_tone_cnn.pth + skin_tone_cnn.onnx
"""
from __future__ import annotations
import os, sys, io, time, json
import numpy as np
import cv2

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)

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
MODEL_PTH = os.path.join(OUTPUT_DIR, 'skin_tone_cnn.pth')
MODEL_ONNX = os.path.join(OUTPUT_DIR, 'skin_tone_cnn.onnx')
REPORT_JSON = os.path.join(OUTPUT_DIR, 'skin_tone_training_report.json')

BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
INPUT_SIZE = 224
LABEL_SMOOTHING = 0.1
MAX_CLASS_WEIGHT = 5.0  # 类权重上限，防止小类翻车

# 3分类标签
IDX2TONE = ['深肤色', '中间肤色', '浅肤色']
TONE2IDX = {t: i for i, t in enumerate(IDX2TONE)}
NUM_CLASSES = len(IDX2TONE)


# ═══════════════════════════════════════════
#  伪标签生成 (仅 ITA°，不含 R/B 细分)
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
    """3分类肤色判定 (ITA° 临床标准):
    ITA° > 48  → 浅肤色 (白皙/红润)
    ITA° > 20  → 中间肤色 (自然)
    ITA° ≤ 20  → 深肤色 (小麦/橄榄)
    """
    if face_bgr.size == 0:
        return '中间肤色'
    try:
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab)
        skin_mask = _compute_skin_pixel_mask(face_bgr)
        if np.count_nonzero(skin_mask) < 100:
            skin_mask = np.ones(face_bgr.shape[:2], dtype=bool)

        # 亮度 Top-30%
        y_ch = ycrcb[:, :, 0][skin_mask]
        y_sorted = np.sort(y_ch)[::-1]
        top_n = max(len(y_sorted) // 3, 50)
        y_top = np.mean(y_sorted[:top_n])

        # Cr/Y 色度比
        cr_ch = ycrcb[:, :, 1][skin_mask]
        cr_mean = float(np.mean(cr_ch))
        chroma_ratio = cr_mean / max(y_top, 1)

        # ITA° (标准 CIE Lab)
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
#  数据增强
# ═══════════════════════════════════════════
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15
    )], p=0.9),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.25),
    T.RandomRotation(degrees=20, fill=128),
    T.RandomGrayscale(p=0.05),
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
#  模型
# ═══════════════════════════════════════════
class SkinToneCNN(nn.Module):
    def __init__(self, num_classes=3, dropout=0.35):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = 576
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.7),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.4),
            nn.Linear(128, num_classes),
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
#  训练工具
# ═══════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
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


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
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
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(model(imgs), 1)  # re-forward is OK for metrics
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


def print_metrics(name, res, class_names):
    yt, yp = res['labels'], res['preds']
    print(f'\n  {name} — Acc: {res["accuracy"]:.4f}  Loss: {res["loss"]:.4f}')
    print(classification_report(yt, yp, target_names=class_names, zero_division=0))


# ═══════════════════════════════════════════
#  ONNX 导出
# ═══════════════════════════════════════════
def export_onnx(model, path, device):
    try:
        import onnx
    except ImportError:
        print('[WARN] onnx 未安装')
        return
    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    torch.onnx.export(
        model, dummy, path,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=18, dynamo=False,
    )
    print(f'[ONNX] 已导出: {path} ({os.path.getsize(path)/1024/1024:.1f} MB)')


# ═══════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════
def main():
    print('=' * 70)
    print('  肤色 3分类 CNN 训练 — MobileNetV3-Small 迁移学习')
    print(f'  类别: {IDX2TONE}')
    print('  (v2: 3分类 ITA° 伪标签 + 强色彩增强 + LabelSmoothing)')
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
    nw = 0  # Windows CPU 训练: 避免共享内存耗尽
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=nw, pin_memory=False)
    print(f'[加载器] batch={BATCH_SIZE}, workers={nw}')

    # ── 类别权重 (上限裁剪) ──
    train_label_counts = [0] * NUM_CLASSES
    for f in train_files:
        idx = TONE2IDX.get(labels.get(f, '中间肤色'), 1)
        train_label_counts[idx] += 1
    max_count = max(train_label_counts)
    cls_weights = [min(max_count / max(c, 1), MAX_CLASS_WEIGHT) for c in train_label_counts]
    cls_weight = torch.tensor(cls_weights, dtype=torch.float32).to(device)
    print(f'[类别权重] {[round(w, 2) for w in cls_weights]} (上限 {MAX_CLASS_WEIGHT})')

    # ── 模型 ──
    model = SkinToneCNN(num_classes=NUM_CLASSES, dropout=0.35).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'[模型] 总参数: {total_params:,}', flush=True)

    criterion = nn.CrossEntropyLoss(weight=cls_weight, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    stopper = EarlyStopping(patience=12)

    # ── 训练 ──
    print(f'\n{"="*70}')
    print(f'  开始训练 ({NUM_EPOCHS} epochs)')
    print(f'{"="*70}', flush=True)

    best_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS)
        val_res = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_res['loss'])
        history['val_acc'].append(val_res['accuracy'])

        lr_now = scheduler.get_last_lr()[0]
        print(f'  Epoch {epoch:2d}/{NUM_EPOCHS} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | '
              f'Val Loss: {val_res["loss"]:.4f} Acc: {val_res["accuracy"]:.4f} | '
              f'LR: {lr_now:.2e}', flush=True)

        if val_res['accuracy'] > best_acc:
            best_acc = val_res['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'  ↑ 最佳模型 (Acc={best_acc:.4f})', flush=True)

        if stopper(val_res['accuracy']):
            print(f'  Early stopping at epoch {epoch}')
            break

    train_time = time.time() - t_start
    print(f'\n[训练完成] 耗时 {train_time/60:.1f} 分钟')

    # ── 保存 ──
    if best_state:
        model.load_state_dict(best_state)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': INPUT_SIZE,
        'class_names': IDX2TONE,
        'tone2idx': TONE2IDX,
        'transform_mean': [0.485, 0.456, 0.406],
        'transform_std': [0.229, 0.224, 0.225],
        'best_val_acc': best_acc,
        'history': history,
    }, MODEL_PTH)
    print(f'[保存] {MODEL_PTH}')

    # ── 测试集评估 ──
    print(f'\n{"="*70}')
    print(f'  测试集最终评估 (SCUT-FBP5500)')
    print(f'{"="*70}')
    test_res = evaluate(model, test_loader, criterion, device)
    print_metrics('Test', test_res, IDX2TONE)

    # ── ONNX ──
    export_onnx(model, MODEL_ONNX, device)

    # ── 报告 ──
    report = {
        'version': 'v2',
        'model': 'MobileNetV3-Small (3-class)',
        'framework': 'PyTorch',
        'input_size': INPUT_SIZE,
        'num_classes': NUM_CLASSES,
        'class_names': IDX2TONE,
        'pseudo_label_method': 'ITA° (CIE Lab, clinical standard)',
        'augmentation': [
            'RandomResizedCrop(0.6-1.0)', 'RandomHorizontalFlip(0.5)',
            'ColorJitter(b=0.5, c=0.5, s=0.4, h=0.15)',
            'GaussianBlur(p=0.25)', 'RandomRotation(±20°)', 'RandomGrayscale(p=0.05)',
        ],
        'train_params': {
            'batch_size': BATCH_SIZE,
            'epochs_trained': len(history['train_loss']),
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'label_smoothing': LABEL_SMOOTHING,
            'max_class_weight': MAX_CLASS_WEIGHT,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
        },
        'metrics': {
            'val_accuracy': round(float(best_acc), 4),
            'test_accuracy': round(float(test_res['accuracy']), 4),
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
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n[报告] {REPORT_JSON}')
    print(f'\n  训练完成! 准确率: {test_res["accuracy"]*100:.1f}%')
    print(f'  后处理: CNN 3分类 → R/B 比值细分 → 5标签 (白皙/红润/自然/小麦/橄榄)')


if __name__ == '__main__':
    main()
