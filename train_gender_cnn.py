"""
性别分类 CNN 模型训练 — MobileNetV3-Small 迁移学习
===================================================
长期方案: 用轻量 CNN 替代手工特征 + LogisticRegressionCV，
通过数据增强解决 Domain Shift 问题。

数据集: SCUT-FBP5500_v2 (5500张标注人脸)
模型: torchvision MobileNetV3-Small (pretrained on ImageNet)
增强: 随机亮度/对比度/色调/翻转/旋转/裁剪 → 对抗光照/风格偏移
输出: stats_output/gender_cnn_v2.pth + gender_cnn_v2.onnx

Python 版本: 确保 pip install torch torchvision onnx onnxruntime
"""
from __future__ import annotations
import os, sys, io, time, json, gc
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

# ═══════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'SCUT-FBP5500_v2', 'SCUT-FBP5500_v2')
IMAGES_DIR = os.path.join(DATASET_DIR, 'Images')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'stats_output', 'gender_cnn_cache')
TRAIN_TXT = os.path.join(DATASET_DIR, 'train_test_files',
                         'split_of_60%training and 40%testing', 'train.txt')
TEST_TXT = os.path.join(DATASET_DIR, 'train_test_files',
                        'split_of_60%training and 40%testing', 'test.txt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'stats_output')
MODEL_PTH = os.path.join(OUTPUT_DIR, 'gender_cnn_v2.pth')
MODEL_ONNX = os.path.join(OUTPUT_DIR, 'gender_cnn_v2.onnx')
REPORT_JSON = os.path.join(OUTPUT_DIR, 'gender_cnn_training_report.json')
REVERSE_HILL_DIR = os.path.join(PROJECT_ROOT, 'Reverse_Hill')

BATCH_SIZE = 32
NUM_EPOCHS = 15  # 充分训练，预计 ~25 分钟 (CPU)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
INPUT_SIZE = 224

CASCADE_PATH = os.path.join(PROJECT_ROOT, 'cascades',
                            'haarcascade_frontalface_default.xml')

# ═══════════════════════════════════════════
#  人脸检测
# ═══════════════════════════════════════════
_face_cascade = None

def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        for p in [CASCADE_PATH,
                  cv2.data.haarcascades + 'haarcascade_frontalface_default.xml']:
            if os.path.exists(p):
                _face_cascade = cv2.CascadeClassifier(p)
                break
    return _face_cascade

def detect_face_roi(img_bgr: np.ndarray, margin=0.3):
    """检测最大人脸，返回带边距的裁剪区域"""
    cascade = _get_cascade()
    if cascade is None or cascade.empty():
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    # 加边距
    mx = int(w * margin)
    my = int(h * margin)
    H, W = img_bgr.shape[:2]
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)
    return img_bgr[y1:y2, x1:x2]


# ═══════════════════════════════════════════
#  数据增强 (关键: 对抗 Domain Shift)
# ═══════════════════════════════════════════
train_transform = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
    )], p=0.8),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.2),
    T.RandomRotation(degrees=15, fill=128),
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
class GenderDataset(Dataset):
    """SCUT-FBP5500 性别分类数据集 (从预缓存人脸目录读取)"""

    def __init__(self, filenames: list[str], images_dir: str,
                 transform=None):
        self.filenames = filenames
        self.images_dir = images_dir  # 缓存目录 (已裁剪人脸)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def parse_gender(fname: str) -> int:
        prefix = fname[:2].upper()
        if prefix in ('AF', 'CF'):
            return 0  # female
        elif prefix in ('AM', 'CM'):
            return 1  # male
        raise ValueError(f'无法解析性别: {fname}')

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        filepath = os.path.join(self.images_dir, fname)

        # 直接从缓存读取 (已是 BGR 裁剪人脸)
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            dummy = np.full((INPUT_SIZE, INPUT_SIZE, 3), 128, dtype=np.uint8)
            dummy_tensor = T.ToTensor()(dummy)
            dummy_tensor = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])(dummy_tensor)
            return dummy_tensor, torch.tensor(-1, dtype=torch.long)

        # BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        label = self.parse_gender(fname)

        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0

        return img_tensor, torch.tensor(label, dtype=torch.long)


# ═══════════════════════════════════════════
#  模型
# ═══════════════════════════════════════════
class GenderCNN(nn.Module):
    """MobileNetV3-Small + 2层分类头"""

    def __init__(self, dropout=0.3):
        super().__init__()
        # 预训练 MobileNetV3-Small
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # 分类头 (MobileNetV3-Small 最后卷积输出 576 通道)
        in_features = 576
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
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
        x = self.classifier(x)
        return x


# ═══════════════════════════════════════════
#  训练工具
# ═══════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
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
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = val_acc
            self.counter = 0
        return False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch=0, total_epochs=0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)
    for bi, (imgs, labels) in enumerate(loader):
        # 跳过损坏的图片
        valid = labels >= 0
        imgs = imgs[valid].to(device)
        labels = labels[valid].to(device)
        if imgs.size(0) == 0:
            continue

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
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
    all_labels, all_preds, all_probs = [], [], []

    for imgs, labels in loader:
        valid = labels >= 0
        imgs = imgs[valid].to(device)
        labels = labels[valid].to(device)
        if imgs.size(0) == 0:
            continue

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return {
        'loss': total_loss / max(total, 1),
        'accuracy': acc,
        'labels': np.array(all_labels),
        'preds': np.array(all_preds),
        'probs': np.array(all_probs),
    }


def print_metrics(name, res):
    yt, yp, yprob = res['labels'], res['preds'], res['probs']
    cm = confusion_matrix(yt, yp)
    print(f'\n  {name} — Acc: {res["accuracy"]:.4f}  Loss: {res["loss"]:.4f}')
    print(f'                预测女  预测男')
    if cm.shape == (2, 2):
        print(f'  实际女        {cm[0,0]:5d}   {cm[0,1]:5d}')
        print(f'  实际男        {cm[1,0]:5d}   {cm[1,1]:5d}')
        f_mis = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        m_mis = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        print(f'  女性误检率: {f_mis:.2%}  男性误检率: {m_mis:.2%}')
    if len(set(yt)) == 2:
        try:
            auc = roc_auc_score(yt, yprob)
            print(f'  ROC-AUC: {auc:.4f}')
        except Exception:
            pass


# ═══════════════════════════════════════════
#  ONNX 导出
# ═══════════════════════════════════════════
def export_onnx(model, path, device):
    """导出 ONNX 模型供 onnxruntime 推理"""
    try:
        import onnx
    except ImportError:
        print('[WARN] onnx 未安装，跳过 ONNX 导出。pip install onnx')
        return

    model.eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
    torch.onnx.export(
        model, dummy, path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},
            'output': {0: 'batch'},
        },
        opset_version=18,
    )
    print(f'[ONNX] 已导出: {path}')


# ═══════════════════════════════════════════
#  Reverse_Hill 验证
# ═══════════════════════════════════════════
@torch.no_grad()
def test_reverse_hill(model, device):
    """用 CNN 模型重新检测 Reverse_Hill"""
    if not os.path.isdir(REVERSE_HILL_DIR):
        print('[跳过] Reverse_Hill 目录不存在')
        return

    model.eval()
    files = sorted([
        f for f in os.listdir(REVERSE_HILL_DIR)
        if os.path.isfile(os.path.join(REVERSE_HILL_DIR, f))
        and f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ], key=lambda x: int(os.path.splitext(x)[0]))

    print(f'\n{"="*80}')
    print(f'  Reverse_Hill CNN 验证 ({len(files)} 张)')
    print(f'{"="*80}')
    print(f'{"编号":<6} {"文件":<12} {"预测":<8} {"prob_男":<10} {"prob_女":<10}')
    print('-' * 60)

    male_cnt = female_cnt = noface_cnt = 0
    for fname in files:
        fpath = os.path.join(REVERSE_HILL_DIR, fname)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            noface_cnt += 1
            print(f'{os.path.splitext(fname)[0]:<6} {fname:<12} {"读取失败":<8}')
            continue

        face_roi = detect_face_roi(img_bgr, margin=0.3)
        if face_roi is None or face_roi.size == 0:
            noface_cnt += 1
            print(f'{os.path.splitext(fname)[0]:<6} {fname:<12} {"无人脸":<8}')
            continue

        img_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        img_tensor = val_transform(img_rgb).unsqueeze(0).to(device)
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        prob_f, prob_m = probs[0].item(), probs[1].item()
        pred = '男性' if prob_m > prob_f else '女性'

        if pred == '男性':
            male_cnt += 1
        else:
            female_cnt += 1

        print(f'{os.path.splitext(fname)[0]:<6} {fname:<12} {pred:<8} '
              f'{prob_m:<10.4f} {prob_f:<10.4f}')

    total = male_cnt + female_cnt
    print(f'\n  ───────────────────────────────────────')
    print(f'  检测汇总: 男性 {male_cnt} ({male_cnt/max(total,1)*100:.0f}%), '
          f'女性 {female_cnt} ({female_cnt/max(total,1)*100:.0f}%), '
          f'无人脸 {noface_cnt}')
    print(f'  (注: 该目录实际绝大多数为女性，女性占比越高效果越好)')

    return {'male': male_cnt, 'female': female_cnt, 'noface': noface_cnt, 'total': total}


# ═══════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════
def main():
    print('=' * 70)
    print('  性别分类 CNN 训练 — MobileNetV3-Small 迁移学习')
    print('  (长期方案: 卷积特征 + 数据增强 → 对抗 Domain Shift)')
    print('=' * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[设备] {device}')

    # ── 加载数据分割 ──
    with open(TRAIN_TXT, 'r') as f:
        train_files = [ln.strip().split()[0] for ln in f if ln.strip()]
    with open(TEST_TXT, 'r') as f:
        test_files = [ln.strip().split()[0] for ln in f if ln.strip()]

    # 过滤实际存在的文件 (缓存目录)
    existing = set(os.listdir(CACHE_DIR))
    train_files = [f for f in train_files if f in existing]
    test_files = [f for f in test_files if f in existing]

    print(f'[数据] Train: {len(train_files)}, Test: {len(test_files)}')

    # 性别分布
    train_f = sum(1 for f in train_files if GenderDataset.parse_gender(f) == 0)
    train_m = len(train_files) - train_f
    test_f = sum(1 for f in test_files if GenderDataset.parse_gender(f) == 0)
    test_m = len(test_files) - test_f
    print(f'  Train 女/男: {train_f}/{train_m}')
    print(f'  Test  女/男: {test_f}/{test_m}')

    # ── Dataset & DataLoader (使用预缓存人脸) ──
    train_ds = GenderDataset(train_files, CACHE_DIR, transform=train_transform)
    test_ds = GenderDataset(test_files, CACHE_DIR, transform=val_transform)

    nw = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=nw, pin_memory=True)

    print(f'[加载器] batch={BATCH_SIZE}, workers={nw}')

    # ── 模型 ──
    model = GenderCNN(dropout=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[模型] 总参数: {total_params:,}, 可训练: {trainable:,}', flush=True)

    # 类别权重 (处理轻微不平衡)
    cls_weight = torch.tensor([1.0, train_f / max(train_m, 1)]).to(device)

    criterion = nn.CrossEntropyLoss(weight=cls_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    stopper = EarlyStopping(patience=10)

    # ── 训练 ──
    print(f'\n{"="*70}')
    print(f'  开始训练 ({NUM_EPOCHS} epochs, lr={LEARNING_RATE})')
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
            print(f'  ↑ 最佳模型 (Acc={best_acc:.4f})')

        if stopper(val_res['accuracy']):
            print(f'  Early stopping at epoch {epoch}')
            break

    train_time = time.time() - t_start
    print(f'\n[训练完成] 耗时 {train_time/60:.1f} 分钟')

    # ── 加载最佳模型并评估 ──
    if best_state:
        model.load_state_dict(best_state)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': INPUT_SIZE,
        'transform_mean': [0.485, 0.456, 0.406],
        'transform_std': [0.229, 0.224, 0.225],
        'best_val_acc': best_acc,
        'history': history,
    }, MODEL_PTH)
    print(f'[保存] {MODEL_PTH}')

    # ── 测试集最终评估 ──
    print(f'\n{"="*70}')
    print(f'  测试集最终评估 (SCUT-FBP5500)')
    print(f'{"="*70}')
    test_res = evaluate(model, test_loader, criterion, device)
    print_metrics('Test', test_res)
    print(f'\n  详细分类报告:')
    print(classification_report(test_res['labels'], test_res['preds'],
                                target_names=['女性', '男性']))

    # ── 分种族评估 ──
    print(f'\n{"="*70}')
    print(f'  分种族评估')
    print(f'{"="*70}')
    for eth_prefix, eth_name in [('A', 'Asian'), ('C', 'Caucasian')]:
        eth_files = [f for f in test_files if f[0].upper() == eth_prefix]
        if len(eth_files) < 10:
            continue
        eth_ds = GenderDataset(eth_files, CACHE_DIR, transform=val_transform)
        eth_loader = DataLoader(eth_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=nw)
        eth_res = evaluate(model, eth_loader, criterion, device)
        print(f'\n  {eth_name} (n={len(eth_files)}): Acc={eth_res["accuracy"]:.4f}')
        e_f = sum(1 for f in eth_files if GenderDataset.parse_gender(f) == 0)
        e_m = len(eth_files) - e_f
        print(f'    女={e_f}, 男={e_m}')

    # ── ONNX 导出 ──
    export_onnx(model, MODEL_ONNX, device)

    # ── Reverse_Hill 验证 ──
    rh_result = test_reverse_hill(model, device)

    # ── 报告 ──
    report = {
        'version': 'v2',
        'model': 'MobileNetV3-Small (transfer learning)',
        'framework': 'PyTorch',
        'input_size': INPUT_SIZE,
        'dataset': 'SCUT-FBP5500_v2 (official 60/40 split)',
        'augmentation': [
            'RandomResizedCrop(0.7-1.0)',
            'RandomHorizontalFlip(0.5)',
            'ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)',
            'GaussianBlur(p=0.2)',
            'RandomRotation(±15°)',
        ],
        'train_params': {
            'batch_size': BATCH_SIZE,
            'epochs_trained': len(history['train_loss']),
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
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
        'training_time_minutes': round(train_time / 60, 1),
        'reverse_hill_result': rh_result,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'\n[报告] {REPORT_JSON}')

    # ── 总结 ──
    print(f'\n{"="*70}')
    print(f'  训练总结')
    print(f'{"="*70}')
    print(f'  模型: MobileNetV3-Small 迁移学习')
    print(f'  SCUT 测试集准确率: {test_res["accuracy"]*100:.1f}%')
    if rh_result:
        fr = rh_result['female'] / max(rh_result['total'], 1)
        print(f'  Reverse_Hill 女性检出率: {fr*100:.0f}% (越接近真实(≈100%)越好)')
    print(f'  模型文件: {MODEL_PTH}')
    print(f'  ONNX 文件: {MODEL_ONNX}')
    print(f'  对比旧模型 (80.9% Acc, Reverse_Hill 女性检出率 ~16%):')
    print(f'    → CNN 方案应显著改善泛化能力与 Domain Shift 鲁棒性')


if __name__ == '__main__':
    main()
