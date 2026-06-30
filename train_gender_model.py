"""
性别分类器训练脚本 — 基于 SCUT-FBP5500_v2 数据集
==================================================
用 5500 张标注人脸 (AF/AM/CF/CM) 训练 ML 性别分类器，
替代 v52/v53 的手工启发式公式。

数据集:
  - AF = Asian Female  (~2000)
  - AM = Asian Male    (~2000)
  - CF = Caucasian Female (~750)
  - CM = Caucasian Male   (~750)

方法: LogisticRegressionCV + StandardScaler + PolynomialFeatures(2)
输出: stats_output/gender_model_v1.pkl
"""
from __future__ import annotations
import os, sys, io, time, pickle, gc
import numpy as np
from typing import Any

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**30)
import cv2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from scipy.stats import mannwhitneyu

# ═══════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'SCUT-FBP5500_v2', 'SCUT-FBP5500_v2')
IMAGES_DIR = os.path.join(DATASET_DIR, 'Images')
TRAIN_TXT = os.path.join(DATASET_DIR, 'train_test_files',
                         'split_of_60%training and 40%testing', 'train.txt')
TEST_TXT = os.path.join(DATASET_DIR, 'train_test_files',
                        'split_of_60%training and 40%testing', 'test.txt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'stats_output')
FEATURES_CACHE = os.path.join(OUTPUT_DIR, 'gender_features_cache.pkl')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'gender_model_v1.pkl')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'gender_training_report.json')

SUBSET_SAMPLE = None  # None=全量5500, N=采样快速测试

# ═══════════════════════════════════════════
#  性别标签解析
# ═══════════════════════════════════════════
def parse_gender(filename: str) -> int:
    """从 SCUT 文件名解析性别: AF/CF→0(女), AM/CM→1(男)"""
    prefix = filename[:2].upper()
    if prefix in ('AF', 'CF'):
        return 0  # female
    elif prefix in ('AM', 'CM'):
        return 1  # male
    raise ValueError(f'无法解析性别: {filename}')

def parse_ethnicity(filename: str) -> str:
    prefix = filename[0].upper()
    return 'Asian' if prefix == 'A' else 'Caucasian'

# ═══════════════════════════════════════════
#  特征提取 (与 _extract_32d_features 一致)
# ═══════════════════════════════════════════
def _hist_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = hist.flatten()
    hist = hist / max(hist.sum(), 1)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def _estimate_symmetry(face_gray: np.ndarray) -> float:
    _h, w = face_gray.shape
    mid = w // 2
    left = face_gray[:, :mid]
    right = cv2.flip(face_gray[:, mid:mid*2], 1) if mid*2 <= w else cv2.flip(face_gray[:, mid-1:], 1)
    min_w = min(left.shape[1], right.shape[1])
    if min_w < 5:
        return 0.5
    diff = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
    sim = 1.0 - np.mean(diff) / 128.0
    return float(np.clip(sim, 0.1, 0.99))

def _block_lap_std(face_gray: np.ndarray) -> float:
    h, w = face_gray.shape
    bh, bw = max(h // 4, 10), max(w // 4, 10)
    vars_list = []
    for i in range(4):
        for j in range(4):
            y1, y2 = i * bh, min((i+1)*bh, h)
            x1, x2 = j * bw, min((j+1)*bw, w)
            if y2 > y1 and x2 > x1:
                blk = face_gray[y1:y2, x1:x2]
                vars_list.append(cv2.Laplacian(blk, cv2.CV_64F).var())
    if not vars_list:
        return 0.0
    return float(np.std(vars_list) / max(np.mean(vars_list), 1))

def extract_features(img: np.ndarray,
                     face_rect: tuple | None = None) -> dict[str, float]:
    """提取 ~32 维视觉特征 (与 beauty_core._extract_32d_features 一致)"""
    h, w = img.shape[:2]
    feats: dict[str, float] = {}

    if img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # 全局特征
    feats['global_brightness'] = float(np.mean(gray)) / 255.0
    feats['global_contrast'] = float(np.std(gray)) / 128.0
    feats['global_sharpness'] = float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()))
    feats['global_entropy'] = float(_hist_entropy(gray))
    feats['global_saturation_mean'] = float(np.mean(hsv[:, :, 1])) / 255.0
    feats['global_saturation_std'] = float(np.std(hsv[:, :, 1])) / 128.0
    for ch, name in enumerate(['B', 'G', 'R']):
        ch_data = bgr[:, :, ch].flatten()
        feats[f'color_{name}_mean'] = float(np.mean(ch_data)) / 255.0
        feats[f'color_{name}_std'] = float(np.std(ch_data)) / 128.0
    r_m = float(np.mean(bgr[:, :, 2]))
    b_m = float(np.mean(bgr[:, :, 0]))
    feats['warmth_ratio'] = (r_m - b_m) / max(r_m + b_m, 1)

    # 人脸区域特征
    if face_rect:
        fx, fy, fw, fh = face_rect
        fx, fy = max(0, fx), max(0, fy)
        fw = min(fw, w - fx)
        fh = min(fh, h - fy)
        if fw > 10 and fh > 10:
            fr = bgr[fy:fy+fh, fx:fx+fw]
            fg = gray[fy:fy+fh, fx:fx+fw]
            fh_s = hsv[fy:fy+fh, fx:fx+fw]

            feats['face_area_ratio'] = (fw * fh) / max(w * h, 1)
            feats['face_aspect'] = fw / max(fh, 1)
            feats['face_x_ratio'] = fx / max(w, 1)
            feats['face_y_ratio'] = fy / max(h, 1)
            feats['face_brightness'] = float(np.mean(fg)) / 255.0
            feats['face_contrast'] = float(np.std(fg)) / 128.0
            feats['face_sharpness'] = float(np.log1p(cv2.Laplacian(fg, cv2.CV_64F).var()))
            feats['face_entropy'] = float(_hist_entropy(fg))
            feats['face_saturation'] = float(np.mean(fh_s[:, :, 1])) / 255.0
            for ch, name in enumerate(['B', 'G', 'R']):
                cd = fr[:, :, ch].flatten()
                feats[f'skin_{name}_mean'] = float(np.mean(cd)) / 255.0
                feats[f'skin_{name}_std'] = float(np.std(cd)) / 128.0
            if fw > 20 and fh > 20:
                feats['face_symmetry'] = _estimate_symmetry(fg)
            edges = cv2.Canny(fg, 50, 150)
            feats['face_edge_density'] = float(np.sum(edges > 0)) / max(edges.size, 1)
            th = fg[:fh//2, :]
            bh = fg[fh//2:, :]
            te = cv2.Canny(th, 50, 150)
            be = cv2.Canny(bh, 50, 150)
            td = np.sum(te > 0) / max(te.size, 1)
            bd = np.sum(be > 0) / max(be.size, 1)
            feats['face_topbot_edge_ratio'] = td / max(bd, 0.001)
            feats['face_lap_block_var'] = _block_lap_std(fg)

            # ── 性别专属特征 ──
            # 下半脸宽高比 (下颌区域)
            lower_face = fg[fh//2:, :]
            lh, lw = lower_face.shape
            if lh > 10 and lw > 10:
                # 下颌区域边缘密度 (男性通常更多胡茬/粗糙)
                le = cv2.Canny(lower_face, 50, 150)
                feats['jaw_edge_density'] = float(np.sum(le > 0)) / max(le.size, 1)
                # 颧骨区域锐度
                cheek_region = fg[fh//4:fh//2, fw//4:3*fw//4]
                if cheek_region.size > 0:
                    feats['cheek_sharpness'] = float(
                        np.log1p(cv2.Laplacian(cheek_region, cv2.CV_64F).var()))

    # 填充默认值
    default_keys = [
        'face_area_ratio', 'face_aspect', 'face_x_ratio', 'face_y_ratio',
        'face_brightness', 'face_contrast', 'face_sharpness', 'face_entropy',
        'face_saturation', 'face_symmetry', 'face_edge_density',
        'face_topbot_edge_ratio', 'face_lap_block_var',
        'jaw_edge_density', 'cheek_sharpness',
    ]
    for k in default_keys:
        if k not in feats:
            feats[k] = 0.0
    for ch in ['B', 'G', 'R']:
        for sf in ['_mean', '_std']:
            k = f'skin_{ch}{sf}'
            if k not in feats:
                feats[k] = 0.0
    return feats


# ═══════════════════════════════════════════
#  Haar Cascade 人脸检测
# ═══════════════════════════════════════════
def load_cascade():
    """加载 Haar 级联分类器"""
    cascade_paths = [
        os.path.join(PROJECT_ROOT, 'cascades', 'haarcascade_frontalface_default.xml'),
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    ]
    for p in cascade_paths:
        if os.path.exists(p):
            return cv2.CascadeClassifier(p)
    # 最后尝试用 cv2 内置路径
    try:
        return cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception:
        return None

def detect_face(img: np.ndarray, cascade) -> tuple | None:
    """检测最大人脸"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    # 选最大的脸
    best = max(faces, key=lambda r: r[2] * r[3])
    return tuple(best)


# ═══════════════════════════════════════════
#  主训练流程
# ═══════════════════════════════════════════
def main():
    print('=' * 60)
    print('  性别分类器训练 — SCUT-FBP5500_v2')
    print('=' * 60)

    # ── 加载级联 ──
    cascade = load_cascade()
    if cascade is None:
        print('[ERROR] 无法加载 Haar Cascade!')
        return
    print('[OK] Haar Cascade 已加载')

    # ── 获取所有图像 ──
    all_images = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    if SUBSET_SAMPLE:
        all_images = all_images[:SUBSET_SAMPLE]
    print(f'\n[数据] 共 {len(all_images)} 张图像')

    # ── 读取官方 train/test 分割 ──
    train_files = set()
    with open(TRAIN_TXT, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                train_files.add(line.split()[0])
    test_files = set()
    with open(TEST_TXT, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                test_files.add(line.split()[0])
    print(f'[Split] Train: {len(train_files)}, Test: {len(test_files)} (官方60/40)')

    # ── 统计性别分布 ──
    gender_counts = {'female': 0, 'male': 0}
    ethnic_counts = {'Asian_female': 0, 'Asian_male': 0,
                     'Caucasian_female': 0, 'Caucasian_male': 0}

    # ── 提取特征 (或读缓存) ──
    if os.path.exists(FEATURES_CACHE):
        print(f'\n[Cache] 加载缓存特征: {FEATURES_CACHE}')
        with open(FEATURES_CACHE, 'rb') as f:
            cache = pickle.load(f)
        features_data = cache['data']
        feat_keys = cache['feat_keys']
    else:
        print('\n[提取] 开始提取特征...')
        features_data = []
        t_start = time.time()
        failed = 0

        for idx, fname in enumerate(all_images):
            filepath = os.path.join(IMAGES_DIR, fname)
            try:
                img = cv2.imread(filepath)
                if img is None:
                    failed += 1
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                face_rect = detect_face(img, cascade)
                feats = extract_features(img, face_rect)

                gender = parse_gender(fname)
                ethnicity = parse_ethnicity(fname)

                entry = {
                    'filename': fname,
                    'gender': gender,
                    'ethnicity': ethnicity,
                    'has_face': face_rect is not None,
                    'feats': feats,
                }
                features_data.append(entry)

                if gender == 0:
                    gender_counts['female'] += 1
                    if ethnicity == 'Asian':
                        ethnic_counts['Asian_female'] += 1
                    else:
                        ethnic_counts['Caucasian_female'] += 1
                else:
                    gender_counts['male'] += 1
                    if ethnicity == 'Asian':
                        ethnic_counts['Asian_male'] += 1
                    else:
                        ethnic_counts['Caucasian_male'] += 1

            except Exception as e:
                failed += 1
                continue

            if (idx + 1) % 500 == 0:
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed
                eta = (len(all_images) - idx - 1) / rate
                print(f'  ... {idx+1}/{len(all_images)} '
                      f'({rate:.1f} 张/秒, ETA {eta:.0f}秒)')

        elapsed = time.time() - t_start
        print(f'[提取完成] {len(features_data)} 张成功, {failed} 张失败, '
              f'耗时 {elapsed:.0f}秒')

        # 确定特征键
        feat_keys = sorted(features_data[0]['feats'].keys())
        print(f'[特征] {len(feat_keys)} 维: {feat_keys}')

        # 保存缓存
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(FEATURES_CACHE, 'wb') as f:
            pickle.dump({'data': features_data, 'feat_keys': feat_keys}, f)
        print(f'[Cache] 已保存: {FEATURES_CACHE}')

    # ── 统计 ──
    print(f'\n[性别分布] 女: {gender_counts["female"]}, 男: {gender_counts["male"]}')
    print(f'[种族分布] 亚女: {ethnic_counts["Asian_female"]}, '
          f'亚男: {ethnic_counts["Asian_male"]}, '
          f'高加索女: {ethnic_counts["Caucasian_female"]}, '
          f'高加索男: {ethnic_counts["Caucasian_male"]}')

    # ── 构建特征矩阵 ──
    feat_keys = sorted(features_data[0]['feats'].keys())
    X = np.array([[d['feats'].get(k, 0.0) for k in feat_keys]
                  for d in features_data], dtype=np.float64)
    y = np.array([d['gender'] for d in features_data], dtype=np.int32)

    # ── 去除无效值 ──
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    # 同时过滤 features_data
    valid_indices = np.where(valid_mask)[0]
    features_data_valid = [features_data[i] for i in valid_indices]
    print(f'[清洗] 有效样本: {len(y)} (丢弃 {len(valid_mask) - len(y)} 个无效)')

    # ── 按官方 split 分割 ──
    train_idx = [i for i, d in enumerate(features_data_valid)
                 if d['filename'] in train_files]
    test_idx = [i for i, d in enumerate(features_data_valid)
                if d['filename'] in test_files]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f'[Split] Train: {len(y_train)}, Test: {len(y_test)}')
    print(f'  Train 女/男: {np.sum(y_train==0)}/{np.sum(y_train==1)}')
    print(f'  Test  女/男: {np.sum(y_test==0)}/{np.sum(y_test==1)}')

    # ── 特征归一化 ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 多项式特征 ──
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    print(f'[特征] 原始 {len(feat_keys)} → 多项式 {X_train_poly.shape[1]} 维')

    # ── 训练 LogisticRegressionCV ──
    print('\n[训练] LogisticRegressionCV (L2, Cs=10)...')
    t0 = time.time()
    clf = LogisticRegressionCV(
        Cs=10, cv=5, penalty='l2', solver='lbfgs',
        max_iter=2000, class_weight='balanced',
        random_state=42, n_jobs=-1,
    )
    clf.fit(X_train_poly, y_train)
    train_time = time.time() - t0
    print(f'[训练完成] 耗时 {train_time:.1f}秒, 最优C={clf.C_[0]:.4f}')

    # ── 评估 ──
    y_pred = clf.predict(X_test_poly)
    y_prob = clf.predict_proba(X_test_poly)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)

    print(f'\n{"="*60}')
    print(f'  性别分类器评估结果')
    print(f'{"="*60}')
    print(f'  准确率 (Accuracy):  {acc:.4f}  ({acc*100:.1f}%)')
    print(f'  精确率 (Precision): {prec:.4f}  (男性预测中真正男性的比例)')
    print(f'  召回率 (Recall):    {rec:.4f}  (真男性中被检出的比例)')
    print(f'  F1 分数:            {f1:.4f}')
    print(f'  ROC-AUC:            {auc:.4f}')
    print(f'\n  混淆矩阵:')
    print(f'                预测女  预测男')
    print(f'  实际女        {cm[0,0]:5d}   {cm[0,1]:5d}')
    print(f'  实际男        {cm[1,0]:5d}   {cm[1,1]:5d}')

    # 女性误检率
    female_misclass = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    male_misclass = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    print(f'\n  女性误检为男性: {female_misclass:.2%}')
    print(f'  男性误检为女性: {male_misclass:.2%}')

    print(f'\n  详细分类报告:')
    print(classification_report(y_test, y_pred, target_names=['女性', '男性']))

    # ── 特征重要性 (系数绝对值) ──
    print(f'\n{"="*60}')
    print(f'  特征重要性 (LogisticRegression |coef|, Top-20)')
    print(f'{"="*60}')
    coef = np.abs(clf.coef_[0])
    feat_names = poly.get_feature_names_out(feat_keys)
    top_indices = np.argsort(coef)[::-1][:20]
    for rank, idx in enumerate(top_indices):
        bar = '█' * int(coef[idx] / coef[top_indices[0]] * 30)
        sign = '+' if clf.coef_[0][idx] > 0 else '-'
        print(f'  {rank+1:2d}. {sign} {feat_names[idx]:50s} '
              f'|coef|={coef[idx]:.4f} {bar}')

    # ── 分种族评估 ──
    print(f'\n{"="*60}')
    print(f'  分种族评估')
    print(f'{"="*60}')
    for eth in ['Asian', 'Caucasian']:
        eth_idx = [i for i in test_idx
                   if features_data_valid[i]['ethnicity'] == eth]
        if len(eth_idx) < 10:
            continue
        X_e = X_test_poly[[test_idx.index(i) for i in eth_idx
                           if i in test_idx]]
        y_e = y_test[[test_idx.index(i) for i in eth_idx
                     if i in test_idx]]
        if len(y_e) < 10:
            continue
        y_pred_e = clf.predict(X_e)
        acc_e = accuracy_score(y_e, y_pred_e)
        print(f'  {eth:12s}: n={len(y_e):4d}, '
              f'Accuracy={acc_e:.4f} ({acc_e*100:.1f}%)')

    # ── 概率分布 ──
    print(f'\n[概率] 男性概率分布:')
    print(f'  Test 女性 prob 均值: {y_prob[y_test==0].mean():.3f} '
          f'(正确→靠近0)')
    print(f'  Test 男性 prob 均值: {y_prob[y_test==1].mean():.3f} '
          f'(正确→靠近1)')

    # ── 保存模型 ──
    model = {
        'version': 'v1',
        'scaler': scaler,
        'poly': poly,
        'clf': clf,
        'feat_keys': feat_keys,
        'metrics': {
            'accuracy': round(float(acc), 4),
            'precision': round(float(prec), 4),
            'recall': round(float(rec), 4),
            'f1': round(float(f1), 4),
            'roc_auc': round(float(auc), 4),
            'female_misclass_rate': round(float(female_misclass), 4),
            'male_misclass_rate': round(float(male_misclass), 4),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'n_features_raw': len(feat_keys),
            'n_features_poly': X_train_poly.shape[1],
        },
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f'\n[保存] 模型已保存至: {MODEL_PATH}')
    print(f'  文件大小: {os.path.getsize(MODEL_PATH) / 1024:.1f} KB')

    # ── 保存报告 ──
    import json
    report = {
        'version': 'v1',
        'dataset': 'SCUT-FBP5500_v2',
        'samples': {
            'total': len(y),
            'train': len(y_train), 'test': len(y_test),
            'female': int(np.sum(y == 0)), 'male': int(np.sum(y == 1)),
        },
        'model': 'LogisticRegressionCV (L2, 5-fold CV)',
        'feature_count_raw': len(feat_keys),
        'feature_count_poly': X_train_poly.shape[1],
        'metrics': model['metrics'],
    }
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'[报告] 已保存至: {REPORT_PATH}')

    # ── 结论 ──
    print(f'\n{"="*60}')
    print(f'  总结')
    print(f'{"="*60}')
    if acc > 0.85:
        print(f'  ✅ 准确率 {acc*100:.1f}% > 85% — 模型性能优秀，可替代手工公式!')
    elif acc > 0.75:
        print(f'  ⚠️  准确率 {acc*100:.1f}% > 75% — 可用，但建议持续改进')
    else:
        print(f'  ❌ 准确率 {acc*100:.1f}% < 75% — 需要更多特征或不同模型')
    print(f'  女性误检率: {female_misclass*100:.1f}% (v52手工公式远高于此)')
    print(f'  模型路径: {MODEL_PATH}')


if __name__ == '__main__':
    main()
