"""
颜值评分模型训练脚本 v1.0
基于 SCUT-FBP5500_v2 数据集训练面部特征→评分回归模型，
并校准 beauty_core 评分管线参数。

数据集: 5500张人脸，每张有60人评分的均值(1-5)
目标:
  - SCUT 1-5 → 2-9 映射
  - 皮尔逊相关系数 ρ ≥ 0.3
  - 高分/低分稀有，中间分较多，方差尽量大
  - 低于2分判定缺陷，高于9分判定增值
"""
from __future__ import annotations
import os, sys, io, math, json, time, pickle, gc, random
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Any

# fix GBK encoding on Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── 抑制 OpenCV/tk 等 GUI 线程警告 ──
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**30)

import cv2
from PIL import Image
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, QuantileTransformer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize

# ═══════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'SCUT-FBP5500_v2', 'SCUT-FBP5500_v2')
LABELS_FILE = os.path.join(DATASET_DIR, 'train_test_files', 'All_labels.txt')
IMAGES_DIR = os.path.join(DATASET_DIR, 'Images')
TRAIN_TXT = os.path.join(DATASET_DIR, 'train_test_files', 'split_of_60%training and 40%testing', 'train.txt')
TEST_TXT = os.path.join(DATASET_DIR, 'train_test_files', 'split_of_60%training and 40%testing', 'test.txt')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'stats_output')
FEATURES_CACHE = os.path.join(OUTPUT_DIR, 'scut_features_full.pkl')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'beauty_model_full.pkl')
REPORT_PATH = os.path.join(OUTPUT_DIR, 'training_report_full.json')

SUBSET_SAMPLE = None  # None=全量训练, 设为 N 则只采样 N 张快速测试

# ── 评分映射参数 ──
SCUT_MIN, SCUT_MAX = 1.0, 5.0
OUR_MIN, OUR_MAX = 2.0, 9.0


# ═══════════════════════════════════════════
#  真实图像特征提取 (替代随机模拟)
# ═══════════════════════════════════════════

def extract_real_features(img: np.ndarray,
                          face_rect: tuple[int, int, int, int] | None) -> dict[str, float]:
    """从人脸图像中提取~25维真实视觉特征。

    不依赖深度模型，基于 OpenCV 经典算法。
    这些特征作为回归模型的输入。
    """
    h, w = img.shape[:2]
    feats: dict[str, float] = {}

    # ── 转到 BGR (OpenCV 惯例) ──
    if img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # ── 全局图像特征 ──
    feats['global_brightness'] = float(np.mean(gray)) / 255.0
    feats['global_contrast'] = float(np.std(gray)) / 128.0
    feats['global_sharpness'] = float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()))
    feats['global_entropy'] = float(_hist_entropy(gray))
    feats['global_saturation_mean'] = float(np.mean(hsv[:, :, 1])) / 255.0
    feats['global_saturation_std'] = float(np.std(hsv[:, :, 1])) / 128.0

    # 颜色矩
    for ch, name in enumerate(['B', 'G', 'R']):
        ch_data = bgr[:, :, ch].flatten()
        feats[f'color_{name}_mean'] = float(np.mean(ch_data)) / 255.0
        feats[f'color_{name}_std'] = float(np.std(ch_data)) / 128.0

    # 暖色调 vs 冷色调
    r_mean = float(np.mean(bgr[:, :, 2]))
    b_mean = float(np.mean(bgr[:, :, 0]))
    feats['warmth_ratio'] = (r_mean - b_mean) / max(r_mean + b_mean, 1)

    # ── 人脸区域特征 ──
    if face_rect:
        fx, fy, fw, fh = face_rect
        fx, fy = max(0, fx), max(0, fy)
        fw = min(fw, w - fx)
        fh = min(fh, h - fy)
        if fw > 10 and fh > 10:
            face_roi = bgr[fy:fy+fh, fx:fx+fw]
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            face_hsv = hsv[fy:fy+fh, fx:fx+fw]

            # 人脸框几何
            feats['face_area_ratio'] = (fw * fh) / max(w * h, 1)
            feats['face_aspect'] = fw / max(fh, 1)
            feats['face_x_ratio'] = fx / max(w, 1)
            feats['face_y_ratio'] = fy / max(h, 1)

            # 人脸区域亮度/对比度/锐度
            feats['face_brightness'] = float(np.mean(face_gray)) / 255.0
            feats['face_contrast'] = float(np.std(face_gray)) / 128.0
            feats['face_sharpness'] = float(np.log1p(cv2.Laplacian(face_gray, cv2.CV_64F).var()))
            feats['face_entropy'] = float(_hist_entropy(face_gray))
            feats['face_saturation'] = float(np.mean(face_hsv[:, :, 1])) / 255.0

            # 肤色相关
            for ch, name in enumerate(['B', 'G', 'R']):
                ch_data = face_roi[:, :, ch].flatten()
                feats[f'skin_{name}_mean'] = float(np.mean(ch_data)) / 255.0
                feats[f'skin_{name}_std'] = float(np.std(ch_data)) / 128.0

            # 对称性估计 (左右翻转差分)
            if fw > 20 and fh > 20:
                feats['face_symmetry'] = _estimate_symmetry(face_gray)

            # 边缘密度
            edges = cv2.Canny(face_gray, 50, 150)
            feats['face_edge_density'] = float(np.sum(edges > 0)) / max(edges.size, 1)

            # 上半/下半边缘比 (人脸结构特征)
            top_h = face_gray[:fh//2, :]
            bot_h = face_gray[fh//2:, :]
            te = cv2.Canny(top_h, 50, 150)
            be = cv2.Canny(bot_h, 50, 150)
            td = np.sum(te > 0) / max(te.size, 1)
            bd = np.sum(be > 0) / max(be.size, 1)
            feats['face_topbot_edge_ratio'] = td / max(bd, 0.001)

            # 局部纹理方差 (分块 Laplacian)
            feats['face_lap_block_var'] = _block_lap_variance(face_gray)
        else:
            _fill_default_face_feats(feats)
    else:
        _fill_default_face_feats(feats)

    return feats


def _hist_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = hist.flatten()
    hist = hist / max(hist.sum(), 1)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _estimate_symmetry(face_gray: np.ndarray) -> float:
    """通过左右翻转差分估算面部对称度 [0,1]"""
    h, w = face_gray.shape
    mid = w // 2
    left = face_gray[:, :mid]
    right = cv2.flip(face_gray[:, mid:mid*2], 1) if mid*2 <= w else cv2.flip(face_gray[:, mid-1:], 1)
    min_w = min(left.shape[1], right.shape[1])
    if min_w < 5:
        return 0.5
    diff = np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))
    sim = 1.0 - np.mean(diff) / 128.0
    return float(np.clip(sim, 0.1, 0.99))


def _block_lap_variance(face_gray: np.ndarray) -> float:
    """分块 Laplacian 方差 (检测局部纹理一致性)"""
    h, w = face_gray.shape
    block_h, block_w = max(h // 4, 10), max(w // 4, 10)
    vars_list = []
    for i in range(4):
        for j in range(4):
            y1, y2 = i * block_h, min((i+1)*block_h, h)
            x1, x2 = j * block_w, min((j+1)*block_w, w)
            if y2 > y1 and x2 > x1:
                blk = face_gray[y1:y2, x1:x2]
                vars_list.append(cv2.Laplacian(blk, cv2.CV_64F).var())
    if not vars_list:
        return 0.0
    return float(np.std(vars_list) / max(np.mean(vars_list), 1))


def _fill_default_face_feats(feats: dict[str, float]):
    for k in ['face_area_ratio', 'face_aspect', 'face_x_ratio', 'face_y_ratio',
              'face_brightness', 'face_contrast', 'face_sharpness', 'face_entropy',
              'face_saturation', 'face_symmetry', 'face_edge_density',
              'face_topbot_edge_ratio', 'face_lap_block_var']:
        if k not in feats:
            feats[k] = 0.0
    for ch in ['B', 'G', 'R']:
        for suffix in ['_mean', '_std']:
            k = f'skin_{ch}{suffix}'
            if k not in feats:
                feats[k] = 0.0


# ═══════════════════════════════════════════
#  Haar 人脸检测器 (复用 image_utils 逻辑的简化版)
# ═══════════════════════════════════════════

def _load_cascade(filename: str) -> cv2.CascadeClassifier | None:
    path = os.path.join(cv2.data.haarcascades, filename)
    if not os.path.exists(path):
        return None
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None
    return cascade


_FC = _load_cascade('haarcascade_frontalface_default.xml')
_PC = _load_cascade('haarcascade_profileface.xml')


def detect_face_roi(img: np.ndarray) -> tuple | None:
    """快速人脸检测, 返回 (x, y, w, h) 或 None"""
    h, w = img.shape[:2]
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    all_rects = []

    # 正面
    if _FC is not None:
        eq = cv2.equalizeHist(gray)
        for sf, mn in [(1.1, 5), (1.05, 3)]:
            try:
                faces = _FC.detectMultiScale(eq, scaleFactor=sf, minNeighbors=mn,
                                             minSize=(60, 60))
                all_rects.extend([(int(x), int(y), int(ww), int(hh))
                                  for (x, y, ww, hh) in faces])
            except Exception:
                pass

    # 侧脸
    if _PC is not None:
        try:
            eq2 = cv2.equalizeHist(gray)
            profiles = _PC.detectMultiScale(eq2, scaleFactor=1.1, minNeighbors=4,
                                            minSize=(60, 60))
            all_rects.extend([(int(x), int(y), int(ww), int(hh))
                              for (x, y, ww, hh) in profiles])
        except Exception:
            pass

    if not all_rects:
        return None

    # NMS
    all_rects.sort(key=lambda r: r[2]*r[3], reverse=True)
    kept = []
    for r in all_rects:
        overlap = False
        rx1, ry1, rw, rh = r
        rx2, ry2 = rx1+rw, ry1+rh
        ra = rw*rh
        for k in kept:
            kx1, ky1, kw, kh = k
            kx2, ky2 = kx1+kw, ky1+kh
            ix1, iy1 = max(rx1, kx1), max(ry1, ky1)
            ix2, iy2 = min(rx2, kx2), min(ry2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                ia = (ix2-ix1)*(iy2-iy1)
                ua = ra+kw*kh-ia
                if ia/ua > 0.3:
                    overlap = True
                    break
        if not overlap:
            kept.append(r)
    kept.sort(key=lambda r: r[2]*r[3], reverse=True)
    return kept[0] if kept else None


# ═══════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════

def load_labels(label_file: str) -> list[tuple[str, float]]:
    """解析标签文件, 返回 [(filename, score), ...]"""
    data = []
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    fname = parts[0]
                    score = float(parts[1])
                    data.append((fname, score))
                except ValueError:
                    continue
    return data


def extract_and_cache_features(label_file: str,
                                cache_path: str,
                                subset: int | None = None) -> list[dict[str, Any]]:
    """批量提取特征并缓存"""
    if os.path.exists(cache_path):
        print(f'[加载] 从缓存读取特征: {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    all_labels = load_labels(label_file)
    if subset:
        all_labels = all_labels[:subset]
        print(f'[采样] 仅使用前 {subset} 张')

    total = len(all_labels)
    results = []
    t0 = time.time()

    print(f'  开始提取 {total} 张图片特征...')
    for idx, (fname, scut_score) in enumerate(all_labels):
        img_path = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(img_path):
            continue

        try:
            pil_img = Image.open(img_path).convert('RGB')
            img = np.array(pil_img)
        except Exception:
            continue

        face_rect = detect_face_roi(img)
        feats = extract_real_features(img, face_rect)  # pyright: ignore[reportArgumentType]
        feats['filename'] = fname  # pyright: ignore[reportArgumentType]
        feats['scut_score'] = scut_score  # pyright: ignore[reportArgumentType]
        feats['has_face'] = face_rect is not None  # pyright: ignore[reportArgumentType]
        results.append(feats)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1)
            eta = (total - idx - 1) / max(rate, 0.001)
            print(f'  [{idx+1}/{total}] {rate:.1f} img/s | ETA {eta/60:.0f}min | '
                  f'face={feats["has_face"]} | SCUT={scut_score:.2f}', flush=True)

    elapsed = time.time() - t0
    print(f'\n[完成] 特征提取: {len(results)}/{total} 张, 耗时 {elapsed/60:.1f} min')

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'[缓存] 已保存到 {cache_path}')
    return results


# ═══════════════════════════════════════════
#  训练 & 校准
# ═══════════════════════════════════════════

def train_regression(features_data: list[dict]) -> dict:
    """训练 Ridge 回归模型, 评估相关性"""
    # 特征列 (排除非数值字段)
    exclude_keys = {'filename', 'scut_score', 'has_face'}
    feat_keys = sorted([k for k in features_data[0].keys() if k not in exclude_keys])

    X_raw = np.array([[d[k] for k in feat_keys] for d in features_data], dtype=np.float64)
    y_raw = np.array([d['scut_score'] for d in features_data], dtype=np.float64)

    # 去除 inf/nan
    mask = np.isfinite(X_raw).all(axis=1) & np.isfinite(y_raw)
    X = X_raw[mask]
    y = y_raw[mask]
    features_data_valid = [d for d, m in zip(features_data, mask) if m]

    print(f'\n[数据] 有效样本: {len(y)} | 特征维度: {X.shape[1]}')
    print(f'[SCUT 原始分布] min={y.min():.3f} max={y.max():.3f} '
          f'mean={y.mean():.3f} std={y.std():.3f}')

    # ── 数量分布 ──
    bins = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    hist, _ = np.histogram(y, bins=bins)
    print('[SCUT 评分分布]')
    for i in range(len(bins)-1):
        bar = '█' * int(hist[i] / max(hist) * 30)
        print(f'  [{bins[i]:.1f}~{bins[i+1]:.1f}] {hist[i]:5d} {bar}')

    # ── 划分 train/test (60/40) ──
    if os.path.exists(TRAIN_TXT) and os.path.exists(TEST_TXT):
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

        train_idx = [i for i, d in enumerate(features_data_valid) if d['filename'] in train_files]
        test_idx = [i for i, d in enumerate(features_data_valid) if d['filename'] in test_files]
        if train_idx and test_idx:
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            print(f'[Split] Train: {len(train_idx)}, Test: {len(test_idx)} (官方60/40)')
        else:
            raise RuntimeError('官方 split 文件解析失败')
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42)
        print(f'[Split] Train: {len(y_train)}, Test: {len(y_test)} (随机60/40)')

    # ── 特征标准化 ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── 多项式特征 (2阶, 交互项) ──
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)
    print(f'[特征] 原始 {X.shape[1]} → 多项式 {X_train_poly.shape[1]} 维')

    # ── RidgeCV 回归 ──
    alphas = np.logspace(-2, 3, 20)
    ridge = RidgeCV(alphas=alphas)
    ridge.fit(X_train_poly, y_train)
    print(f'[RidgeCV] 最优 alpha = {ridge.alpha_:.4f}')

    y_pred = ridge.predict(X_test_poly)
    y_pred = np.clip(y_pred, 0, 5)  # 限制在合理范围

    # ── 相关性评估 ──
    r_pearson, p_pearson = pearsonr(y_test, y_pred)
    r_spearman, p_spearman = spearmanr(y_test, y_pred)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    r2 = float(1.0 - float(np.sum((y_test - y_pred) ** 2)) / float(np.sum((y_test - np.mean(y_test)) ** 2)))

    print(f'\n{"="*60}')
    print(f'📊  回归评估结果 (Test Set, n={len(y_test)})')
    print(f'{"="*60}')
    print(f'  Pearson  r   = {r_pearson:+.4f}  (p={p_pearson:.2e})')
    print(f'  Spearman ρ   = {r_spearman:+.4f}  (p={p_spearman:.2e})')
    print(f'  MAE          = {mae:.4f}')
    print(f'  RMSE         = {rmse:.4f}')
    print(f'  R²           = {r2:.4f}')

    # ── 我们的 2-9 映射后的分布 ──
    our_pred = scut_to_our_scale(y_pred)
    our_true = scut_to_our_scale(np.asarray(y_test, dtype=np.float64))

    our_pearson_res = pearsonr(our_true, our_pred)
    r_our = float(our_pearson_res.statistic)  # pyright: ignore[reportAttributeAccessIssue]
    print(f'\n[映射 2-9] Pearson r(our) = {r_our:+.4f}')
    print(f'[映射 2-9] 预测分布: min={our_pred.min():.2f} max={our_pred.max():.2f} '
          f'mean={our_pred.mean():.2f} std={our_pred.std():.2f}')

    our_bins = np.linspace(2, 9, 15)
    our_hist, _ = np.histogram(our_pred, bins=our_bins)
    print('[映射 2-9 评分分布]')
    for i in range(len(our_bins)-1):
        bar = '█' * int(our_hist[i] / max(our_hist.max(), 1) * 30)
        print(f'  [{our_bins[i]:.1f}~{our_bins[i+1]:.1f}] {our_hist[i]:5d} {bar}')

    # 返回结果
    return {
        'n_train': len(y_train),
        'n_test': len(y_test),
        'n_features_raw': X.shape[1],
        'n_features_poly': X_train_poly.shape[1],
        'pearson_r': round(r_pearson, 4),
        'pearson_p': float(p_pearson),
        'spearman_rho': round(r_spearman, 4),
        'spearman_p': float(p_spearman),
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'r2': round(r2, 4),
        'pearson_r_our': round(r_our, 4),
        'ridge_alpha': float(ridge.alpha_) if ridge.alpha_ is not None else 0.0,
        'y_test_mean': float(np.mean(y_test)),
        'y_test_std': float(np.std(y_test)),
        'y_pred_mean': float(np.mean(y_pred)),
        'y_pred_std': float(np.std(y_pred)),
        'our_pred_mean': float(np.mean(our_pred)),
        'our_pred_std': float(np.std(our_pred)),
        'our_dist': {f'{our_bins[i]:.1f}': int(our_hist[i]) for i in range(len(our_bins)-1)},
        # 保存模型组件
        '_scaler': scaler,
        '_poly': poly,
        '_ridge': ridge,
        '_feat_keys': feat_keys,
    }


def scut_to_our_scale(scut_scores: np.ndarray) -> np.ndarray:
    """SCUT 1-5 → 我们的 2-9 非线性映射

    使用 sigmoid 平滑映射，保留两端稀有性:
    - SCUT 1.0-1.5 → 2.0-3.0 (缺陷区)
    - SCUT 2.5-3.5 → 5.0-6.5 (大众区)
    - SCUT 4.5-5.0 → 8.0-9.0 (增值区)
    """
    # 先用幂函数拉伸，再用 sigmoid 压缩两端
    # target = 2 + 7 * (scut - 1) / 4   # 线性映射
    # 使用 tanh 非线性增强两端差异
    x = (scut_scores - 3.0) / 1.5  # 中心化到 [-1.33, +1.33]
    stretched = np.tanh(x * 1.2)   # tanh 压缩
    # 映射到 [2, 9]
    result = 5.5 + 3.5 * stretched  # 中心 5.5, 幅度 ±3.5
    return np.clip(result, 2.0, 9.0)


def calibrate_scoring_params(train_results: dict[str, Any]) -> dict[str, Any]:
    """基于训练结果校准 beauty_core 评分管线参数

    目标: 让评分管线输出分布覆盖 [2, 9], 中心在 5.5, 方差够大。
    """
    print(f'\n{"="*60}')
    print(f'[Calibrate] 评分管线参数校准')
    print(f'{"="*60}')

    current = {
        'vmax': 5.5, 'k_half': 8.0, 'q_weight': 22.0,
        'det_cap': 15.0, 'calib_offset': -0.2, 'geo_weight': 3.2,
    }

    # 从训练数据驱动校准:
    # 我们映射后预测均值≈5.5, 需要 std 足够大
    our_pred_mean = train_results.get('our_pred_mean', 5.5)
    our_pred_std = train_results.get('our_pred_std', 1.0)

    # Hill 函数: base = vmax * x/(x+k_half) + offset
    # 目标: base(0)=2.0, base(inf)=9.0
    #   offset = 2.0
    #   vmax = 7.0  (因为 7+2=9)
    # 中点: base(k_half) = 7*0.5 + 2 = 5.5 ✓

    # x = min(|det|*2, det_cap) + quality * q_weight
    # 典型 quality∈[2,8], det∈[-5,+5]
    # 我们希望 x 中位数 ≈ k_half 以让评分分散在线性区

    # 品质主导时: x≈5*q_weight, 设 q_weight=3 → x≈15
    # k_half=15 → 评分刚好在线性区中心

    k_half = max(our_pred_mean * 3.0, 10.0)  # 动态调整
    std_boost = max(1.5 / max(our_pred_std, 0.1), 1.0)  # std 越小 boost 越大
    det_cap = min(10.0 * std_boost, 40.0)  # 放大 det 贡献

    calibrated = {
        'vmax': 7.0,                    # [2, 9] 范围需要 7+2=9
        'k_half': round(k_half, 1),     # 半饱和点: 与品质×权重匹配
        'q_weight': 5.0,                # 降低品质权重, 让 det 起作用
        'det_cap': round(det_cap, 1),   # det 贡献上限随 std 自适应
        'calib_offset': 2.0,            # 底线: 最低分 = 2.0
        'geo_weight': 1.5,              # 几何贡献温和, 避免随机主导
    }

    print(f'\n  数据驱动: pred_mean={our_pred_mean:.2f}, pred_std={our_pred_std:.2f}')
    print(f'  参数对比:')
    print(f'  {"参数":<16s} {"旧值":>8s} {"新值":>8s} {"变化":>8s}')
    print(f'  {"-"*42}')
    for k in current:
        c_old = current[k]
        c_new = calibrated[k]
        delta = c_new - c_old
        print(f'  {k:<16s} {c_old:>8.2f} {c_new:>8.2f} {delta:>+8.2f}')

    # 模拟评分范围
    print(f'\n  模拟评分 (q=品质, d=det):')
    header = '    ' + ''.join(f' d={d:+d}   ' for d in [-5, 0, 3, 6, 10])
    print(header)
    for q in [2.0, 3.5, 5.0, 6.5, 8.0]:
        row = f'  q={q:.1f}'
        for det in [-5, 0, 3, 6, 10]:
            dc = min(abs(det)*2.0, calibrated['det_cap'])
            x = dc + q * calibrated['q_weight']
            base = calibrated['vmax'] * x / (x + calibrated['k_half']) + calibrated['calib_offset']
            row += f' {base:5.2f}'
        print(row)

    return calibrated


# ═══════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'╔{"═"*58}╗')
    print(f'║  颜值评分模型训练 — 基于 SCUT-FBP5500_v2            ║')
    print(f'╚{"═"*58}╝')
    print(f'  数据集: {LABELS_FILE}')
    print(f'  图片目录: {IMAGES_DIR}')
    if SUBSET_SAMPLE:
        print(f'  ⚠ 快速模式: 仅 {SUBSET_SAMPLE} 张')
    print()

    # 1. 特征提取 & 缓存
    features_data = extract_and_cache_features(
        LABELS_FILE, FEATURES_CACHE, subset=SUBSET_SAMPLE)

    if not features_data:
        print('[错误] 未提取到任何特征')
        return

    # 2. 训练
    train_results = train_regression(features_data)

    # 3. 校准参数
    calibrated_params = calibrate_scoring_params(train_results)

    # 4. 保存模型
    model_data = {
        'scaler': train_results.pop('_scaler'),
        'poly': train_results.pop('_poly'),
        'ridge': train_results.pop('_ridge'),
        'feat_keys': train_results.pop('_feat_keys'),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f'\n[模型] 已保存到 {MODEL_PATH}')

    # 5. 生成报告
    report = {
        'version': 'v1.0',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'SCUT-FBP5500_v2',
        'n_total': len(features_data),
        'training': train_results,
        'calibrated_params': calibrated_params,
        'targets': {
            'pearson_r_min': 0.3,
            'our_scale': '2-9',
            'mapping': 'scut_to_our_scale (tanh nonlinear)',
            'below_2': '缺陷判定',
            'above_9': '增值判定',
        },
    }

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f'[报告] 已保存到 {REPORT_PATH}')

    # 6. 质量判定
    r = train_results['pearson_r_our']
    print(f'\n{"="*60}')
    if r >= 0.3:
        print(f'✅ 训练成功! Pearson r(2-9) = {r:+.4f} ≥ 0.3 ✓')
    else:
        print(f'⚠ 相关系数不足: Pearson r(2-9) = {r:+.4f} < 0.3')
        print(f'  建议: 增大特征维度或使用非线性模型')

    print(f'{"="*60}')


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    main()
