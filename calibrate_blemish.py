"""
瑕疵检测器校准 v49 -> v50
================================================================
数据集:  SCUT-FBP5500_v2 (洁净脸, 5500张) + ISIC (病变色斑, 50张)
策略:   SCUT 分布 = 假阳性基线, ISIC = 真阳性灵敏度
输出:   校准后的阈值 → 更新 image_utils.py
"""

from __future__ import annotations
import os, sys, json, time, cv2, numpy as np
from collections import defaultdict
from typing import Any

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

SCUT_DIR = os.path.join(BASE, 'SCUT-FBP5500_v2', 'SCUT-FBP5500_v2')
IMG_DIR = os.path.join(SCUT_DIR, 'Images')
LABELS_FILE = os.path.join(SCUT_DIR, 'train_test_files', 'All_labels.txt')
ISIC_DIR = os.path.join(BASE, 'ISIC')
STATS_DIR = os.path.join(BASE, 'stats_output')
os.makedirs(STATS_DIR, exist_ok=True)

from image_utils import load_and_normalize_image, detect_face


# ═══════════════════════════════════════════════════
# 1. 比例人脸掩码 (排除眼/鼻/唇/眉区域)
# ═══════════════════════════════════════════════════
def create_proportional_skin_mask(h: int, w: int) -> np.ndarray:
    """基于面部比例创建皮肤掩码, True=皮肤(可检测), False=排除
    
    排除区域 (人脸比例):
    - 眼部: y=[0.20, 0.45], x=[0.15, 0.85]  (双眼及眼周)
    - 鼻部: y=[0.40, 0.65], x=[0.30, 0.70]  (鼻梁鼻翼)
    - 唇部: y=[0.65, 0.90], x=[0.20, 0.80]  (嘴唇及周围)
    - 眉部: y=[0.12, 0.28], x=[0.12, 0.88]  (眉毛区域)
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # 合并所有排除区域
    exclude = np.zeros((h, w), dtype=np.uint8)
    
    # 眼部 (双眼 + 眉骨)
    cv2.rectangle(exclude, (int(w*0.12), int(h*0.12)), (int(w*0.88), int(h*0.45)), 255, -1)
    # 鼻部
    cv2.rectangle(exclude, (int(w*0.30), int(h*0.40)), (int(w*0.70), int(h*0.65)), 255, -1)
    # 唇部
    cv2.rectangle(exclude, (int(w*0.20), int(h*0.65)), (int(w*0.80), int(h*0.92)), 255, -1)
    
    # 柔和边缘 (高斯模糊 → 二值化)
    exclude = cv2.GaussianBlur(exclude, (9, 9), 3)
    exclude = (exclude > 30).astype(np.uint8) * 255
    
    mask = mask - exclude
    mask = np.clip(mask, 0, 255)
    
    return mask


# ═══════════════════════════════════════════════════
# 2. 统计自适应瑕疵检测器
# ═══════════════════════════════════════════════════
def detect_blemishes_calibrated(face_bgr: np.ndarray,
                                 params: dict | None = None) -> dict[str, Any]:
    """校准版瑕疵检测, 使用统计自适应阈值
    
    Args:
        face_bgr: 人脸BGR图像
        params: 校准参数字典 (None=默认值)
    """
    if params is None:
        params = {}
    
    h, w = face_bgr.shape[:2]
    
    try:
        skin_mask = create_proportional_skin_mask(h, w)
        skin_area = np.count_nonzero(skin_mask)
        if skin_area < 0.1 * h * w:
            return {'blemish_score': 0.0, 'acne_count': 0, 'spot_count': 0,
                    'details': '皮肤区域不足', 'mask': None}
        
        # ── 获取此人脸的肤色统计 ──
        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab)
        
        # 只在皮肤区域计算肤色统计
        skin_pixels_h = hsv[:, :, 0][skin_mask > 0].flatten()
        skin_pixels_s = hsv[:, :, 1][skin_mask > 0].flatten()
        skin_pixels_v = hsv[:, :, 2][skin_mask > 0].flatten()
        skin_l = lab[:, :, 0][skin_mask > 0].flatten()
        
        if len(skin_pixels_h) < 100:
            return {'blemish_score': 0.0, 'acne_count': 0, 'spot_count': 0,
                    'details': '皮肤像素不足', 'mask': None}
        
        # 人脸肤色统计
        h_median = np.median(skin_pixels_h)
        s_median = np.median(skin_pixels_s)
        s_mad = np.median(np.abs(skin_pixels_s - s_median))  # Median Absolute Deviation
        v_median = np.median(skin_pixels_v)
        l_median = np.median(skin_l)
        l_mad = np.median(np.abs(skin_l - l_median))
        
        # ── 动态阈值 (基于人脸自身肤色统计) ──
        # 痘印: S通道偏高 (发炎红>正常肤色)
        acne_s_thresh = s_median + max(2.5 * max(s_mad, 3), 20)
        
        # 色斑: L通道偏低 (暗斑<正常肤色)
        spot_l_thresh = l_median - max(2.5 * max(l_mad, 3), 15)
        
        # ── 红色痘印检测 ──
        lower_red1 = np.array([0, int(acne_s_thresh), 50], dtype=np.uint8)
        upper_red1 = np.array([15, 255, 230], dtype=np.uint8)
        lower_red2 = np.array([160, int(acne_s_thresh), 50], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 230], dtype=np.uint8)
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_red = cv2.bitwise_and(mask_red, skin_mask)
        
        # 形态学
        k_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k_s)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, k_m)
        
        # 连通组件 (面积 + 圆形度过滤)
        nl, lb, st, _ = cv2.connectedComponentsWithStats(mask_red, connectivity=8)
        acne_count = 0
        for i in range(1, nl):
            area = st[i, cv2.CC_STAT_AREA]
            # 过滤: 太小(噪点) or 太大(非痘印)
            if area < 8 or area > 350:
                continue
            # 圆形度检查
            x, y, bw, bh = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP], \
                          st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            perimeter = cv2.arcLength(np.array([[[x, y]], [[x+bw, y]], [[x+bw, y+bh]], [[x, y+bh]]], 
                                               dtype=np.float32), True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            # 痘印通常较圆 (circularity > 0.3)
            if circularity > 0.25:
                acne_count += 1
        
        # ── 暗色色斑检测 ──
        _, mask_dark = cv2.threshold(lab[:, :, 0], int(spot_l_thresh), 255, cv2.THRESH_BINARY_INV)
        mask_dark = cv2.bitwise_and(mask_dark, skin_mask)
        mask_dark = cv2.bitwise_and(mask_dark, cv2.bitwise_not(mask_red))
        
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_OPEN, k_s)
        mask_dark = cv2.morphologyEx(mask_dark, cv2.MORPH_CLOSE, k_m)
        
        nl_d, lb_d, st_d, _ = cv2.connectedComponentsWithStats(mask_dark, connectivity=8)
        spot_count = 0
        for i in range(1, nl_d):
            area = st_d[i, cv2.CC_STAT_AREA]
            if area < 6 or area > 300:
                continue
            x, y, bw, bh = st_d[i, cv2.CC_STAT_LEFT], st_d[i, cv2.CC_STAT_TOP], \
                          st_d[i, cv2.CC_STAT_WIDTH], st_d[i, cv2.CC_STAT_HEIGHT]
            perimeter = cv2.arcLength(np.array([[[x, y]], [[x+bw, y]], [[x+bw, y+bh]], [[x, y+bh]]], 
                                               dtype=np.float32), True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
            if circularity > 0.2:
                spot_count += 1
        
        # ── 评分 (面积归一化) ──
        area_norm = max(skin_area / 30000.0, 0.3)  # ~173x173 基准
        acne_norm = acne_count / area_norm
        spot_norm = spot_count / area_norm
        blemish_raw = np.sqrt(acne_norm * 2.0 + spot_norm * 1.0)
        blemish_score = round(min(blemish_raw, 10.0), 1)
        
        # stats for calibration
        stats_extra = {
            'h_median': float(h_median), 's_median': float(s_median),
            'acne_s_thresh': float(acne_s_thresh), 'spot_l_thresh': float(spot_l_thresh),
            'skin_area': int(skin_area),
        }
        
        return {
            'blemish_score': blemish_score,
            'acne_count': acne_count,
            'spot_count': spot_count,
            'details': f'痘:{acne_count} 斑:{spot_count}',
            'mask': None,
            'stats': stats_extra,
        }
    except Exception as e:
        return {'blemish_score': 0.0, 'acne_count': 0, 'spot_count': 0,
                'details': f'err:{e}', 'mask': None, 'stats': {}}


# ═══════════════════════════════════════════════════
# 3. 批量校准
# ═══════════════════════════════════════════════════
def run_calibration(max_scut: int = 5500):
    """在 SCUT + ISIC 上运行校准版检测器"""
    
    # ── 加载 SCUT 标签 ──
    labels = {}
    with open(LABELS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    labels[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    print(f"SCUT labels: {len(labels)}")

    # ── SCUT 洁净脸测试 ──
    scut_results = []
    scut_files = sorted(labels.keys())
    if max_scut < len(scut_files):
        import random
        random.seed(42)
        random.shuffle(scut_files)
        scut_files = scut_files[:max_scut]
    
    print(f"Testing {len(scut_files)} SCUT faces...")
    t0 = time.time()
    
    for idx, fname in enumerate(scut_files):
        path = os.path.join(IMG_DIR, fname)
        if not os.path.exists(path):
            continue
        
        if idx % 500 == 0 and idx > 0:
            elapsed = time.time() - t0
            eta = elapsed / idx * (len(scut_files) - idx)
            print(f"  {idx}/{len(scut_files)} ({idx*100//len(scut_files)}%) "
                  f"elapsed:{elapsed:.0f}s eta:{eta:.0f}s")
        
        try:
            img = load_and_normalize_image(path, source_type='path')
            faces = detect_face(img)
            if not faces:
                continue
            
            fx, fy, fw, fh = faces[0]
            face_bgr = cv2.cvtColor(img[fy:fy+fh, fx:fx+fw], cv2.COLOR_RGB2BGR)
            
            result = detect_blemishes_calibrated(face_bgr)
            result['rating'] = labels[fname]
            result['file'] = fname
            scut_results.append(result)
        except Exception as e:
            pass
    
    elapsed = time.time() - t0
    print(f"  Done: {len(scut_results)} faces in {elapsed:.0f}s")
    
    # ── ISIC 病变测试 ──
    isic_results = []
    if os.path.exists(ISIC_DIR):
        isic_files = [f for f in os.listdir(ISIC_DIR) if f.endswith('.jpg')]
        print(f"\nTesting {len(isic_files)} ISIC lesion images...")
        
        for fname in isic_files:
            path = os.path.join(ISIC_DIR, fname)
            try:
                img = load_and_normalize_image(path, source_type='path')
                h, w = img.shape[:2]
                # ISIC 多为皮肤镜特写, 整图即为病变区
                face_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                result = detect_blemishes_calibrated(face_bgr)
                result['file'] = fname
                result['rating'] = -1  # ISIC无评
                isic_results.append(result)
            except Exception as e:
                pass
        print(f"  Done: {len(isic_results)} ISIC images")
    
    # ── 分析 ──
    _analyze(scut_results, isic_results)
    
    return scut_results, isic_results


def _analyze(scut: list, isic: list):
    scores_scut = np.array([r['blemish_score'] for r in scut])
    acne_scut = np.array([r['acne_count'] for r in scut])
    spot_scut = np.array([r['spot_count'] for r in scut])
    
    ratings = np.array([r['rating'] for r in scut])
    
    scores_isic = np.array([r['blemish_score'] for r in isic]) if isic else np.array([])
    acne_isic = np.array([r['acne_count'] for r in isic]) if isic else np.array([])
    spot_isic = np.array([r['spot_count'] for r in isic]) if isic else np.array([])
    
    print("\n" + "=" * 65)
    print("  校准分析报告 (自适应肤色统计版)")
    print("=" * 65)
    
    # SCUT 整体分布
    print(f"\n── SCUT 洁净脸 (N={len(scores_scut)}) ──")
    print(f"  blemish:  mean={np.mean(scores_scut):.2f}  "
          f"med={np.median(scores_scut):.2f}  std={np.std(scores_scut):.2f}")
    print(f"  acne:     mean={np.mean(acne_scut):.2f}  "
          f"med={np.median(acne_scut):.2f}  max={np.max(acne_scut):.0f}")
    print(f"  spots:    mean={np.mean(spot_scut):.2f}  "
          f"med={np.median(spot_scut):.2f}  max={np.max(spot_scut):.0f}")
    print(f"  percentile: P50={np.percentile(scores_scut,50):.2f}  "
          f"P75={np.percentile(scores_scut,75):.2f}  "
          f"P90={np.percentile(scores_scut,90):.2f}  "
          f"P95={np.percentile(scores_scut,95):.2f}  "
          f"P99={np.percentile(scores_scut,99):.2f}")
    
    # 各评分段的分布
    over_2 = np.mean(scores_scut > 2.0) * 100
    over_4 = np.mean(scores_scut > 4.0) * 100
    over_6 = np.mean(scores_scut > 6.0) * 100
    print(f"  >2.0: {over_2:.1f}%  >4.0: {over_4:.1f}%  >6.0: {over_6:.1f}%")
    
    # ISIC 分布
    if len(scores_isic) > 0:
        print(f"\n── ISIC 色斑/病变 (N={len(scores_isic)}) ──")
        print(f"  blemish:  mean={np.mean(scores_isic):.2f}  "
              f"med={np.median(scores_isic):.2f}  std={np.std(scores_isic):.2f}")
        print(f"  acne:     mean={np.mean(acne_isic):.2f}")
        print(f"  spots:    mean={np.mean(spot_isic):.2f}")
        sep = np.mean(scores_isic) - np.mean(scores_scut)
        print(f"  分离度 (ISIC_mean - SCUT_mean): {sep:.2f}")
    
    # 按颜值分桶
    if len(ratings) > 0:
        print(f"\n── 按颜值等级分桶 ──")
        buckets = defaultdict(list)
        for s, r in zip(scores_scut, ratings):
            bk = f"{round(r * 2) / 2:.1f}"
            buckets[bk].append(s)
        
        print(f"  {'Rating':>7s}  {'N':>5s}  {'Mean':>6s}  {'Med':>6s}  {'P90':>6s}  {'>2%':>6s}  {'>4%':>6s}")
        for bk in sorted(buckets.keys(), key=float):
            vals = np.array(buckets[bk])
            print(f"  {bk:>7s}  {len(vals):>5d}  {np.mean(vals):>6.2f}  "
                  f"{np.median(vals):>6.2f}  {np.percentile(vals,90):>6.2f}  "
                  f"{np.mean(vals>2)*100:>5.1f}%  {np.mean(vals>4)*100:>5.1f}%")
    
    # 校准参数建议
    print(f"\n── 参考阈值建议 ──")
    print(f"  SCUT P90={np.percentile(scores_scut,90):.2f}  "
          f"P95={np.percentile(scores_scut,95):.2f}  "
          f"P99={np.percentile(scores_scut,99):.2f}")
    print(f"  ISIC mean={np.mean(scores_isic) if len(scores_isic)>0 else 0:.2f}")
    
    # 保存
    save_path = os.path.join(STATS_DIR, 'blemish_calibration_v50.json')
    output = {
        'version': 'v50',
        'scut_n': len(scores_scut),
        'isic_n': len(scores_isic),
        'scut_stats': {
            'blemish': {k: round(float(v), 3) for k, v in {
                'mean': np.mean(scores_scut), 'median': np.median(scores_scut),
                'std': np.std(scores_scut), 'p90': np.percentile(scores_scut, 90),
                'p95': np.percentile(scores_scut, 95), 'p99': np.percentile(scores_scut, 99),
                'over_2_pct': over_2, 'over_4_pct': over_4,
            }.items()},
            'acne': {'mean': np.mean(acne_scut), 'median': np.median(acne_scut)},
            'spots': {'mean': np.mean(spot_scut), 'median': np.median(spot_scut)},
        },
        'isic_stats': {
            'blemish': {k: round(float(v), 3) for k, v in {
                'mean': np.mean(scores_isic), 'median': np.median(scores_isic),
                'std': np.std(scores_isic),
            }.items()},
        } if len(scores_isic) > 0 else {},
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  参数已保存: {save_path}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max', type=int, default=2000, help='SCUT最大数量 (default: 2000)')
    args = ap.parse_args()
    run_calibration(max_scut=args.max)
