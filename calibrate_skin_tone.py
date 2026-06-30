"""
SCUT-FBP5500_v2 肤色校准脚本
基于 5500 张标注人脸，统计 Asian/Caucasian 各子集的 ITA° 分布，
输出校准后的肤色分类阈值，供 classify_skin_tone() 使用。

v51: 修正 ITA° 公式为标准 CIE Lab (L*:0-100, b*:-128~127)
     旧版 SCUT 校准数据(均值40.3/35.9等)基于错误公式, 需重新运行本脚本。
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import json
from collections import defaultdict


def compute_ita(bgr_arr: np.ndarray) -> tuple[float, float, float, float, int]:
    """计算单张人脸的 ITA° 和相关肤色指标
    
    Returns: (ita_deg, l_mean, b_mean, chroma_ratio, pixel_count)
    """
    h, w = bgr_arr.shape[:2]
    # 取中心 50% 区域避免边缘干扰
    cx, cy = w // 2, h // 2
    rw, rh = w // 2, h // 2
    x1, y1 = max(0, cx - rw // 2), max(0, cy - rh // 2)
    x2, y2 = min(w, x1 + rw), min(h, y1 + rh)
    roi = bgr_arr[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    
    # 亮度 Top-30%
    y_ch = ycrcb[:, :, 0].flatten()
    y_sorted = np.sort(y_ch)[::-1]
    top_n = max(len(y_sorted) // 3, 100)
    y_top = np.mean(y_sorted[:top_n])
    
    # Cr/Y 归一化色度比
    cr_ch = ycrcb[:, :, 1].flatten()
    cr_mean = np.mean(cr_ch)
    chroma_ratio = cr_mean / max(y_top, 1)
    
    # ITA° (v51: 标准 CIE Lab 公式, OpenCV Lab → CIE 转换)
    l_ch = lab[:, :, 0].astype(np.float64)
    b_ch = lab[:, :, 2].astype(np.float64)
    l_cie = float(np.mean(l_ch)) * 100.0 / 255.0   # OpenCV L(0-255) → CIE L*(0-100)
    b_cie = float(np.mean(b_ch)) - 128.0            # OpenCV b(0-255) → CIE b*(-128~127)
    ita = float(np.degrees(np.arctan2(l_cie - 50.0, b_cie)))
    l_mean, b_mean = l_cie, b_cie  # 返回 CIE 值
    
    return ita, l_mean, b_mean, chroma_ratio, roi.size


def detect_face_center(bgr_arr: np.ndarray) -> tuple[int, int, int, int] | None:
    """检测人脸，返回 (x, y, w, h)"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    gray = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        # 降阈值重试
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(40, 40))
    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        return (fx, fy, fw, fh)
    return None


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'SCUT-FBP5500_v2', 'SCUT-FBP5500_v2')
    images_dir = os.path.join(data_dir, 'Images')
    
    if not os.path.isdir(images_dir):
        print(f"[ERROR] 找不到图片目录: {images_dir}")
        return
    
    # 按子集分组统计
    subsets = {
        'Asian_Female':   {'prefix': 'AF', 'itas': [], 'chromas': []},
        'Asian_Male':     {'prefix': 'AM', 'itas': [], 'chromas': []},
        'Caucasian_Female': {'prefix': 'CF', 'itas': [], 'chromas': []},
        'Caucasian_Male': {'prefix': 'CM', 'itas': [], 'chromas': []},
    }
    
    all_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
    total = len(all_files)
    
    print(f"开始校准肤色判定 (共 {total} 张图片)...")
    
    for i, fname in enumerate(all_files):
        fpath = os.path.join(images_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        
        # 确定子集
        subset = None
        for name, info in subsets.items():
            if fname.startswith(info['prefix']):
                subset = name
                break
        if subset is None:
            continue
        
        # 检测人脸
        face_rect = detect_face_center(img)
        if face_rect is None:
            continue
        
        fx, fy, fw, fh = face_rect
        # 取人脸区域 (稍微内缩避免边缘)
        margin = int(fw * 0.1)
        x1 = max(0, fx + margin)
        y1 = max(0, fy + margin)
        x2 = min(img.shape[1], fx + fw - margin)
        y2 = min(img.shape[0], fy + fh - margin)
        face_roi = img[y1:y2, x1:x2]
        
        if face_roi.size < 1000:
            continue
        
        ita, l_mean, b_mean, chroma, _ = compute_ita(face_roi)
        subsets[subset]['itas'].append(ita)
        subsets[subset]['chromas'].append(chroma)
        
        if (i + 1) % 500 == 0:
            print(f"  进度: {i + 1}/{total}")
    
    print("\n" + "=" * 70)
    print("SCUT-FBP5500_v2 肤色 ITA° 统计 (皮肤科标准)")
    print("=" * 70)
    
    results = {}
    for name in ['Asian_Female', 'Asian_Male', 'Caucasian_Female', 'Caucasian_Male']:
        info = subsets[name]
        itas = np.array(info['itas'])
        chromas = np.array(info['chromas'])
        
        if len(itas) < 10:
            print(f"\n{name}: 样本不足 ({len(itas)})")
            continue
        
        p5, p25, p50, p75, p95 = np.percentile(itas, [5, 25, 50, 75, 95])
        p5c, p50c, p95c = np.percentile(chromas, [5, 50, 95])
        
        results[name] = {
            'count': int(len(itas)),
            'ita_mean': float(np.mean(itas)),
            'ita_std': float(np.std(itas)),
            'ita_p5': float(p5),
            'ita_p25': float(p25),
            'ita_p50': float(p50),
            'ita_p75': float(p75),
            'ita_p95': float(p95),
            'chroma_p50': float(p50c),
        }
        
        print(f"\n  {name} (n={len(itas)})")
        print(f"    ITA°: 均值={np.mean(itas):.1f} ± {np.std(itas):.1f}")
        print(f"    分位: P5={p5:.1f}, P25={p25:.1f}, P50={p50:.1f}, P75={p75:.1f}, P95={p95:.1f}")
        print(f"    Cr/Y: P50={p50c:.3f}")
    
    # ── 合并 Asian vs Caucasian ──
    asian_itas = np.concatenate([np.array(subsets['Asian_Female']['itas']),
                                  np.array(subsets['Asian_Male']['itas'])])
    caucasian_itas = np.concatenate([np.array(subsets['Caucasian_Female']['itas']),
                                      np.array(subsets['Caucasian_Male']['itas'])])
    
    print("\n" + "=" * 70)
    print("合并对比: Asian vs Caucasian")
    print("=" * 70)
    for label, arr in [('Asian', asian_itas), ('Caucasian', caucasian_itas)]:
        p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
        print(f"\n  {label} (n={len(arr)})")
        print(f"    均值={np.mean(arr):.1f}, 中位={p50:.1f}")
        print(f"    P5={p5:.1f}, P25={p25:.1f}, P75={p75:.1f}, P95={p95:.1f}")
    
    # ── v51: 推荐阈值 (基于标准 CIE ITA° 公式) ──
    # 临床皮肤科 ITA 标准: >55非常浅 / 41-55浅 / 28-41中 / 10-28褐 / <-30深
    # 东亚人群微调: L* 偏低(黄调吸光), 浅肤色门槛从55→48
    cauc_p25 = float(np.percentile(caucasian_itas, 25))
    asian_p75 = float(np.percentile(asian_itas, 75))
    asian_p25 = float(np.percentile(asian_itas, 25))
    
    # v51 推荐阈值 (标准CIE ITA公式)
    scut_light = max(cauc_p25, 48)   # 参考临床标准"浅肤色"41, 上调至48补偿东亚L*偏低
    scut_dark = max(min(asian_p25, 20), 10)  # 参考临床标准"褐色"10-28
    
    print("\n" + "=" * 70)
    print("v51 SCUT 校准推荐阈值 (标准 CIE ITA° 公式)")
    print("=" * 70)
    print(f"  ITA° > {scut_light:.0f}  → 浅肤色 (白皙/红润)")
    print(f"  {scut_dark:.0f} < ITA° ≤ {scut_light:.0f} → 中间肤色 (自然)")
    print(f"  ITA° ≤ {scut_dark:.0f}  → 深肤色 (小麦/橄榄)")
    print()
    print(f"  临床皮肤科标准(参考): >55非常浅 / 41-55浅 / 28-41中 / 10-28褐 / <-30深")
    print(f"  注: 东亚人群 L* 偏低(黄调), 浅肤色门槛下调至48")
    
    # ── 保存结果 ──
    output = {
        'source': 'SCUT-FBP5500_v2',
        'total_images': total,
        'subsets': results,
        'calibrated_thresholds': {
            'ita_light_min': round(scut_light, 1),
            'ita_dark_max': round(scut_dark, 1),
            'ita_clinical_light': 55,
            'ita_clinical_mid': 28,
            'ita_clinical_dark': -30,
            'note': 'SCUT阈值基于Asian/Caucasian分布，临床阈值基于皮肤科ITA标准',
        },
        'recommendation': '使用混合阈值: SCUT校准主要阈值 + 临床ITA作为边界兜底',
    }
    
    out_path = os.path.join(os.path.dirname(__file__), 'stats_output', 'skin_tone_calibration.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {out_path}")


if __name__ == '__main__':
    main()
