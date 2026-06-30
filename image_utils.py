"""
图片处理工具模块 v39
支持: 纹理预检、人脸检测（多策略Haar+软检测）、图片缩放、格式转换、批量分析
v39: MediaPipe FaceMesh 关键点提取 — 解剖学面宽比计算
"""
from __future__ import annotations
import cv2
import numpy as np
import os
import sys
from typing import Any
from PIL import Image, ImageOps
import io


# ═══════════════════════════════════════════════
#  级联分类器懒加载 (v38: PyInstaller _MEIPASS 兼容)
# ═══════════════════════════════════════════════

_NULL_CASCADE = cv2.CascadeClassifier()
SENTINEL_EMPTY = object()
_face_cascade: cv2.CascadeClassifier | None = None
_profile_cascade: cv2.CascadeClassifier | None = None
_eye_cascade: cv2.CascadeClassifier | None = None
_nose_cascade: cv2.CascadeClassifier | None = None


def _resolve_cascade_path(filename: str) -> str:
    """解析级联文件路径，兼容开发环境和 PyInstaller 打包后的 exe

    优先级: 1) cv2自带 2) EXE内置cascades/ 3) 本地项目cascades/ 4) 回退
    """
    # 1. 优先使用 OpenCV 自带的级联文件 (系统已安装 opencv 时最可靠)
    opencv_path = os.path.join(cv2.data.haarcascades, filename)  # pyright: ignore[reportAttributeAccessIssue]
    if os.path.exists(opencv_path):
        return opencv_path

    # 2. PyInstaller 打包后内置路径 (_MEIPASS/cascades/)
    meipass = getattr(sys, '_MEIPASS', '')
    if meipass:
        path = os.path.join(meipass, 'cascades', filename)
        if os.path.exists(path):
            return path
        path = os.path.join(meipass, filename)
        if os.path.exists(path):
            return path

    # 3. 开发环境项目内 cascades/ 目录 (v52.1: 与EXE共享同一套级联)
    local_path = os.path.join(os.path.dirname(__file__), 'cascades', filename)
    if os.path.exists(local_path):
        return local_path
    # 也尝试项目根目录
    project_cascades = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cascades', filename)
    if os.path.exists(project_cascades):
        return project_cascades

    # 4. 最后回退 (cv2路径, 即使不存在也返回供调用方处理)
    return opencv_path


def _safe_load_cascade(filename: str, cascade_name: str = "") -> cv2.CascadeClassifier:
    """安全加载级联分类器: 逐路径尝试, 检查 .empty(), 失败返回 _NULL_CASCADE"""
    try:
        path = _resolve_cascade_path(filename)
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            print(f'[WARN] Cascade {cascade_name or filename} load failed: all paths invalid, fallback to NULL_CASCADE')
            return _NULL_CASCADE
        return cascade
    except Exception:
        return _NULL_CASCADE


def _get_face_cascade() -> cv2.CascadeClassifier:
    """懒加载正面人脸 Haar 级联分类器"""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = _safe_load_cascade('haarcascade_frontalface_default.xml', '人脸正面')
    return _face_cascade


def _get_profile_cascade() -> cv2.CascadeClassifier:
    """懒加载侧脸 Haar 级联分类器"""
    global _profile_cascade
    if _profile_cascade is None:
        _profile_cascade = _safe_load_cascade('haarcascade_profileface.xml')
    return _profile_cascade


def _get_eye_cascade() -> cv2.CascadeClassifier:
    """懒加载眼睛 Haar 级联分类器（用于人脸二次验证）"""
    global _eye_cascade
    if _eye_cascade is None:
        _eye_cascade = _safe_load_cascade('haarcascade_eye.xml')
    return _eye_cascade


def _get_nose_cascade() -> cv2.CascadeClassifier:
    """懒加载鼻子 Haar 级联分类器 (v28.1: 安全加载, 文件缺失时 graceful fallback)"""
    global _nose_cascade
    if _nose_cascade is None:
        _nose_cascade = _safe_load_cascade('haarcascade_mcs_nose.xml')
    return _nose_cascade


# ═══════════════════════════════════════════════
#  图片加载与预处理 (v38: 格式白名单 + EXIF旋转 + 尺寸验证)
# ═══════════════════════════════════════════════

def load_and_normalize_image(source: str | bytes, source_type: str = "path") -> np.ndarray:
    """统一图片加载与归一化 — 所有格式进入后输出等价像素域.

    Args:
        source: 文件路径 (source_type="path") 或 bytes (source_type="bytes")
        source_type: "path" 或 "bytes"

    白名单: PNG, BMP, TIFF, JPG, JPEG, WebP, HEIC
    拒绝: GIF, SVG, RAW, ICO, <500px
    """
    if source_type == "path":
        fname = str(source)
        ext = os.path.splitext(fname)[1].lower()

        # GIF 拒绝
        if ext == '.gif':
            raise ValueError('GIF 格式因调色板量化会导致分析严重失真，请转换为 PNG 或 JPG')

        # 不支持的格式
        allowed = {'.png', '.bmp', '.tiff', '.tif', '.jpg', '.jpeg', '.webp', '.heic'}
        if ext not in allowed:
            raise ValueError(f'不支持的图片格式: {ext}')

        try:
            pil_img = Image.open(fname)
        except Exception as e:
            raise ValueError(f'无法打开图片: {e}')
    elif source_type == "bytes":
        try:
            pil_img = Image.open(io.BytesIO(source))  # pyright: ignore[reportArgumentType]
        except Exception as e:
            raise ValueError(f'无法解析图片数据: {e}')
    else:
        raise ValueError(f'未知 source_type: {source_type}')

    # EXIF 自动旋转
    pil_img = ImageOps.exif_transpose(pil_img)

    # 尺寸检查
    w, h = pil_img.size
    if w < 200 or h < 200:
        raise ValueError(f'图片尺寸过小 ({w}x{h})，最小要求 200x200')

    # 转 RGB + numpy
    if pil_img.mode == 'RGBA':
        pil_img = pil_img.convert('RGB')

    return np.array(pil_img)


def load_image(image_data: bytes) -> np.ndarray:
    """从字节数据加载图片为 RGB numpy 数组"""
    return load_and_normalize_image(image_data, source_type="bytes")


def load_image_from_path(path: str) -> np.ndarray:
    """从文件路径加载图片"""
    return load_and_normalize_image(path, source_type="path")


def resize_for_analysis(
    img: np.ndarray,
    max_mp: float = 20.0,
    quick_mode: bool = False
) -> np.ndarray:
    """缩放图片到适合分析的分辨率"""
    h, w = img.shape[:2]
    current_mp = (h * w) / 1_000_000

    target_mp = 2.0 if quick_mode else max_mp

    if current_mp <= target_mp:
        return img

    scale = (target_mp / current_mp) ** 0.5
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ═══════════════════════════════════════════════
#  人脸检测 (v38: 多策略 Haar + 多层验证）
# ═══════════════════════════════════════════════

def _nms_face_rects(rects: list[tuple[int, int, int, int]], iou_thresh: float = 0.3) -> list[tuple[int, int, int, int]]:
    """Non-Maximum Suppression 去重合并.
    对 Haar 级联产生的重叠检测框按面积排序，IoU > iou_thresh 的保留最大者。
    """
    if not rects:
        return []

    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept = []

    for r in rects:
        overlap = False
        rx1, ry1, rw, rh = r
        rx2, ry2 = rx1 + rw, ry1 + rh
        r_area = rw * rh

        for k in kept:
            kx1, ky1, kw, kh = k
            kx2, ky2 = kx1 + kw, ky1 + kh

            inter_x1 = max(rx1, kx1)
            inter_y1 = max(ry1, ky1)
            inter_x2 = min(rx2, kx2)
            inter_y2 = min(ry2, ky2)

            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union_area = r_area + kw * kh - inter_area
                if inter_area / union_area > iou_thresh:
                    overlap = True
                    break

        if not overlap:
            kept.append(r)

    return kept


def _check_face_aspect_ratio(w: int, h: int, ar_min: float = 0.55, ar_max: float = 1.90) -> bool:
    """人脸宽高比验证。正常人脸：宽/高 ≈ 0.55~1.90（覆盖正面→侧脸→仰头/低头）"""
    if h <= 0:
        return False
    ar = w / h
    return ar_min <= ar <= ar_max


def _check_face_edge_structure(face_roi_gray: np.ndarray) -> bool:
    """人脸结构性边缘分布验证（v25）。
    真实人脸的上半区（眼睛、眉毛、发际线）边缘密度远高于下半区。
    """
    h, w = face_roi_gray.shape
    if h < 30 or w < 30:
        return False

    edges = cv2.Canny(face_roi_gray, 50, 150)
    top_half = edges[:h // 2, :]
    bot_half = edges[h // 2:, :]

    top_density = np.sum(top_half > 0) / max(top_half.size, 1)
    bot_density = np.sum(bot_half > 0) / max(bot_half.size, 1)

    if bot_density < 0.001:
        return True
    return bool(top_density / bot_density > 1.15)


def _check_face_texture_variance(face_roi_gray: np.ndarray, min_var: float = 30.0) -> bool:
    """纹理方差验证（v25）。Laplacian 方差对纹理变化敏感。"""
    lap_var = cv2.Laplacian(face_roi_gray, cv2.CV_64F).var()
    return bool(lap_var >= min_var)


def _check_face_features_present(face_roi_gray: np.ndarray, nose_cascade: cv2.CascadeClassifier | None = None) -> tuple[int, int]:
    """检测人脸 ROI 内是否包含眼睛和鼻子等面部特征（v25 + v38 minSize上限）。

    v38: minSize 加绝对值上限, 防止大脸时眼睛/鼻子检测参数爆炸。
    Returns: (眼睛数, 鼻子数)
    """
    h, w = face_roi_gray.shape
    eye_cascade = _get_eye_cascade()
    nose_cascade = nose_cascade or _get_nose_cascade()

    # 眼睛检测 - v38 minSize 上限
    eye_min = max(10, min(w // 8, 80))
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(eye_min, eye_min),
        maxSize=(w // 2, h // 3),
    )
    n_eyes = len(eyes)

    # 鼻子检测
    n_nose = 0
    if nose_cascade is not None and not (hasattr(nose_cascade, 'empty') and nose_cascade.empty()):
        try:
            noses = nose_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(20, 20),
            )
            n_nose = len(noses)
        except Exception:
            pass

    return n_eyes, n_nose


def _is_skin_region(bgr_arr: np.ndarray, min_ratio: float = 0.30) -> bool:
    """判断一个 ROI 区域是否包含足够比例的肤色像素。
    使用 HSV + YCrCb 双空间肤色检测，覆盖全人种肤色。
    """
    hsv = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2YCrCb)

    # 浅肤色
    lower_hsv_light = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv_light = np.array([25, 170, 255], dtype=np.uint8)
    mask_light = cv2.inRange(hsv, lower_hsv_light, upper_hsv_light)

    # 深肤色
    lower_hsv_dark = np.array([0, 20, 30], dtype=np.uint8)
    upper_hsv_dark = np.array([25, 200, 180], dtype=np.uint8)
    mask_dark = cv2.inRange(hsv, lower_hsv_dark, upper_hsv_dark)

    # YCrCb 辅助
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_or(cv2.bitwise_or(mask_light, mask_dark), mask_ycrcb)
    skin_ratio = np.sum(skin_mask > 0) / max(skin_mask.size, 1)
    return bool(skin_ratio >= min_ratio)


def _run_haar_detection(gray_img: np.ndarray,
                         face_cascade: cv2.CascadeClassifier,
                         profile_cascade: cv2.CascadeClassifier | None = None) -> list[tuple[int, int, int, int]]:
    """执行 Haar 级联检测 (正面 + 侧脸) 并去重合并"""
    all_rects = []

    # 标准直方图均衡
    eq = cv2.equalizeHist(gray_img)

    # 正面检测 - 多组参数
    if face_cascade is not None and not (hasattr(face_cascade, 'empty') and face_cascade.empty()):
        try:
            # 参数组1: 标准
            faces = face_cascade.detectMultiScale(
                eq, scaleFactor=1.1, minNeighbors=5,
                minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            all_rects.extend([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces])

            # 参数组2: 宽松 (大脸/弱特征)
            faces2 = face_cascade.detectMultiScale(
                eq, scaleFactor=1.05, minNeighbors=3,
                minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            all_rects.extend([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces2])

            # 参数组3: CLAHE 增强
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_eq = clahe.apply(gray_img)
            faces3 = face_cascade.detectMultiScale(
                clahe_eq, scaleFactor=1.1, minNeighbors=4,
                minSize=(70, 70), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            all_rects.extend([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces3])
        except Exception as e:
            print(f'[_run_haar_detection] OpenCV 错误: {e}')

    # 侧脸检测
    if profile_cascade is not None and not (hasattr(profile_cascade, 'empty') and profile_cascade.empty()):
        try:
            profiles = profile_cascade.detectMultiScale(
                eq, scaleFactor=1.1, minNeighbors=5,
                minSize=(80, 80),
            )
            all_rects.extend([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in profiles])
        except Exception:
            pass

    return _nms_face_rects(all_rects, iou_thresh=0.3)


def _compute_skin_pixel_mask(bgr_arr: np.ndarray) -> np.ndarray:
    """生成肤色像素布尔掩码 (HSV + YCrCb 双空间, 覆盖全人种)

    返回: (H, W) 布尔数组, True=肤色像素
    """
    hsv = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2YCrCb)

    # 浅肤色
    lower_hsv_light = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv_light = np.array([25, 170, 255], dtype=np.uint8)
    mask_light = cv2.inRange(hsv, lower_hsv_light, upper_hsv_light)

    # 深肤色
    lower_hsv_dark = np.array([0, 20, 30], dtype=np.uint8)
    upper_hsv_dark = np.array([25, 200, 180], dtype=np.uint8)
    mask_dark = cv2.inRange(hsv, lower_hsv_dark, upper_hsv_dark)

    # YCrCb 辅助
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_or(cv2.bitwise_or(mask_light, mask_dark), mask_ycrcb)
    return skin_mask > 0


def classify_skin_tone(bgr_arr: np.ndarray) -> str:
    """肤色分类 (v51 重写): 返回 "浅肤色"/"中间肤色"/"深肤色"/"未知"

    v51 重大修复:
    - Bug 修复: ITA° 公式使用标准 CIE Lab 范围 (L*:0-100, b*:-128~127)
      旧版错误使用了 OpenCV 原生值 (L:0-255, b:0-255), 导致 ITA 偏低误判为深肤色
    - 肤色像素过滤: 先用 HSV+YCrCb 掩码排除头发/眉毛/阴影等非肤色像素
      旧版对整个 ROI 取均值, 深色头发/阴影会严重拉低 L 均值
    - 阈值基于临床皮肤科 ITA 标准 + 东亚人群分布微调:
      * ITA° > 48 → 浅肤色 (白皙/红润, 符合东亚审美)
      * ITA° > 20 → 中间肤色 (自然色, 东亚人群主体)
      * ITA° ≤ 20 → 深肤色 (小麦/橄榄/深色)
    - Cr/Y 色度比辅助判别: 高 chroma + 中高 ITA → 红润调
    """
    if bgr_arr.size == 0:
        return '未知'

    try:
        ycrcb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2Lab)

        # ── v51: 肤色像素掩码, 排除头发/眉毛/阴影干扰 ──
        skin_mask = _compute_skin_pixel_mask(bgr_arr)
        skin_count = np.count_nonzero(skin_mask)
        if skin_count < 100:
            # 肤色像素太少, 回退到全 ROI 均值
            skin_mask = np.ones(bgr_arr.shape[:2], dtype=bool)

        # ── 亮度 Top-30% 肤色像素 (高光区域, 避免阴影) ──
        y_ch = ycrcb[:, :, 0][skin_mask]
        y_order = np.argsort(y_ch)[::-1]          # 亮度降序索引
        top_n = max(len(y_ch) // 3, 50)
        y_top = np.mean(y_ch[y_order[:top_n]])

        # ── Cr/Y 归一化色度比 (红润度指标, Cr越高越暖) ──
        cr_ch = ycrcb[:, :, 1][skin_mask]
        cr_mean = float(np.mean(cr_ch))
        chroma_ratio = cr_mean / max(y_top, 1)

        # ── v53.6: 标准 CIE ITA° (Top-30% 最亮肤色像素, 排除阴影光照干扰) ──
        # OpenCV Lab → CIE Lab 转换:
        #   L*_cie = L_opencv * 100 / 255    (0-255 → 0-100)
        #   b*_cie = b_opencv - 128           (0-255 → -128~127)
        #   ITA° = arctan2(L* - 50, b*) × 180/π
        l_ch = lab[:, :, 0][skin_mask].astype(np.float64)
        b_ch = lab[:, :, 2][skin_mask].astype(np.float64)
        l_cie = np.mean(l_ch[y_order[:top_n]]) * 100.0 / 255.0   # Top-30%
        b_cie = np.mean(b_ch[y_order[:top_n]]) - 128.0           # Top-30%
        ita = float(np.degrees(np.arctan2(l_cie - 50.0, b_cie)))

        # ── v51: 临床 ITA 标准 + 东亚人群微调 ──
        # 标准临床 ITA: >55很浅 / 41-55浅 / 28-41中等 / 10-28褐 / <-30深
        # 考虑东亚人群 L* 整体偏低(肤色黄调吸收光), 将"浅肤色"门槛从55降到48
        if ita > 50 and chroma_ratio < 0.78:
            return '浅肤色'   # 特别白皙 (高ITA + 低色度 = 冷白皮)
        if ita > 48:
            return '浅肤色'   # 白净型 (ITA极高)
        elif ita > 20:
            return '中间肤色' # 自然色 (东亚人群主体)
        else:
            return '深肤色'   # 小麦/橄榄/深色 (ITA≤20)
    except Exception:
        return '未知'


def _create_skin_mask(h: int, w: int) -> np.ndarray:
    """v50: 面部比例掩码, 排除眼/鼻/唇/眉区域的干扰
    
    返回: 255=皮肤区域, 0=排除区域
    """
    mask = np.ones((h, w), dtype=np.uint8) * 255
    exclude = np.zeros((h, w), dtype=np.uint8)

    # 眼部+眉骨: y=[12%, 45%], x=[12%, 88%]
    cv2.rectangle(exclude, (int(w * 0.12), int(h * 0.12)),
                  (int(w * 0.88), int(h * 0.45)), 255, -1)
    # 鼻部: y=[40%, 65%], x=[30%, 70%]
    cv2.rectangle(exclude, (int(w * 0.30), int(h * 0.40)),
                  (int(w * 0.70), int(h * 0.65)), 255, -1)
    # 唇部: y=[65%, 92%], x=[20%, 80%]
    cv2.rectangle(exclude, (int(w * 0.20), int(h * 0.65)),
                  (int(w * 0.80), int(h * 0.92)), 255, -1)

    # 柔和边缘
    exclude = cv2.GaussianBlur(exclude, (9, 9), 3)
    exclude = (exclude > 30).astype(np.uint8) * 255
    mask = np.clip(mask.astype(np.int16) - exclude.astype(np.int16), 0, 255).astype(np.uint8)
    return mask


def detect_skin_blemishes(
    face_bgr: np.ndarray,
    debug: bool = False,
) -> dict[str, Any]:
    """v50: 统计自适应瑕疵检测 (SCUT-FBP5500 2300张校准)

    改进:
    1. 面部比例掩码: 排除眼/鼻/唇/眉区域假阳性
    2. 肤色统计自适应: 痘印阈值=皮肤S中位数+2.5MAD, 色斑=L中位数-2.5MAD
    3. 圆形度过滤: 排除不规则纹理误检
    4. SCUT校准: P50=1.7, P90=3.9, P95=4.5, >4.0仅8.5%

    Args:
        face_bgr: 人脸 ROI (BGR 格式)
        debug: 是否返回可视化 mask

    Returns:
        {blemish_score: float 0-10, acne_count: int, spot_count: int,
         details: str, mask: np.ndarray | None}
    """
    if face_bgr.size == 0:
        return {'blemish_score': 0.0, 'acne_count': 0, 'spot_count': 0,
                'details': '无人脸区域', 'mask': None}

    h, w = face_bgr.shape[:2]

    try:
        skin_mask = _create_skin_mask(h, w)
        skin_area = float(max(np.count_nonzero(skin_mask), 100))

        hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2Lab)

        # ── 皮肤区域肤色统计 ──
        skin_h = hsv[:, :, 0][skin_mask > 0].flatten()
        skin_s = hsv[:, :, 1][skin_mask > 0].flatten()
        skin_l = lab[:, :, 0][skin_mask > 0].flatten()

        if len(skin_h) < 100:
            return {'blemish_score': 0.0, 'acne_count': 0, 'spot_count': 0,
                    'details': '皮肤区域不足', 'mask': None}

        s_median = float(np.median(skin_s))
        s_mad = float(np.median(np.abs(skin_s - s_median)))
        l_median = float(np.median(skin_l))
        l_mad = float(np.median(np.abs(skin_l - l_median)))

        # ── 自适应阈值 (2.5σ/MAD, 至少20/15的绝对偏移) ──
        acne_s_thresh = s_median + max(2.5 * max(s_mad, 3.0), 20.0)
        spot_l_thresh = l_median - max(2.5 * max(l_mad, 3.0), 15.0)

        # ══ 红色痘印检测 ══
        lower_r1 = np.array([0, int(acne_s_thresh), 50], dtype=np.uint8)
        upper_r1 = np.array([15, 255, 230], dtype=np.uint8)
        lower_r2 = np.array([160, int(acne_s_thresh), 50], dtype=np.uint8)
        upper_r2 = np.array([180, 255, 230], dtype=np.uint8)

        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_r1, upper_r1),
            cv2.inRange(hsv, lower_r2, upper_r2),
        )
        mask_red = cv2.bitwise_and(mask_red, skin_mask)

        k_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k_s)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, k_m)

        nl, lb, st, _ = cv2.connectedComponentsWithStats(mask_red, connectivity=8)
        acne_count = 0
        for i in range(1, nl):
            area = st[i, cv2.CC_STAT_AREA]
            if area < 8 or area > 350:
                continue
            x, y = st[i, cv2.CC_STAT_LEFT], st[i, cv2.CC_STAT_TOP]
            bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            peri = 2.0 * (bw + bh)  # 矩形周长近似
            circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
            if circ > 0.25:
                acne_count += 1

        # ══ 暗色色斑检测 ══
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
            bw, bh = st_d[i, cv2.CC_STAT_WIDTH], st_d[i, cv2.CC_STAT_HEIGHT]
            peri = 2.0 * (bw + bh)
            circ = 4.0 * np.pi * area / max(peri * peri, 1.0)
            if circ > 0.20:
                spot_count += 1

        # ══ SCUT校准评分 ══
        area_norm = max(skin_area / 30000.0, 0.3)
        acne_norm = acne_count / area_norm
        spot_norm = spot_count / area_norm
        blemish_raw = np.sqrt(acne_norm * 2.0 + spot_norm * 1.0)
        blemish_score = round(min(blemish_raw, 10.0), 1)

        # ══ 详细描述 ══
        parts = []
        if acne_count >= 12:
            parts.append(f'明显痘印({acne_count}处)')
        elif acne_count >= 4:
            parts.append(f'少量痘印({acne_count}处)')
        elif acne_count >= 1:
            parts.append(f'个别痘印({acne_count}处)')
        if spot_count >= 10:
            parts.append(f'明显色斑({spot_count}处)')
        elif spot_count >= 3:
            parts.append(f'少量色斑({spot_count}处)')
        elif spot_count >= 1:
            parts.append(f'个别色斑({spot_count}处)')
        if not parts:
            details = '肤质洁净,未检测到明显瑕疵'
        else:
            details = '检测到: ' + ', '.join(parts)

        # Debug mask
        if debug:
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            vis[mask_red > 0] = [0, 0, 255]
            vis[mask_dark > 0] = [128, 0, 128]
        else:
            vis = None

        return {
            'blemish_score': blemish_score,
            'acne_count': acne_count,
            'spot_count': spot_count,
            'details': details,
            'mask': vis,
        }
    except Exception as e:
        return {'blemish_score': 0.0, 'acne_count': 0, 'spot_count': 0,
                'details': f'检测异常: {e}', 'mask': None}


def precheck_texture_quality(img: np.ndarray) -> dict[str, Any]:
    """v38: 全图纹理质量预检, 检测重度美颜/滤镜/压缩导致的纹理消失。

    Laplacian 方差分级:
        >= 80:  正常 (原图/轻微修图)
        30~80:  偏低 (中度修图, 分析仍可进行但可靠性下降)
        < 30:   极低 (重度修图, 面部纹理被抹平, 建议换图)

    Returns:
        (quality_label, detail_info)
    """
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if lap_var >= 80:
        label = 'good'
        text = '纹理质量正常'
    elif lap_var >= 30:
        label = 'warn'
        text = '纹理质量偏低，分析可靠性可能下降'
    else:
        label = 'bad'
        text = '纹理质量极低，检测到重度修图/滤镜，建议使用原图'

    return {
        'quality_label': label,
        'quality_text': text,
        'laplacian_var': round(lap_var, 2),
    }


def detect_face(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """v38: 多策略 Haar 检测, 防止大脸/弱特征漏检

    策略: 标准(3种级联参数配合) + CLAHE增强(无条件, 不只是在0检出时回退)
    """
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    face_cascade = _get_face_cascade()
    profile_cascade = _get_profile_cascade()
    img_h, img_w = gray.shape[:2]

    all_rects = _run_haar_detection(gray, face_cascade, profile_cascade)

    # 多人脸场景: 侧脸增强 (旋转 ±15°, ±30°)
    if profile_cascade is not None and not profile_cascade.empty():
        for angle in [15, -15, 30, -30]:
            try:
                M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1.0)
                rotated = cv2.warpAffine(gray, M, (img_w, img_h))
                profiles = profile_cascade.detectMultiScale(
                    rotated, scaleFactor=1.1, minNeighbors=4,
                    minSize=(70, 70),
                )
                for (x, y, w, h) in profiles:
                    all_rects.append((int(x), int(y), int(w), int(h)))
            except Exception:
                pass

    # NMS 去重
    rects = _nms_face_rects(all_rects, iou_thresh=0.3)

    # 多层验证过滤假阳性
    verified = []
    for (x, y, w, h) in rects:
        if not _check_face_aspect_ratio(w, h):
            continue

        # 肤色比例
        if y >= 0 and x >= 0 and y + h <= img_h and x + w <= img_w:
            roi = img[y:y+h, x:x+w]
            if img.shape[2] == 3:
                roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            else:
                roi_bgr = roi
            if not _is_skin_region(roi_bgr, min_ratio=0.30):
                continue

        # 边缘结构
        roi_gray = gray[y:y+h, x:x+w]
        if not _check_face_edge_structure(roi_gray):
            continue

        # 纹理方差
        if not _check_face_texture_variance(roi_gray, min_var=30.0):
            continue

        # 面部特征
        n_eyes, n_nose = _check_face_features_present(roi_gray)
        if n_eyes == 0 and n_nose == 0:
            continue

        verified.append((x, y, w, h))

    # 按面积降序排列
    verified.sort(key=lambda r: r[2] * r[3], reverse=True)
    return verified


def _check_face_features_soft(face_roi_gray: np.ndarray, nose_cascade: cv2.CascadeClassifier | None = None) -> dict[str, int]:
    """v36: 宽松面部特征检测, 用于软评分.
    与 _check_face_features_present 区别:
    - eye: minNeighbors=2 (宽松, 原版=3)
    - 额外尝试 left/right eye cascades
    - nose 不可用时只依赖眼睛
    """
    h, w = face_roi_gray.shape
    eye_cascade = _get_eye_cascade()
    nose_cascade = nose_cascade or _get_nose_cascade()

    # 眼睛检测 - 宽松参数
    eye_min = max(8, min(w // 10, 60))
    eyes = eye_cascade.detectMultiScale(
        face_roi_gray,
        scaleFactor=1.1,
        minNeighbors=2,  # 宽松
        minSize=(eye_min, eye_min),
        maxSize=(w // 2, h // 3),
    )
    n_eyes = len(eyes)

    # 鼻子检测
    n_nose = 0
    if nose_cascade is not None and not (hasattr(nose_cascade, 'empty') and nose_cascade.empty()):
        try:
            noses = nose_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(15, 15),
            )
            n_nose = len(noses)
        except Exception:
            pass

    return {'n_eyes': n_eyes, 'n_nose': n_nose}


def _compute_skin_ratio_soft(bgr_arr: np.ndarray) -> float:
    """计算 ROI 中肤色像素占比 (无阈值版本)"""
    return float(_is_skin_region(bgr_arr, min_ratio=0.01))


def _compute_edge_ratio_soft(face_roi_gray: np.ndarray) -> float:
    """计算上半区/下半区边缘密度比"""
    h, w = face_roi_gray.shape
    if h < 30 or w < 30:
        return 0.0
    edges = cv2.Canny(face_roi_gray, 50, 150)
    top_d = np.sum(edges[:h // 2, :] > 0) / max(edges[:h // 2, :].size, 1)
    bot_d = np.sum(edges[h // 2:, :] > 0) / max(edges[h // 2:, :].size, 1)
    if bot_d < 0.001:
        return 1.0
    return float(top_d / bot_d)


def _compute_laplacian_var_soft(face_roi_gray: np.ndarray) -> float:
    """计算 Laplacian 方差"""
    return float(cv2.Laplacian(face_roi_gray, cv2.CV_64F).var())


def detect_face_soft(img: np.ndarray) -> dict[str, Any]:
    """v36 软检测: 标记所有疑似人脸区域, 输出算法判定的人脸概率。

    Returns:
        { 'candidates': [...], 'probs': [...], 'scores_detail': [...] }
    """
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    face_cascade = _get_face_cascade()
    profile_cascade = _get_profile_cascade()
    img_h, img_w = gray.shape[:2]

    all_rects = _run_haar_detection(gray, face_cascade, profile_cascade)
    rects = _nms_face_rects(all_rects, iou_thresh=0.3)

    candidates = []
    probs = []
    scores_detail = []

    for (x, y, w, h) in rects:
        roi_gray = gray[y:y+h, x:x+w]
        if img.shape[2] == 3:
            roi_bgr = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_RGB2BGR)
        else:
            roi_bgr = img[y:y+h, x:x+w]

        # 6维软特征
        s1 = float(_check_face_aspect_ratio(w, h))
        s2 = _compute_skin_ratio_soft(roi_bgr)
        s3 = 1.0 if _check_face_edge_structure(roi_gray) else 0.0
        s4 = float(_check_face_texture_variance(roi_gray, min_var=10.0))

        soft_f = _check_face_features_soft(roi_gray)
        s5 = min(soft_f['n_eyes'] / 2.0, 1.0)
        s6 = min(soft_f['n_nose'], 1.0)

        # 自适应权重 (鼻子不可用时提高其他维度)
        if soft_f['n_nose'] == 0:
            weights = [0.20, 0.25, 0.20, 0.15, 0.20, 0.0]
        else:
            weights = [0.15, 0.20, 0.20, 0.10, 0.15, 0.20]

        prob = s1 * weights[0] + s2 * weights[1] + s3 * weights[2] + \
               s4 * weights[3] + s5 * weights[4] + s6 * weights[5]

        candidates.append((x, y, w, h))
        probs.append(round(prob, 3))
        scores_detail.append({
            'aspect_ratio': round(s1, 3),
            'skin_ratio': round(s2, 3),
            'edge_structure': round(s3, 3),
            'texture': round(s4, 3),
            'eyes': round(s5, 3),
            'nose': round(s6, 3),
        })

    return {
        'candidates': candidates,
        'probs': probs,
        'scores_detail': scores_detail,
    }


def detect_faces(img: np.ndarray,
                 enhance_side: bool = False,
                 enhance_large: bool = False) -> list[tuple[int, int, int, int]]:
    """v44 增强接口: 支持按需开启侧脸/大脸增强

    Args:
        img: RGB 图片
        enhance_side: True=启用侧脸旋转检测, False=跳过旋转增强保留基础侧脸检测
        enhance_large: True=启用大脸宽松参数组, False=跳过(更快但可能漏检大脸)
    """
    if enhance_side and enhance_large:
        # 默认完整版: 所有检测策略全开
        return detect_face(img)

    # 按需裁剪: 复制 detect_face() 逻辑但可跳过部分策略
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    face_cascade = _get_face_cascade()
    img_h, img_w = gray.shape[:2]
    all_rects = []

    # 标准直方图均衡
    eq = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_eq = clahe.apply(gray)

    if face_cascade is not None and not face_cascade.empty():
        try:
            # 参数组1: 标准 (始终开启)
            faces = face_cascade.detectMultiScale(
                eq, scaleFactor=1.1, minNeighbors=5,
                minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            all_rects.extend([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces])

            if enhance_large:
                # 参数组2: 宽松 (大脸/弱特征)
                faces2 = face_cascade.detectMultiScale(
                    eq, scaleFactor=1.05, minNeighbors=3,
                    minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE,
                )
                all_rects.extend([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces2])

            # 参数组3: CLAHE 增强 (始终开启)
            faces3 = face_cascade.detectMultiScale(
                clahe_eq, scaleFactor=1.1, minNeighbors=4,
                minSize=(70, 70), flags=cv2.CASCADE_SCALE_IMAGE,
            )
            all_rects.extend([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces3])
        except Exception as e:
            print(f'[detect_faces] OpenCV error: {e}')

    # 侧脸基础检测 (始终开启, 不依赖旋转)
    profile_cascade = _get_profile_cascade()
    if profile_cascade is not None and not profile_cascade.empty():
        try:
            profiles = profile_cascade.detectMultiScale(
                eq, scaleFactor=1.1, minNeighbors=4,
                minSize=(70, 70),
            )
            all_rects.extend([(int(x), int(y), int(w), int(h)) for (x, y, w, h) in profiles])
        except Exception:
            pass

    # 侧脸旋转增强 (可选)
    if enhance_side and profile_cascade is not None and not profile_cascade.empty():
        for angle in [15, -15, 30, -30]:
            try:
                M = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1.0)
                rotated = cv2.warpAffine(gray, M, (img_w, img_h))
                profiles = profile_cascade.detectMultiScale(
                    rotated, scaleFactor=1.1, minNeighbors=4,
                    minSize=(70, 70),
                )
                for (x, y, w, h) in profiles:
                    all_rects.append((int(x), int(y), int(w), int(h)))
            except Exception:
                pass

    # NMS 去重 + 多层验证 (与 detect_face 相同流程)
    rects = _nms_face_rects(all_rects, iou_thresh=0.3)

    verified = []
    for (x, y, w, h) in rects:
        if not _check_face_aspect_ratio(w, h):
            continue
        if y >= 0 and x >= 0 and y + h <= img_h and x + w <= img_w:
            roi = img[y:y+h, x:x+w]
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else roi
            if not _is_skin_region(roi_bgr, min_ratio=0.30):
                continue
        roi_gray = gray[y:y+h, x:x+w]
        if not _check_face_edge_structure(roi_gray):
            continue
        if not _check_face_texture_variance(roi_gray, min_var=30.0):
            continue
        n_eyes, n_nose = _check_face_features_present(roi_gray)
        if n_eyes == 0 and n_nose == 0:
            continue
        verified.append((x, y, w, h))

    verified.sort(key=lambda r: r[2] * r[3], reverse=True)
    return verified


# ═══════════════════════════════════════════════
#  纹理预检 (v38 增强版)
# ═══════════════════════════════════════════════

def texture_precheck(img: np.ndarray) -> dict[str, Any]:
    """图片纹理预检 - 检查清晰度、噪点、压缩伪影、光照"""
    h, w = img.shape[:2]
    if img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    # 1. 清晰度 (拉普拉斯方差)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. 噪点评估
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = np.mean(np.abs(gray.astype(float) - blurred.astype(float)))

    # 3. 光照均匀度
    h_blocks, w_blocks = 3, 3
    block_h = h // h_blocks
    block_w = w // w_blocks
    block_means = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.size > 0:
                block_means.append(np.mean(block))
    illum = 1.0 - (np.std(block_means) / max(np.mean(block_means), 1))

    # 4. 综合判定 (v38 分级)
    issues = []
    if laplacian_var < 30:
        issues.append('图片纹理极低，检测到重度修图/滤镜')
    elif laplacian_var < 50:
        issues.append('图片较模糊，建议使用清晰照片')
    elif laplacian_var < 80:
        issues.append('清晰度一般，建议光线充足环境拍摄')

    if noise > 15:
        issues.append('噪点较高，可能影响分析准确性')
    if illum < 0.6:
        issues.append('光照不均匀，建议正面均匀光源')

    if not issues:
        label, text = 'good', '图片质量优秀'
    elif len(issues) <= 1:
        label, text = 'warn', '图片质量一般'
    else:
        label, text = 'bad', '图片质量较差，建议更换'

    # v53: 纹理标签 (基于拉普拉斯方差, 供性别推断用)
    if laplacian_var > 200:
        texture_label = 'coarse'   # 高锐度 → 粗纹理
    elif laplacian_var > 80:
        texture_label = 'medium'
    else:
        texture_label = 'fine'     # 低锐度 → 细纹理/平滑

    return {
        'quality_label': label,
        'quality_text': text,
        'issues': issues,
        'laplacian_var': round(laplacian_var, 2),
        'noise_level': round(noise, 2),
        'illumination_uniformity': round(illum, 2),
        'resolution': f'{w}×{h}',
        'megapixels': round((w * h) / 1_000_000, 2),
        'texture_label': texture_label,
    }


# ═══════════════════════════════════════════════
#  背景去除 / 肤色检测
# ═══════════════════════════════════════════════

def remove_background(img: np.ndarray) -> np.ndarray:
    """背景去除 (基于肤色检测 + 边缘保持椭圆遮罩)"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        refined = np.zeros_like(skin_mask)
        cv2.drawContours(refined, [largest], -1, 255, -1)
        skin_mask = cv2.dilate(refined, kernel, iterations=3)

    skin_mask = cv2.GaussianBlur(skin_mask, (21, 21), 11)
    skin_mask = skin_mask.astype(float) / 255.0
    skin_mask = np.expand_dims(skin_mask, axis=2)

    white_bg = np.ones_like(img, dtype=float) * 255
    result = img.astype(float) * skin_mask + white_bg * (1 - skin_mask)
    return result.astype(np.uint8)


def estimate_skin_tone(img: np.ndarray, face_rects: list[tuple[int, int, int, int]]) -> str:
    """根据人脸区域估算肤色"""
    if not face_rects:
        return '未知'
    x, y, w, h = face_rects[0]
    cx, cy = x + w // 2, y + h // 2
    roi_w, roi_h = w // 3, h // 4
    rx = max(0, cx - roi_w // 2)
    ry = max(0, cy + h // 8)
    rw = min(roi_w, img.shape[1] - rx)
    rh = min(roi_h, img.shape[0] - ry)
    if rw <= 0 or rh <= 0:
        return '自然'
    roi = img[ry:ry+rh, rx:rx+rw]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    return classify_skin_tone(roi_bgr)


# ═══════════════════════════════════════════════
#  v39: MediaPipe FaceMesh 关键点提取
# ═══════════════════════════════════════════════

# MediaPipe FaceMesh 面部轮廓关键点索引 (36点, 覆盖额头→下巴→两侧颧弓)
_FACE_OVAL_IDXS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

_face_landmarker: Any = None  # 懒加载单例
_landmarker_last_error: str | None = None


def _get_face_landmarker() -> Any:
    """懒加载 MediaPipe FaceLandmarker (v0.10.x task API)"""
    global _face_landmarker, _landmarker_last_error
    if _face_landmarker is None and _landmarker_last_error is None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

            model_path = os.path.join(
                os.path.dirname(__file__), 'beauty_system', 'face_landmarker.task'
            )
            if not os.path.exists(model_path):
                _landmarker_last_error = f'模型文件不存在: {model_path}'
                return None

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionTaskRunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
            _face_landmarker = FaceLandmarker.create_from_options(options)
        except ImportError as e:
            _landmarker_last_error = f'MediaPipe 导入失败: {e}'
            return None
        except Exception as e:
            _landmarker_last_error = f'FaceLandmarker 初始化失败: {e}'
            return None
    if _landmarker_last_error:
        return None
    return _face_landmarker


def extract_face_landmarks(
    img: np.ndarray,
    face_rect: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    """v39: 使用 MediaPipe FaceLandmarker 提取 478 关键点

    Args:
        img: RGB 图片 (H, W, 3) uint8
        face_rect: 可选 (x, y, w, h) 人脸框, 传入则裁剪 ROI 加速

    Returns:
        landmarks: (478, 3) 数组 [x, y, z] 像素坐标, 或 None
    """
    landmarker = _get_face_landmarker()
    if landmarker is None:
        return None

    try:
        from mediapipe import Image as MpImage, ImageFormat as MpImageFormat

        if face_rect is not None:
            x, y, w, h = face_rect
            # 扩展 15% 边距确保轮廓完整
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img.shape[1], x + w + margin_x)
            y2 = min(img.shape[0], y + h + margin_y)
            roi = img[y1:y2, x1:x2]
        else:
            x1, y1 = 0, 0
            roi = img

        # MediaPipe 要求 RGB uint8，连续内存
        if roi.dtype != np.uint8:
            roi = roi.astype(np.uint8)
        if not roi.flags['C_CONTIGUOUS']:
            roi = np.ascontiguousarray(roi)

        # 创建 MediaPipe Image
        mp_image = MpImage(image_format=MpImageFormat.SRGB, data=roi)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        face_lm = result.face_landmarks[0]
        h_roi, w_roi = roi.shape[:2]

        # 转为像素坐标 (相对于原图)
        n_pts = len(face_lm)
        landmarks = np.zeros((n_pts, 3), dtype=np.float32)
        for i, lm in enumerate(face_lm):
            landmarks[i, 0] = lm.x * w_roi + x1  # 像素 x
            landmarks[i, 1] = lm.y * h_roi + y1  # 像素 y
            landmarks[i, 2] = lm.z * w_roi        # 深度 (与宽度同单位)
        return landmarks

    except Exception:
        return None


def compute_face_ratio_from_landmarks(landmarks: np.ndarray) -> float | None:
    """从 FaceLandmarker 关键点 (468/478) 计算解剖学面宽比 (颧弓宽 / 面部高度)

    使用面部轮廓点 (FACE_OVAL 36点) 的外包矩形:
    - 宽度 = max(x) - min(x)  (覆盖两侧颧弓)
    - 高度 = max(y) - min(y)  (额头→下巴)

    Returns:
        face_ratio 或 None (关键点不足)
    """
    if landmarks is None or landmarks.shape[0] < 36:
        return None

    try:
        # 只用面部轮廓关键点 (36点)
        oval = landmarks[_FACE_OVAL_IDXS]
        xs, ys = oval[:, 0], oval[:, 1]
        face_w = float(xs.max() - xs.min())
        face_h = float(ys.max() - ys.min())
        if face_h < 10:
            return None
        return round(face_w / face_h, 4)
    except (IndexError, ValueError):
        return None


# ═══════════════════════════════════════════════
#  图片格式转换
# ═══════════════════════════════════════════════

def numpy_to_png_bytes(img: np.ndarray) -> bytes:
    """numpy 数组转 PNG 字节"""
    pil_img = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return buf.getvalue()


def create_thumbnail(img: np.ndarray, size: tuple[int, int] = (200, 200)) -> np.ndarray:
    """生成缩略图"""
    h, w = img.shape[:2]
    tw, th = size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    result = np.ones((th, tw, 3), dtype=np.uint8) * 30
    y_offset = (th - new_h) // 2
    x_offset = (tw - new_w) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return result
