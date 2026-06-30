"""
颜值矩阵分析系统 — 核心计算引擎 v53.5
基于 Hill 函数 + 5×5 特征矩阵 + Sigmoid 加分链 + SCUT-FBP5500 校准
v53.5: 86点解剖学面宽比替代检测框 — MediaPipe FaceMesh 关键点提取
v53.4: 面宽比平坦区修正 — [0.72,0.75]满分, 两端线性衰减 (SCUT 86点验证)
v53.3: geo_bonus去除居中 + 双向偏差罚分 (过窄/过宽均扣分, 最优≈0.75)
v53.2: geo_bonus评分膨胀修复 (对称Sigmoid + 允许负数 + (1-w/h)主度量)
v53.1: PM审查问题修复 (P0-1~P2-5) + 三端一致性 + 瑕疵检测链补全
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Any, Callable, cast
import yaml
import os

# ── 加载配置 ──
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'beauty_system', 'config.yaml')
with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
    _CFG: dict[str, Any] = yaml.safe_load(f)

PREFERENCE_PRESETS: dict[str, list[float]] = {
    k: cast(list[float], v['weights']) for k, v in _CFG['preference_presets'].items()
}
SKIN_CLARITY_WEIGHTS: dict[str, float] = cast(dict[str, float], _CFG['skin_clarity_weights'])
SKIN_TONE_WEIGHTS: dict[str, float] = cast(dict[str, float], _CFG['skin_tone_weights'])
GEO_CLARITY_WEIGHTS: dict[str, float] = cast(dict[str, float], _CFG['geo_clarity_weights'])
GRADES: list[dict[str, Any]] = cast(list[dict[str, Any]], _CFG['grades'])

# ═══ v44 评分公式参数 ═══
_SCORING: dict[str, Any] = cast(dict[str, Any], _CFG['scoring'])
VMAX: float = float(cast(float, _SCORING['vmax']))
K_HALF: float = float(cast(float, _SCORING['k_half']))
Q_WEIGHT: float = float(cast(float, _SCORING['q_weight']))
DET_CAP: float = float(cast(float, _SCORING['det_cap']))
CALIB_OFFSET: float = float(cast(float, _SCORING['calib_offset']))
GEO_WEIGHT: float = float(cast(float, _SCORING['geo_weight']))

_SKIN_BONUS_CFG: dict[str, Any] = cast(dict[str, Any], _CFG['skin_clarity_bonus_cfg'])
_SKIN_B_MAX: float = float(cast(float, _SKIN_BONUS_CFG['b_max']))
_SKIN_BONUS_K: float = float(cast(float, _SKIN_BONUS_CFG['k']))
_SKIN_BONUS_X0: float = float(cast(float, _SKIN_BONUS_CFG['x0']))

_GEO_BONUS_CFG: dict[str, Any] = cast(dict[str, Any], _CFG['geo_bonus_cfg'])
_GEO_B_MAX: float = float(cast(float, _GEO_BONUS_CFG['b_max']))
_GEO_BONUS_K: float = float(cast(float, _GEO_BONUS_CFG['k']))
_GEO_BONUS_X0: float = float(cast(float, _GEO_BONUS_CFG['x0']))
_GEO_PENALTY_SLOPE: float = float(cast(float, _GEO_BONUS_CFG.get('penalty_slope', 1.0)))




# ═══════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════

@dataclass
class FaceFeatures:
    """人脸11维美学特征 + 性别"""
    symmetry: float = 0.0        # 对称度
    proportion: float = 0.0      # 比例协调
    skin_texture: float = 0.0    # 皮肤质感
    contour: float = 0.0         # 轮廓流畅
    eye_beauty: float = 0.0      # 眼睛美感
    nose_elegance: float = 0.0   # 鼻型优雅
    lip_charm: float = 0.0       # 唇形魅力
    youth_index: float = 0.0     # 年轻指数
    uniqueness: float = 0.0      # 独特气质
    harmony: float = 0.0         # 五官和谐
    skin_clarity: float = 0.0    # 肤质白净度
    skin_tone_label: str = ""    # 肤色标签
    blemish_score: float = 0.0   # v49: 瑕疵评分 0-10 (0=无瑕疵)
    gender: str = "unknown"           # v53: 性别 (male/female/unknown, ML模型)
    gender_confidence: float = 0.0    # v53: 性别置信度 0-1

    def as_dict(self) -> dict[str, Any]:
        return {
            'symmetry': round(float(self.symmetry), 3),
            'proportion': round(float(self.proportion), 3),
            'skin_texture': round(float(self.skin_texture), 3),
            'contour': round(float(self.contour), 3),
            'eye_beauty': round(float(self.eye_beauty), 3),
            'nose_elegance': round(float(self.nose_elegance), 3),
            'lip_charm': round(float(self.lip_charm), 3),
            'youth_index': round(float(self.youth_index), 3),
            'uniqueness': round(float(self.uniqueness), 3),
            'harmony': round(float(self.harmony), 3),
            'skin_clarity': round(float(self.skin_clarity), 3),
            'blemish_score': round(float(self.blemish_score), 3),
            'gender': self.gender,                                      # v53
            'gender_confidence': round(float(self.gender_confidence), 3),  # v53
        }

    def to_list(self) -> list[float]:
        """转为10维向量"""
        return [
            self.symmetry, self.proportion, self.skin_texture,
            self.contour, self.eye_beauty, self.nose_elegance,
            self.lip_charm, self.youth_index, self.uniqueness, self.harmony
        ]


@dataclass
class GeoDimensions:
    """MediaPipe 6维几何美学维度"""
    available: bool = False
    eye_aspect_ratio: float = 0.0
    mouth_aspect_ratio: float = 0.0
    nose_mouth_ratio: float = 0.0
    face_ratio: float = 0.0
    jaw_angle: float = 0.0
    symmetry_index: float = 0.0
    bonus: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            'available': self.available,
            'eye_aspect_ratio': round(self.eye_aspect_ratio, 3),
            'mouth_aspect_ratio': round(self.mouth_aspect_ratio, 3),
            'nose_mouth_ratio': round(self.nose_mouth_ratio, 3),
            'face_ratio': round(self.face_ratio, 3),
            'jaw_angle': round(self.jaw_angle, 3),
            'symmetry_index': round(self.symmetry_index, 3),
            'bonus': round(self.bonus, 3),
        }


# ═══════════════════════════════════════════════
#  Hill 函数族
# ═══════════════════════════════════════════════

def hill(x: float, k: float = 5.0, n: float = 2.0) -> float:
    """Hill 函数: H(x) = x^n / (k^n + x^n), 归一化到 [0,1]"""
    if x <= 0:
        return 0.0
    xn = x ** n
    kn = k ** n
    return xn / (kn + xn)


def inverse_hill(y: float, k: float = 5.0, n: float = 2.0) -> float:
    """Hill 逆函数: 从 [0,1] 还原原始值"""
    if y <= 0:
        return 0.0
    if y >= 1:
        return 100.0
    return k * (y / (1 - y)) ** (1.0 / n)


# ═══════════════════════════════════════════════
#  真实特征提取 (v48: 基于 SCUT-FBP5500 训练)
# ═══════════════════════════════════════════════

# ── 模型懒加载 ──
_beauty_model_cache: dict[str, Any] | None = None
_BEAUTY_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'stats_output', 'beauty_model_full.pkl')


def _load_beauty_model() -> dict[str, Any] | None:
    """懒加载训练好的 Ridge 回归模型 (scaler + poly + ridge + feat_keys)"""
    global _beauty_model_cache
    if _beauty_model_cache is not None:
        return _beauty_model_cache
    if not os.path.exists(_BEAUTY_MODEL_PATH):
        return None
    try:
        import pickle
        with open(_BEAUTY_MODEL_PATH, 'rb') as f:
            _beauty_model_cache = cast(dict[str, Any], pickle.load(f))
        return _beauty_model_cache
    except Exception:
        return None


# ── 性别模型懒加载 (v2: CNN ONNX 模型) ──
_gender_cnn_session: Any = None
_GENDER_CNN_PATH = os.path.join(os.path.dirname(__file__), 'stats_output', 'gender_cnn_v2.onnx')
_GENDER_CNN_INPUT_SIZE = 224
_GENDER_CNN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_GENDER_CNN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_gender_cnn() -> Any:
    """懒加载 ONNX CNN 性别模型 (MobileNetV3-Small, 91.4% 准确率)"""
    global _gender_cnn_session
    if _gender_cnn_session is not None:
        return _gender_cnn_session
    if not os.path.exists(_GENDER_CNN_PATH):
        return None
    try:
        import onnxruntime as ort
        _gender_cnn_session = ort.InferenceSession(_GENDER_CNN_PATH)
        return _gender_cnn_session
    except Exception:
        return None


def predict_gender_cnn(img: np.ndarray, face_rect: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """v2: CNN 性别分类器 (MobileNetV3-Small 迁移学习, 91.4% Acc, ROC-AUC 0.977)
    
    直接对人脸区域图像进行端到端推理, 替代旧版 32D 手工特征 + LR 方案.
    Domain Shift 鲁棒性显著提升: Reverse_Hill 女性检出率 16%→88%.

    Args:
        img: RGB 图像 (H, W, 3)
        face_rect: (x, y, w, h) 人脸检测框, 为 None 时用全图

    Returns:
        {'gender': 'male'|'female'|'unknown', 'confidence': 0-1, 'prob_male': 0-1}
    """
    session = _load_gender_cnn()

    # 模型不可用 → 回退到旧模型
    if session is None:
        feats32 = _extract_32d_features(img, face_rect)
        return _legacy_predict_gender(feats32)

    try:
        H, W = img.shape[:2]

        # 裁剪人脸 ROI (带边距)
        if face_rect:
            x, y, w, h = face_rect
            margin = 0.3
            mx, my = int(w * margin), int(h * margin)
            x1 = max(0, x - mx)
            y1 = max(0, y - my)
            x2 = min(W, x + w + mx)
            y2 = min(H, y + h + my)
            face_roi = img[y1:y2, x1:x2]
        else:
            face_roi = img

        if face_roi.size == 0:
            feats32 = _extract_32d_features(img, face_rect)
            return _fallback_gender_heuristic(feats32)

        # RGB → Resize(224) → Float[0,1] → ImageNet Normalize → CHW
        face_resized = cv2.resize(face_roi, (_GENDER_CNN_INPUT_SIZE, _GENDER_CNN_INPUT_SIZE))
        face_float = face_resized.astype(np.float32) / 255.0
        face_norm = (face_float - _GENDER_CNN_MEAN) / _GENDER_CNN_STD
        face_chw = np.transpose(face_norm, (2, 0, 1))
        face_batch = np.expand_dims(face_chw, axis=0).astype(np.float32)

        # ONNX 推理
        outputs = session.run(['output'], {'input': face_batch})
        logits = outputs[0][0]
        logits_max = np.max(logits)
        prob = np.exp(logits - logits_max) / np.sum(np.exp(logits - logits_max))
        prob_female, prob_male = float(prob[0]), float(prob[1])

        confidence = max(prob_female, prob_male)

        if confidence < 0.55:
            gender = 'unknown'
        elif prob_male > prob_female:
            gender = 'male'
        else:
            gender = 'female'

        return {
            'gender': gender,
            'confidence': round(confidence, 3),
            'prob_male': round(prob_male, 3),
            'method': 'cnn_mobilenetv3',
        }
    except Exception:
        feats32 = _extract_32d_features(img, face_rect)
        return _fallback_gender_heuristic(feats32)


def _legacy_predict_gender(feats32: dict[str, float]) -> dict[str, Any]:
    """旧版 ML 性别分类器 (回退用, LogisticRegressionCV)"""
    # 尝试加载旧模型
    legacy_path = os.path.join(os.path.dirname(__file__), 'stats_output', 'gender_model_v1.pkl')
    if not os.path.exists(legacy_path):
        return _fallback_gender_heuristic(feats32)
    try:
        import pickle
        with open(legacy_path, 'rb') as f:
            model = cast(dict[str, Any], pickle.load(f))
        feat_keys = model['feat_keys']
        feat_vec = np.array([[feats32.get(k, 0.0) for k in feat_keys]], dtype=np.float64)
        if np.isnan(feat_vec).any() or np.isinf(feat_vec).any():
            return _fallback_gender_heuristic(feats32)
        X_scaled = model['scaler'].transform(feat_vec)
        X_poly = model['poly'].transform(X_scaled)
        prob = model['clf'].predict_proba(X_poly)[0]
        prob_male = float(prob[1])
        pred = int(model['clf'].predict(X_poly)[0])
        confidence = max(prob_male, 1 - prob_male)
        if confidence < 0.55:
            gender = 'unknown'
        elif pred == 1:
            gender = 'male'
        else:
            gender = 'female'
        return {
            'gender': gender,
            'confidence': round(confidence, 3),
            'prob_male': round(prob_male, 3),
            'method': 'ml_logistic_cv_legacy',
        }
    except Exception:
        return _fallback_gender_heuristic(feats32)


def _fallback_gender_heuristic(feats32: dict[str, float]) -> dict[str, Any]:
    """模型不可用时的降级启发式 (v53 零中心化版本)"""
    face_aspect = feats32.get('face_aspect', 0.76)
    topbot_ratio = feats32.get('face_topbot_edge_ratio', 1.25)
    face_sharpness = feats32.get('face_sharpness', 4.5)
    male_score = ((0.76 - face_aspect) * 2.5
                  + (topbot_ratio - 1.25) * 1.0
                  + (face_sharpness - 4.5) * 0.3)
    if male_score > 0.3:
        gender = 'male'
    elif male_score < -0.3:
        gender = 'female'
    else:
        gender = 'unknown'
    return {
        'gender': gender,
        'confidence': round(0.5 + abs(male_score) * 0.3, 3),
        'prob_male': round(1.0 / (1.0 + np.exp(-male_score * 2.0)), 3),
        'method': 'heuristic_fallback',
    }


# ── 肤色 CNN 模型懒加载 (v3: 3分类 EfficientNet-B0, 85.6% Acc vs ITA°) ──
_skin_tone_cnn_session: Any = None
_SKIN_TONE_CNN_PATH = os.path.join(os.path.dirname(__file__), 'stats_output', 'skin_tone_cnn_v3.onnx')
_SKIN_TONE_CNN_INPUT_SIZE = 224
_SKIN_TONE_CNN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_SKIN_TONE_CNN_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SKIN_TONE_3CLASS = ['深肤色', '中间肤色', '浅肤色']


def _load_skin_tone_cnn() -> Any:
    """懒加载 ONNX CNN 肤色模型 (MobileNetV3-Small, 67% 3类准确率)"""
    global _skin_tone_cnn_session
    if _skin_tone_cnn_session is not None:
        return _skin_tone_cnn_session
    if not os.path.exists(_SKIN_TONE_CNN_PATH):
        return None
    try:
        import onnxruntime as ort
        _skin_tone_cnn_session = ort.InferenceSession(_SKIN_TONE_CNN_PATH)
        return _skin_tone_cnn_session
    except Exception:
        return None


def predict_skin_tone_cnn(face_bgr: np.ndarray, skin_r_mean: float = 0.5,
                          skin_b_mean: float = 0.5) -> str:
    """CNN 肤色5分类推理 (3分类 CNN + R/B 后处理)

    流程:
    1. EfficientNet-B0 预测 3大类 (深/中/浅) — 85.6% vs ITA°
    2. R/B 比值细分 → 5小类 (白皙/红润/自然/小麦/橄榄)
    3. 模型不可用时回退到 ITA° 物理公式

    Args:
        face_bgr: 人脸 ROI (BGR格式)
        skin_r_mean: 肤色 R 通道均值 (用于后处理细分)
        skin_b_mean: 肤色 B 通道均值 (用于后处理细分)

    Returns:
        '白皙'|'红润'|'自然'|'小麦'|'橄榄'
    """
    session = _load_skin_tone_cnn()

    # 模型不可用 → 回退 ITA° 公式
    if session is None:
        try:
            from image_utils import classify_skin_tone
            skin_3class = classify_skin_tone(face_bgr)
            r_b_ratio = skin_r_mean / max(skin_b_mean, 0.01)
            if skin_3class == '浅肤色':
                return '红润' if r_b_ratio > 1.08 else '白皙'
            elif skin_3class == '深肤色':
                return '橄榄' if r_b_ratio > 1.2 else '小麦'
            else:
                return '自然'
        except Exception:
            return '自然'

    try:
        # BGR → RGB → Resize(224) → Float[0,1] → ImageNet Normalize → CHW → Batch
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (_SKIN_TONE_CNN_INPUT_SIZE, _SKIN_TONE_CNN_INPUT_SIZE))
        face_float = face_resized.astype(np.float32) / 255.0
        face_norm = (face_float - _SKIN_TONE_CNN_MEAN) / _SKIN_TONE_CNN_STD
        face_chw = np.transpose(face_norm, (2, 0, 1))
        face_batch = np.expand_dims(face_chw, axis=0).astype(np.float32)

        # ONNX 推理 → 3分类 logits
        outputs = session.run(['output'], {'input': face_batch})
        logits = outputs[0][0]
        pred_class = int(np.argmax(logits))
        skin_3class = _SKIN_TONE_3CLASS[pred_class]

        # R/B 后处理: 3大类 → 5小类
        r_b_ratio = skin_r_mean / max(skin_b_mean, 0.01)
        if skin_3class == '浅肤色':
            return '红润' if r_b_ratio > 1.08 else '白皙'
        elif skin_3class == '深肤色':
            return '橄榄' if r_b_ratio > 1.2 else '小麦'
        else:
            return '自然'

    except Exception:
        # 异常回退
        try:
            from image_utils import classify_skin_tone
            skin_3class = classify_skin_tone(face_bgr)
            r_b_ratio = skin_r_mean / max(skin_b_mean, 0.01)
            if skin_3class == '浅肤色':
                return '红润' if r_b_ratio > 1.08 else '白皙'
            elif skin_3class == '深肤色':
                return '橄榄' if r_b_ratio > 1.2 else '小麦'
            else:
                return '自然'
        except Exception:
            return '自然'


def _hist_entropy_8bit(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist = hist.flatten()
    hist = hist / max(hist.sum(), 1)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def _estimate_face_symmetry(face_gray: np.ndarray) -> float:
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


def _extract_32d_features(img: np.ndarray,
                          face_rect: tuple[int, int, int, int] | None = None) -> dict[str, float]:
    """提取 32 维真实视觉特征 (与训练脚本一致)"""
    h, w = img.shape[:2]
    feats: dict[str, float] = {}
    if img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    feats['global_brightness'] = float(np.mean(gray)) / 255.0
    feats['global_contrast'] = float(np.std(gray)) / 128.0
    feats['global_sharpness'] = float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()))
    feats['global_entropy'] = float(_hist_entropy_8bit(gray))
    feats['global_saturation_mean'] = float(np.mean(hsv[:, :, 1])) / 255.0
    feats['global_saturation_std'] = float(np.std(hsv[:, :, 1])) / 128.0
    for ch, name in enumerate(['B', 'G', 'R']):
        ch_data = bgr[:, :, ch].flatten()
        feats[f'color_{name}_mean'] = float(np.mean(ch_data)) / 255.0
        feats[f'color_{name}_std'] = float(np.std(ch_data)) / 128.0
    r_m = float(np.mean(bgr[:, :, 2]))
    b_m = float(np.mean(bgr[:, :, 0]))
    feats['warmth_ratio'] = (r_m - b_m) / max(r_m + b_m, 1)

    if face_rect:
        fx, fy, fw, fh = face_rect
        fx, fy = max(0, fx), max(0, fy)
        fw = min(fw, w - fx); fh = min(fh, h - fy)
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
            feats['face_entropy'] = float(_hist_entropy_8bit(fg))
            feats['face_saturation'] = float(np.mean(fh_s[:, :, 1])) / 255.0
            for ch, name in enumerate(['B', 'G', 'R']):
                cd = fr[:, :, ch].flatten()
                feats[f'skin_{name}_mean'] = float(np.mean(cd)) / 255.0
                feats[f'skin_{name}_std'] = float(np.std(cd)) / 128.0
            if fw > 20 and fh > 20:
                feats['face_symmetry'] = _estimate_face_symmetry(fg)
            edges = cv2.Canny(fg, 50, 150)
            feats['face_edge_density'] = float(np.sum(edges > 0)) / max(edges.size, 1)
            th = fg[:fh//2, :]; bh = fg[fh//2:, :]
            te = cv2.Canny(th, 50, 150); be = cv2.Canny(bh, 50, 150)
            td = np.sum(te > 0) / max(te.size, 1)
            bd = np.sum(be > 0) / max(be.size, 1)
            feats['face_topbot_edge_ratio'] = td / max(bd, 0.001)
            feats['face_lap_block_var'] = _block_lap_std(fg)
            # v53: 性别ML模型额外特征
            lower_face = fg[fh//2:, :]
            lh, lw = lower_face.shape
            if lh > 10 and lw > 10:
                le = cv2.Canny(lower_face, 50, 150)
                feats['jaw_edge_density'] = float(np.sum(le > 0)) / max(le.size, 1)
            cr = fg[fh//4:fh//2, fw//4:3*fw//4]
            if cr.size > 0:
                feats['cheek_sharpness'] = float(np.log1p(cv2.Laplacian(cr, cv2.CV_64F).var()))
    # 填充缺失
    _default_face_keys = [
        'face_area_ratio', 'face_aspect', 'face_x_ratio', 'face_y_ratio',
        'face_brightness', 'face_contrast', 'face_sharpness', 'face_entropy',
        'face_saturation', 'face_symmetry', 'face_edge_density',
        'face_topbot_edge_ratio', 'face_lap_block_var',
        'jaw_edge_density', 'cheek_sharpness',  # v53: 性别ML特征
    ]
    for k in _default_face_keys:
        if k not in feats:
            feats[k] = 0.0
    for ch in ['B', 'G', 'R']:
        for sf in ['_mean', '_std']:
            k = f'skin_{ch}{sf}'
            if k not in feats:
                feats[k] = 0.0
    return feats


def predict_calibrated_score(img: np.ndarray,
                             face_rect: tuple[int, int, int, int] | None = None) -> float | None:
    """v48: 使用 Ridge 回归模型预测校准后的颜值评分 (2-9)

    Returns:
        float [2, 9] 或 None (模型未加载时)
    """
    model = _load_beauty_model()
    if model is None:
        return None
    feats = _extract_32d_features(img, face_rect)
    feat_keys = model['feat_keys']
    X_raw = np.array([[feats.get(k, 0.0) for k in feat_keys]], dtype=np.float64)
    X_scaled = model['scaler'].transform(X_raw)
    X_poly = model['poly'].transform(X_scaled)
    scut_pred = float(model['ridge'].predict(X_poly)[0])
    scut_pred = np.clip(scut_pred, 1.0, 5.0)
    # SCUT 1-5 → 2-9 映射 (同训练脚本的 tanh 非线性)
    x = (scut_pred - 3.0) / 1.5
    stretched = np.tanh(x * 1.2)
    our_score = 5.5 + 3.5 * stretched
    return round(float(np.clip(our_score, 2.0, 9.0)), 2)


def extract_face_roi_features(
    img: np.ndarray,
    face_rects: list[tuple[int, int, int, int]],
    _remove_bg: bool = True
) -> FaceFeatures:
    """v48: 从人脸 ROI 提取10维美学特征 (基于真实图像特征)

    使用 32 维真实视觉特征映射到 10 维美学特征，
    替代 v45 之前的随机模拟。
    """
    face_rect = face_rects[0] if face_rects else None
    feats32 = _extract_32d_features(img, face_rect)

    # 从 32 维特征推导 10 维 FaceFeatures
    # C1~C5 对应: symmetry, proportion, youth_index, uniqueness, harmony
    # 器官列对应: eye_beauty, nose_elegance, lip_charm, contour, skin_texture

    # 利用已有的 face_symmetry + 几何特征
    symmetry_val = round(feats32.get('face_symmetry', 0.5) * 10.0, 1)
    # 比例: 宽高比接近 0.75~0.85 为佳
    aspect = feats32.get('face_aspect', 0.75)
    proportion_val = round(10.0 - abs(aspect - 0.8) * 15.0, 1)
    proportion_val = max(0.5, min(9.5, proportion_val))
    # 年轻: 锐度高 + 饱和适中 + 对称好
    youth_val = round((feats32.get('face_sharpness', 1.0) / 5.0 +
                       feats32.get('face_symmetry', 0.5) +
                       (1 - abs(feats32.get('face_saturation', 0.5) - 0.4))) * 3.3, 1)
    youth_val = max(0.5, min(9.5, youth_val))
    # 独特: 偏离典型值的程度
    uniqueness_val = round(abs(aspect - 0.8) * 8.0 +
                           feats32.get('face_topbot_edge_ratio', 1.0) * 2.0 +
                           abs(feats32.get('warmth_ratio', 0.0)) * 5.0, 1)
    uniqueness_val = max(0.5, min(9.5, uniqueness_val))
    # 和谐: 对称+边缘密度适中
    harmony_val = round((feats32.get('face_symmetry', 0.5) +
                        (1 - abs(feats32.get('face_edge_density', 0.1) - 0.08) * 20)) * 5.0, 1)
    harmony_val = max(0.5, min(9.5, harmony_val))

    # 器官: 基于全局/肤色特征
    skin_quality = round(feats32.get('face_sharpness', 1.0) / 6.0 * 10.0, 1)
    skin_quality = max(0.5, min(9.5, skin_quality))
    contour_quality = round(feats32.get('face_edge_density', 0.1) * 40.0, 1)
    contour_quality = max(0.5, min(9.5, contour_quality))
    # 眼睛/鼻/唇: 用上半区边缘密度估算
    eye_ratio = feats32.get('face_topbot_edge_ratio', 1.2)
    eye_val = round(min(eye_ratio * 4.0, 9.5), 1)
    nose_val = round(5.0 + (feats32.get('face_aspect', 0.75) - 0.7) * 10.0, 1)
    nose_val = max(0.5, min(9.5, nose_val))
    lip_val = round(feats32.get('face_contrast', 0.3) * 12.0, 1)
    lip_val = max(0.5, min(9.5, lip_val))

    # 肤色标签 (v3: CNN MobileNetV3-Small 3分类 + R/B 后处理 → 5小类)
    # 优先使用 CNN 端到端推理, 回退到 ITA° 物理公式
    skin_r = feats32.get('skin_R_mean', 0.5)
    skin_b = feats32.get('skin_B_mean', 0.5)
    try:
        fx, fy, fw, fh = face_rects[0]
        fx_c = max(0, fx); fy_c = max(0, fy)
        fw_c = min(fw, img.shape[1] - fx_c); fh_c = min(fh, img.shape[0] - fy_c)
        face_roi_rgb = img[fy_c:fy_c+fh_c, fx_c:fx_c+fw_c]
        face_roi_bgr = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2BGR)

        # v3: CNN 优先, 内部有完整的 ITA° 回退链
        tone = predict_skin_tone_cnn(face_roi_bgr, skin_r_mean=skin_r, skin_b_mean=skin_b)
    except Exception:
        # 最终降级: 纯 RGB 比值法
        r_b_ratio = skin_r / max(skin_b, 0.01)
        if r_b_ratio > 1.08:
            tone = '红润' if skin_r < 0.7 else '小麦'
        elif r_b_ratio > 0.9:
            tone = '自然'
        else:
            tone = '白皙'

    # v2: CNN 性别分类器 (MobileNetV3-Small, 91.4%, ROC-AUC 0.977)
    # 替代 v53 手工特征+LR方案, Domain Shift鲁棒性大幅提升
    gender_result = predict_gender_cnn(img, face_rects[0] if face_rects else None)
    gender = gender_result['gender']
    gender_confidence = gender_result['confidence']

    if face_rects:
        return FaceFeatures(
            symmetry=symmetry_val, proportion=proportion_val,
            skin_texture=skin_quality, contour=contour_quality,
            eye_beauty=eye_val, nose_elegance=nose_val,
            lip_charm=lip_val, youth_index=youth_val,
            uniqueness=uniqueness_val, harmony=harmony_val,
            skin_clarity=round(skin_quality * 0.7 + 2.0, 1),
            skin_tone_label=tone,
            gender=gender,
            gender_confidence=gender_confidence,
        )
    return FaceFeatures()


# ── 矩阵行/列标签 (v48 转置) ──
# 行(R1~R5) = 面部器官, 列(C1~C5) = 审美维度
_MATRIX_ROW_NAMES = ['R1 眼',   'R2 鼻',   'R3 唇',   'R4 轮廓', 'R5 肤质']
_MATRIX_COL_NAMES = ['C1 对称性', 'C2 比例',  'C3 年轻',  'C4 独特性', 'C5 和谐度']
# 行列在 FaceFeatures 中的字段映射
_MATRIX_ROW_FIELDS = ['eye_beauty', 'nose_elegance', 'lip_charm', 'contour', 'skin_texture']
_MATRIX_COL_FIELDS = ['symmetry', 'proportion', 'youth_index', 'uniqueness', 'harmony']


def features_to_matrix(feats: FaceFeatures, pref_raw: list[float] | None = None) -> np.ndarray:
    """v48: 将10维特征映射为 5×5 美学矩阵 (转置布局)

    行 (5个面部器官):  R1眼 R2鼻 R3唇 R4轮廓 R5肤质
    列 (5维审美标准): C1对称性 C2比例 C3年轻 C4独特性 C5和谐度

    M[i][j] = hill(器官[i]) × hill(审美维度[j]) × 10.0
    偏好权重 pref_raw 作用于对角元, 打破秩1使得 det(A) 非零
    """
    f_dict = feats.as_dict()

    # 提取行(器官)和列(审美维度)
    row_vals = np.array([f_dict.get(f, 0.5) for f in _MATRIX_ROW_FIELDS])
    col_vals = np.array([f_dict.get(f, 0.5) for f in _MATRIX_COL_FIELDS])

    # 构建 5×5 矩阵
    mat = np.zeros((5, 5))
    for i in range(5):
        hi = hill(float(row_vals[i]))
        for j in range(5):
            mat[i][j] = hi * hill(float(col_vals[j])) * 10.0

    # 应用偏好权重: 添加对角身份矩阵打破秩1 → det(A) 非零
    if pref_raw and len(pref_raw) >= 5:
        # outer product mat 是秩1, 通过添加独立的对角身份矩阵达到满秩
        # scale = 矩阵均值的固定比例
        id_scale = np.mean(np.abs(mat)) * 0.25
        diag_perturb = np.array(pref_raw[:5]) * id_scale
        mat = mat + np.diag(diag_perturb)

    return mat


def matrix_for_display(feats: FaceFeatures, pref_raw: list[float] | None = None) -> dict[str, Any]:
    """v48: 返回带行列标签的矩阵数据, 供 GUI 展示 (转置布局)

    Returns:
        {
            'row_names': [5],   # R1~R5 器官
            'col_names': [5],   # C1~C5 审美维度
            'matrix':    [[5×5]],
            'det':       float,
        }
    """
    mat = features_to_matrix(feats, pref_raw=pref_raw)
    det_val = float(np.linalg.det(mat))

    # 归一化用于显示 (不影响 det 计算)
    max_val = float(np.max(np.abs(mat)))
    M_display = (mat / max_val * 5.0).astype(np.float64) if max_val > 0 else mat.astype(np.float64)

    return {
        'row_names': list(_MATRIX_ROW_NAMES),
        'col_names': list(_MATRIX_COL_NAMES),
        'matrix': M_display.round(4).tolist(),
        'det': round(det_val, 4),
        'max_val': round(float(np.max(M_display)), 2),
        'min_val': round(float(np.min(M_display)), 2),
    }


def features_quality(feats: FaceFeatures, pref_raw: list[float] | None = None) -> float:
    """计算特征质量得分 (0~10)"""
    f_vec = np.array(feats.to_list())
    
    # 基础质量: 均值 × Hill加权
    mu = np.mean(f_vec)
    sigma = np.std(f_vec)
    quality = mu * (1 - 0.3 * min(sigma / max(mu, 0.1), 1.0))
    
    # 偏好加权
    if pref_raw and len(pref_raw) >= 5:
        # 前5维对应偏好维度
        weighted = sum(f_vec[i] * pref_raw[i] for i in range(min(5, len(f_vec))))
        unweighted = sum(pref_raw[:5])
        if unweighted > 0:
            quality = quality * 0.5 + (weighted / unweighted) * 0.5
    
    return round(float(min(quality, 10.0)), 2)


def features_to_style(feats: FaceFeatures) -> dict[str, Any]:
    """v53: 性别感知风格推断 (男性/女性双套美学标签)"""
    f = feats
    is_male = (f.gender == 'male')
    styles: list[dict[str, Any]] = []

    # ── 男性风格 ──
    if is_male:
        if f.youth_index > 7.0 and f.skin_texture > 7.0:
            styles.append({'name': '阳光少年', 'purity': 0.9})
        if f.uniqueness > 7.5:
            styles.append({'name': '硬朗型男', 'purity': 0.85})
        if f.contour > 7.0 and f.harmony > 7.0:
            styles.append({'name': '儒雅绅士', 'purity': 0.8})
        if f.eye_beauty > 7.5 and f.nose_elegance > 7.0:
            styles.append({'name': '韩系俊朗', 'purity': 0.85})
        if f.uniqueness > 7.0 and f.symmetry > 6.5:
            styles.append({'name': '日系清新', 'purity': 0.75})
        if f.symmetry > 7.5 and f.proportion > 7.5:
            styles.append({'name': '硬朗雕塑', 'purity': 0.9})
    # ── 女性/未知 风格 ──
    else:
        if f.youth_index > 7.0 and f.skin_texture > 7.0:
            styles.append({'name': '甜美少女', 'purity': 0.9})
        if f.uniqueness > 7.5:
            styles.append({'name': '高级超模', 'purity': 0.85})
        if f.contour > 7.0 and f.harmony > 7.0:
            styles.append({'name': '古典东方', 'purity': 0.8})
        if f.eye_beauty > 7.5 and f.nose_elegance > 7.0:
            styles.append({'name': '韩系精致', 'purity': 0.85})
        if f.uniqueness > 7.0 and f.symmetry > 6.5:
            styles.append({'name': '日系自然', 'purity': 0.75})
        if f.symmetry > 7.5 and f.proportion > 7.5:
            styles.append({'name': '建模标杆', 'purity': 0.9})

    # ── 中性风格 (男女共用) ──
    if f.skin_clarity > 7.0:
        styles.append({'name': '瓷肌白肤', 'purity': 0.7})

    if not styles:
        styles.append({'name': '均衡风格', 'purity': 0.5})

    # 取纯度最高的
    styles.sort(key=lambda s: s['purity'], reverse=True)
    primary = styles[0]

    return {
        'primary_style': primary['name'],
        'purity': round(primary['purity'], 2),
        'positive_ratio': round(sum(1 for s in styles if s['purity'] > 0.6) / max(len(styles), 1), 2),
        'gender': f.gender,
    }


# ═══════════════════════════════════════════════
#  评分计算
# ═══════════════════════════════════════════════

def raw_to_beauty(det_val: float, quality: float = 5.0) -> float:
    """v48: 从行列式 + 质量分计算颜值基础评分
    
    核心公式 (v48, SCUT-FBP5500 校准):
    det_contrib = min(|det(A)| × 2, DET_CAP)    # DET_CAP 防止极端值
    x = det_contrib + quality × Q_WEIGHT          # 品质感知权重(降低)
    base = VMAX × x / (x + K_HALF) + CALIB_OFFSET  # Hill饱和 + 底线=2.0
    
    v48变动: VMAX 5.5→7.0, K_HALF 8→16.3, Q_WEIGHT 22→5, CALIB_OFFSET -0.2→2.0
    目标: 覆盖[2,9]区间, 中心5.5, Pearson r=0.63 on SCUT-FBP5500
    """
    det_contrib = min(abs(det_val) * 2.0, DET_CAP)
    x = det_contrib + quality * Q_WEIGHT
    base = VMAX * x / (x + K_HALF) + CALIB_OFFSET
    return round(max(0.0, base), 2)


def skin_clarity_bonus(skin_clarity: float, pref_skin: float = 1.0) -> float:
    """v44: 肤质白净透亮加分 (Sigmoid 非线性)
    
    公式: bonus = B_MAX / (1 + exp(-k * (s/10 - x0))) * pref_skin
    参数: B_MAX=0.5, k=3.0, x0=0.50
    
    s/10 归一化到 [0,1], x0=0.50 即 s=5 时 bonus=0.5*B_MAX=0.25 (中性)
    s=7.5 时 bonus≈0.42*pref_skin (高分触发)
    s=2.5 时 bonus≈0.08*pref_skin (低分抑制)
    """
    s_norm = skin_clarity / 10.0
    sig = 1.0 / (1.0 + math.exp(-_SKIN_BONUS_K * (s_norm - _SKIN_BONUS_X0)))
    bonus = _SKIN_B_MAX * sig * pref_skin
    return round(bonus, 3)


def skin_tone_affinity_bonus(skin_tone_label: str, pref_tone: float = 0.0) -> float:
    """肤色偏好亲和加分
    
    pref_tone > 0: 偏好白皙 → 给白皙肤色加分
    pref_tone < 0: 偏好深色/小麦 → 给深肤色加分
    """
    tone_scores = {
        '白皙': 1.0,
        '红润': 0.8,
        '自然': 0.5,
        '橄榄': 0.3,
        '小麦': 0.1,
    }
    base = tone_scores.get(skin_tone_label, 0.5)
    
    if pref_tone > 0:
        # 偏好浅肤色
        bonus = base * pref_tone * 0.8
    elif pref_tone < 0:
        # 偏好深肤色
        bonus = (1 - base) * abs(pref_tone) * 0.8
    else:
        bonus = 0.0
    
    return round(bonus, 2)


def blemish_penalty(blemish_score: float, severity_mult: float = 1.0) -> float:
    """v50: 痘/色斑瑕疵减分 (SCUT-FBP5500 2300张校准)

    校准分布: P50=1.7, P90=3.9, P95=4.5, >4.0仅8.5%
    
    瑕疵评分 → 减分量:
    - ≤2.5 分 (洁净, ~61%): 不减分
    - 2.5~4.0 (少量, ~30%): 线性减 0→0.3
    - 4.0~6.0 (明显, ~8%): 加速减 0.3→1.0
    - 6.0+   (严重, ~0.4%): 大幅减 1.0→2.0

    Args:
        blemish_score: 0-10 瑕疵评分 (detect_skin_blemishes 输出)
        severity_mult: 严重度乘数 (偏好预设可调整, 默认1.0)

    Returns: 减分 (≥0, 从总分中扣除)
    """
    if blemish_score <= 2.5:
        return 0.0
    elif blemish_score <= 4.0:
        penalty = (blemish_score - 2.5) / 1.5 * 0.3
    elif blemish_score <= 6.0:
        t = (blemish_score - 4.0) / 2.0
        penalty = 0.3 + t * t * 0.7
    else:
        t = min((blemish_score - 6.0) / 4.0, 1.0)
        penalty = 1.0 + t * 1.0
    return round(penalty * severity_mult, 2)


def compute_geo_dimensions(
    img: np.ndarray,
    face_rects: list[tuple[int, int, int, int]],
    landmarks: np.ndarray | None = None,
) -> GeoDimensions:
    """v53.5: 基于 MediaPipe 关键点计算解剖学面宽比 (检测框兜底)

    核心: 优先使用 MediaPipe 468 关键点计算真实颧弓宽/面部高度比
    若关键点不可用 → 回退到检测框 w/h
    [0.72, 0.75] 平坦区满分, 两端 S×deviation 线性衰减
    bonus 可正可负, 供 geo_clarity_bonus 对称Sigmoid使用.
    """
    if not face_rects:
        return GeoDimensions(available=False)
    
    x, y, w, h = face_rects[0]
    ih, iw = img.shape[:2]
    
    # v53.5: 优先使用关键点计算解剖学面宽比
    landmark_ratio: float | None = None
    if landmarks is not None:
        from image_utils import compute_face_ratio_from_landmarks
        landmark_ratio = compute_face_ratio_from_landmarks(landmarks)
    
    if landmark_ratio is not None:
        face_ratio = landmark_ratio
    else:
        # 回退: 检测框宽高比 (不够精确但兼容性好)
        face_ratio = round(w / max(h, 1), 3)
    
    # v53.5: 平坦区 + 双斜率双向衰减 (解剖学面宽比)
    # 校准: 100张SCUT回归 r86=0.8312*r_mp+0.0322, R²=0.52
    # 86点最优[0.72,0.75]→MediaPipe映射[0.83,0.86]
    # 回退: 检测框 ratio 用 [0.72, 0.75] (SCUT 86点校准)
    if landmark_ratio is not None:
        # 回归映射: low=(0.72-0.0322)/0.8312≈0.83, high=(0.75-0.0322)/0.8312≈0.86
        _OPTIMAL_LOW = 0.83
        _OPTIMAL_HIGH = 0.86
    else:
        # 检测框回退: SCUT 86点校准值
        _OPTIMAL_LOW = 0.72
        _OPTIMAL_HIGH = 0.75
    _MAX_BOOST = 0.30
    if face_ratio < _OPTIMAL_LOW:
        deviation = _OPTIMAL_LOW - face_ratio
    elif face_ratio > _OPTIMAL_HIGH:
        deviation = face_ratio - _OPTIMAL_HIGH
    else:
        deviation = 0.0
    ratio_score = _MAX_BOOST - _GEO_PENALTY_SLOPE * deviation
    bonus = round(ratio_score, 2)
    
    # 居中偏移计算 (仅用于 symmetry_index, 不参与评分)
    center_x = x + w / 2
    center_y = y + h / 2
    x_offset = abs(center_x / iw - 0.5) * 2
    y_offset = abs(center_y / ih - 0.5) * 2
    
    # 下颌角: 宽脸→大角度 (不变)
    jaw_angle = round(110 + (1.0 - min(max(face_ratio, 0.55), 0.95)) * 75, 1)
    
    # 对称度 (保留, 用于 gender_inference 性别推断)
    symmetry_index = round(0.92 - (x_offset + y_offset) * 0.05, 3)
    symmetry_index = max(0.78, min(0.98, symmetry_index))
    
    return GeoDimensions(
        available=True,
        eye_aspect_ratio=round(3.0 + (face_ratio - 0.75) * 2.0, 2),
        mouth_aspect_ratio=round(1.6 + (face_ratio - 0.75) * 1.5, 2),
        nose_mouth_ratio=round(1.1, 2),
        face_ratio=face_ratio,
        jaw_angle=jaw_angle,
        symmetry_index=symmetry_index,
        bonus=bonus,
    )


def geo_clarity_bonus(geo_dims: GeoDimensions, pref_weight: float = 1.0) -> float:
    """v53.5: 几何维度加减分 (对称Sigmoid, 解剖学面宽比 + S×deviation衰减)
    
    公式: bonus = GEO_WEIGHT * (2*sigmoid(k*ratio_score) - 1) * pref_weight
    ratio_score>0→加分, ratio_score<0→扣分, ratio_score=0→中性
    参数: GEO_WEIGHT=0.5 (config), k=4.0, penalty_slope=3.0
    ratio_score = MAX_BOOST(0.30) - penalty_slope * deviation
    face_ratio: 优先 MediaPipe 468关键点, 回退检测框 w/h
    
    平坦区 [0.72,0.75]: deviation=0 → ratio=+0.30 → geo≈+0.27 (满分)
    略偏 (0.67/0.80): deviation=0.05 → ratio=+0.15 → geo≈+0.14
    偏窄/偏宽 (0.62/0.85): deviation=0.10 → ratio=0.00 → geo≈0.00 (中性线)
    过窄/过宽 (0.57/0.90): deviation=0.15 → ratio=-0.15 → geo≈-0.14 (扣分)
    理论范围: [-0.5, +0.5], 占总分~5%, S=3增大跨度至~0.40
    """
    if not geo_dims.available:
        return 0.0
    geo_score = geo_dims.bonus
    # 对称Sigmoid: sig∈[0,1] → normalized∈[-1,1]
    sig = 1.0 / (1.0 + math.exp(-_GEO_BONUS_K * geo_score))
    normalized = 2.0 * sig - 1.0
    bonus = GEO_WEIGHT * normalized * pref_weight
    return round(bonus, 3)


# ═══════════════════════════════════════════════
#  等级评定
# ═══════════════════════════════════════════════

def get_grade(score: float) -> dict[str, Any]:
    """根据分数返回等级"""
    for g in GRADES:
        if score >= g['threshold']:
            return {'label': g['label'], 'color': g['color']}
    return {'label': 'D 普通', 'color': '#808080'}


# ═══════════════════════════════════════════════
#  v48: 缺陷判定 (<2分)  & 增值判定 (>9分)
# ═══════════════════════════════════════════════

# ── 缺陷知识库 ──
_DEFECT_DB: dict[str, list[dict[str, Any]]] = {
    'symmetry': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '面部严重不对称',
         'detail': '左右脸结构差异过大，可能影响第一印象与亲和力。',
         'advice': '可通过发型不对称修饰（偏分刘海）、化矫正眉形、或医美微调。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '面部对称性不足',
         'detail': '左右眼/眉/唇存在可察觉的不对称。',
         'advice': '化妆时注意眉形对齐、唇线对称描画。'},
    ],
    'proportion': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '面部比例失调',
         'detail': '三庭五眼比例偏离黄金分割较大，五官位置不协调。',
         'advice': '发型可调整视觉比例（增高颅顶/修饰额头），化妆用高光阴影微调。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '面部比例欠佳',
         'detail': '部分区域（额头/下巴/中庭）比例偏离理想值。',
         'advice': '通过刘海长度、眉峰位置、腮红打法可视觉改善。'},
    ],
    'youth_index': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '面部老化特征明显',
         'detail': '皮肤松弛、法令纹/鱼尾纹深度较大，整体呈现疲惫感。',
         'advice': '加强抗衰护理（VA/胜肽精华）、保证充足睡眠、考虑专业护肤方案。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '年轻指数偏低',
         'detail': '皮肤光泽度和紧致度有待提升，略显疲态。',
         'advice': '规律作息+补水保湿+防晒是改善年轻感的三驾马车。'},
    ],
    'uniqueness': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '面部辨识度极低',
         'detail': '五官特征过于普通，缺乏个人特色和记忆点。',
         'advice': '尝试标志性配饰/发型/妆容风格，打造个人视觉符号。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '面部辨识度偏低',
         'detail': '长相偏向大众化，缺少令人印象深刻的特点。',
         'advice': '强化某一优势特征（眼妆/唇色/眉形），形成视觉焦点。'},
    ],
    'harmony': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '五官和谐度差',
         'detail': '五官风格不统一（如柔和的眼+尖锐的鼻），整体协调感不足。',
         'advice': '统一妆容色调和风格方向，避免五官间冲突感。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '五官协调度不足',
         'detail': '部分五官在风格/比例上与整体不协调。',
         'advice': '化妆时优先统一眉-眼-唇的风格走向（柔美/锐利/自然）。'},
    ],
    'skin_texture': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '肤质问题突出',
         'detail': '粗糙感、毛孔粗大或瑕疵明显，影响整体精致度。',
         'advice': '建议水光/点阵等专业项目改善肤质基底，日常做好清洁+保湿。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '肤质有待改善',
         'detail': '皮肤细腻度和平滑度有提升空间。',
         'advice': '坚持温和清洁+保湿+防晒，每周1-2次面膜护理。'},
    ],
    'contour': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '面部轮廓线条不佳',
         'detail': '下颌线/颧骨/太阳穴区域线条不够流畅清晰。',
         'advice': '修容+高光可重塑轮廓感，发型遮挡也可改善视觉轮廓。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '面部轮廓欠流畅',
         'detail': '部分轮廓线条有突兀或凹陷感。',
         'advice': '化自然修容、选择适合脸型的发型可有效修饰。'},
    ],
    'eye_beauty': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '眼型美感不足',
         'detail': '眼型/眼距/大小偏离主流审美，眼部为面部焦点却缺少吸引力。',
         'advice': '眼妆是改造空间最大的区域：双眼皮贴+眼线+睫毛+美瞳可大幅提升。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '眼部魅力偏弱',
         'detail': '眼型有提升空间，眼部未成为面部优势焦点。',
         'advice': '学习适合自己眼型的眼妆技法，突出眼部神采。'},
    ],
    'nose_elegance': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '鼻型美学缺陷',
         'detail': '鼻梁/鼻翼/鼻头比例偏离理想范围，影响面部立体感。',
         'advice': '鼻影修容可极大改善鼻型视觉效果（鼻梁高光+鼻翼阴影）。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '鼻型不够精致',
         'detail': '鼻部某些比例可进一步优化。',
         'advice': '轻扫鼻影+鼻头高光，瞬间提升立体精致感。'},
    ],
    'lip_charm': [
        {'threshold': 2.5, 'severity': '严重',
         'label': '唇形魅力不足',
         'detail': '唇形/唇厚/唇色缺乏吸引力，唇部是表情的重要载体。',
         'advice': '唇线笔+唇膏可重塑唇形，选择提气色的唇色是关键。'},
        {'threshold': 4.0, 'severity': '明显',
         'label': '唇部美感偏弱',
         'detail': '唇形有改善空间，唇部未成为加分项。',
         'advice': '用唇线勾勒理想唇形，叠加水光唇釉增加丰盈感。'},
    ],
}

# 维度中文名映射
_DIM_CN: dict[str, str] = {
    'symmetry': '对称度', 'proportion': '比例协调', 'youth_index': '年轻指数',
    'uniqueness': '独特气质', 'harmony': '五官和谐', 'skin_texture': '皮肤质感',
    'contour': '轮廓流畅', 'eye_beauty': '眼睛美感', 'nose_elegance': '鼻型优雅',
    'lip_charm': '唇形魅力',
}

# ── 增值判定知识库 ──
_BONUS_DB: dict[str, dict[str, str]] = {
    'symmetry': {
        'label': '镜像对称之美',
        'detail': '面部左右高度对称，这是最稀有且跨文化最一致的审美标准。科学研究表明高度对称的脸在潜意识中传递健康基因信号。',
    },
    'proportion': {
        'label': '黄金比例架构',
        'detail': '三庭五眼比例极其接近黄金分割(1.618)，这种结构美感是历代艺术家追求的理想模板。',
    },
    'youth_index': {
        'label': '逆龄生长特质',
        'detail': '面部呈现显著的幼态持续特征(饱满苹果肌/紧致轮廓/明亮眼神)，是抗衰基因的外在表达。',
    },
    'uniqueness': {
        'label': '超模级辨识度',
        'detail': '这张脸拥有极高的视觉辨识度——并非"标准美"而是"让人过目不忘"。这是T台超模和顶级演员的稀缺特质。',
    },
    'harmony': {
        'label': '浑然天成的协调感',
        'detail': '五官风格高度统一，每个元素都恰到好处地服务于整体美感，不存在任何"出戏"的冲突。',
    },
    'skin_texture': {
        'label': '瓷肌级别肤质',
        'detail': '皮肤细腻度/光泽度/均匀度达到顶级水平，如同精修过的质感，这是高级感的重要基底。',
    },
    'contour': {
        'label': '雕塑级轮廓线条',
        'detail': '面部轮廓如雕塑般流畅利落，从下颌线到颧骨的过渡近乎完美，光影效果极佳。',
    },
    'eye_beauty': {
        'label': '摄人心魄的眼眸',
        'detail': '眼型/眼距/神采均属上乘，眼睛是这张脸最强大的叙事中心，一顾倾城。',
    },
    'nose_elegance': {
        'label': '精雕细琢的鼻型',
        'detail': '鼻梁挺拔度与鼻翼比例堪称范本，为面部提供了极佳的立体骨架支撑。',
    },
    'lip_charm': {
        'label': '完美唇形比例',
        'detail': '唇形/厚度/弧度高度协调，微笑时唇角的弧度尤为动人。',
    },
}


def diagnose_defects(feats: FaceFeatures, total_score: float,
                     pref_name: str = '均衡审美') -> list[dict[str, Any]]:
    """v48: 缺陷判定 — 当总分 < 2.0 时分析具体缺陷维度

    检查所有10维特征，低于阈值的维度标记为缺陷，
    按严重程度排序，严重缺陷优先展示。

    Returns:
        list[dict]: 缺陷列表, 为空列表时表示无显著缺陷
    """
    defects: list[dict[str, Any]] = []

    # v50: 瑕疵检测校准阈值 (SCUT P50=1.7, P90=3.9, P95=4.5)
    if feats.blemish_score > 2.5:
        if feats.blemish_score >= 6.0:
            sev, label = '严重', '面部瑕疵较多'
            detail = '检测到较多痘印/色斑，影响肤质洁净度和整体精致感。'
            advice = '建议配合皮肤科治疗痘印，日常做好防晒避免色斑加重，可配合遮瑕产品快速改善视觉效果。'
        elif feats.blemish_score >= 4.0:
            sev, label = '明显', '面部存在瑕疵'
            detail = '检测到痘印或色斑，虽然不严重但影响肤质均匀度。'
            advice = '坚持温和清洁+消炎精华，痘印可配合淡化精华(VC/烟酰胺)，色斑加强防晒。'
        else:
            sev, label = '轻微', '面部偶见瑕疵'
            detail = '检测到少量痘印或色斑，问题轻微。'
            advice = '日常做好清洁护理，注意饮食和作息即可改善。'
        defects.append({
            'dimension': 'blemish',
            'dim_name': '面部瑕疵',
            'value': round(feats.blemish_score, 1),
            'severity': sev,
            'label': label,
            'detail': detail,
            'advice': advice,
            'severity_score': round(feats.blemish_score / 10.0, 3),
        })

    if total_score >= 2.0 and not defects:
        return []

    f_dict = feats.as_dict()
    pref_raw = PREFERENCE_PRESETS.get(pref_name, [1, 1, 1, 1, 1])

    # 10个维度: C1~C5 审美维度 + 5个器官维度
    dim_names = ['symmetry', 'proportion', 'youth_index', 'uniqueness', 'harmony',
                 'skin_texture', 'contour', 'eye_beauty', 'nose_elegance', 'lip_charm']
    # 对应的偏好权重 (审美维度用pref_raw, 器官维度用默认1.0)
    dim_weights = {
        'symmetry': pref_raw[0], 'proportion': pref_raw[1],
        'youth_index': pref_raw[2], 'uniqueness': pref_raw[3],
        'harmony': pref_raw[4],
    }

    for dim_key in dim_names:
        val = f_dict.get(dim_key, 5.0)
        if dim_key not in _DEFECT_DB:
            continue
        # 检查阈值规则
        matched = None
        for rule in _DEFECT_DB[dim_key]:
            if val <= rule['threshold']:
                if matched is None or rule['threshold'] < matched['threshold']:
                    matched = rule
        if matched:
            w = dim_weights.get(dim_key, 1.0)
            # 严重度得分 = 权重 × (1 - val/10) → 值越大越需要关注
            severity_score = round(w * (1.0 - val / 10.0), 3)
            defects.append({
                'dimension': dim_key,
                'dim_name': _DIM_CN.get(dim_key, dim_key),
                'value': round(val, 1),
                'severity': matched['severity'],
                'label': matched['label'],
                'detail': matched['detail'],
                'advice': matched['advice'],
                'severity_score': severity_score,
            })

    # 按 severity_score 降序排序（越需要关注的越靠前）
    defects.sort(key=lambda d: d['severity_score'], reverse=True)

    # 至少输出3条（不足时用通用建议填充）
    if len(defects) < 3:
        remaining = 3 - len(defects)
        generic_defects = [
            {'dimension': '_lighting', 'dim_name': '光照/环境', 'value': 0,
             'severity': '提示', 'label': '图片光线欠佳',
             'detail': '不均匀或偏暗的光线可能导致面部特征被误判为缺陷。',
             'advice': '建议在自然均匀光线下重新拍摄/上传照片，获得更准确的分析。',
             'severity_score': 0.1},
            {'dimension': '_angle', 'dim_name': '拍摄角度', 'value': 0,
             'severity': '提示', 'label': '拍摄角度可能影响判断',
             'detail': '非正面/俯仰角度可能导致对称性和比例分析偏差。',
             'advice': '正面平视角度拍摄可获得最准确的美学分析结果。',
             'severity_score': 0.1},
            {'dimension': '_resolution', 'dim_name': '图像质量', 'value': 0,
             'severity': '提示', 'label': '图像清晰度可能不足',
             'detail': '低分辨率或模糊的照片会降低特征提取精度。',
             'advice': '使用高清照片（建议500×500像素以上）可获得更准确的分析。',
             'severity_score': 0.1},
        ]
        defects.extend(generic_defects[:remaining])

    return defects[:5]  # 最多展示5条


def diagnose_bonuses(feats: FaceFeatures, total_score: float,
                     _pref_name: str = '均衡审美') -> list[dict[str, Any]]:
    """v48: 增值判定 — 当总分 > 9.0 时分析卓越维度

    找出哪些维度达到了"出类拔萃"水平，解释为何超越了常规美的天花板。

    Returns:
        list[dict]: 增值维度列表
    """
    if total_score <= 9.0:
        return []

    bonuses: list[dict[str, Any]] = []
    f_dict = feats.as_dict()
    dim_names = ['symmetry', 'proportion', 'youth_index', 'uniqueness', 'harmony',
                 'skin_texture', 'contour', 'eye_beauty', 'nose_elegance', 'lip_charm']

    for dim_key in dim_names:
        val = f_dict.get(dim_key, 5.0)
        if val >= 8.5 and dim_key in _BONUS_DB:
            b = _BONUS_DB[dim_key]
            bonuses.append({
                'dimension': dim_key,
                'dim_name': _DIM_CN.get(dim_key, dim_key),
                'value': round(val, 1),
                'label': b['label'],
                'detail': b['detail'],
            })

    # 按值降序（最优维度排最前）
    bonuses.sort(key=lambda d: d['value'], reverse=True)

    if not bonuses:
        # 极端评分但无单维度超8.5 → 综合和谐效应
        bonuses.append({
            'dimension': '_synergy', 'dim_name': '整体协调',
            'value': 9.0,
            'label': '神级协调共振',
            'detail': '虽然没有单一维度达到极值，但所有维度的组合产生了1+1>2的协同效应——这是最罕见的"浑然天成"之美。',
        })

    return bonuses[:4]  # 最多展示4条


# ═══════════════════════════════════════════════
#  化妆模拟
# ═══════════════════════════════════════════════

def simulate_makeup(base_score: float, delta: float = 1.0) -> dict[str, Any]:
    """模拟化妆效果
    
    delta ∈ [0, 2.0], 1.0=素颜, >1.0=化妆加强
    线性插值: score' = base + (10 - base) × tanh(delta - 1) × 0.5
    """
    effect = math.tanh(delta - 1.0) * 0.5
    new_score = base_score + (10.0 - base_score) * max(effect, -0.3)
    new_score = max(2.0, min(10.0, new_score))
    
    return {
        'base_score': base_score,
        'makeup_delta': round(delta, 2),
        'simulated_score': round(new_score, 2),
        'effect_label': _makeup_label(delta),
    }


def _makeup_label(delta: float) -> str:
    if delta <= 0.3:
        return '极淡裸妆'
    elif delta <= 0.7:
        return '轻薄淡妆'
    elif delta <= 1.1:
        return '自然裸感'
    elif delta <= 1.3:
        return '精致日常'
    elif delta <= 1.6:
        return '韩系水光'
    elif delta <= 1.9:
        return '全妆上镜'
    else:
        return '浓妆大片'


# ═══════════════════════════════════════════════
#  偏好对比
# ═══════════════════════════════════════════════

def compute_all_preference_scores(feats: FaceFeatures, geo_dims: GeoDimensions | None = None) -> list[dict[str, Any]]:
    """v44: 在同一张脸上用所有11种审美偏好计算得分
    
    完整评分链: base + skin_bonus + tone_bonus + geo_bonus → clamp(10.0)
    v44 fix: 每种偏好独立计算矩阵行列式, 避免 det(A) 恒零
    """
    results: list[dict[str, Any]] = []
    
    for pref_name, pref_weights in PREFERENCE_PRESETS.items():
        a_mat = features_to_matrix(feats, pref_raw=pref_weights)
        det_val = float(np.linalg.det(a_mat))
        quality = features_quality(feats, pref_raw=pref_weights)
        beauty = raw_to_beauty(det_val, quality=quality)
        
        skin_w = SKIN_CLARITY_WEIGHTS.get(pref_name, 1.0)
        tone_w = SKIN_TONE_WEIGHTS.get(pref_name, 0.0)
        geo_w = GEO_CLARITY_WEIGHTS.get(pref_name, 1.0)
        
        skin_b = skin_clarity_bonus(feats.skin_clarity, pref_skin=skin_w)
        tone_b = skin_tone_affinity_bonus(feats.skin_tone_label, tone_w)
        geo_b = geo_clarity_bonus(geo_dims, pref_weight=geo_w) if geo_dims and geo_dims.available else 0.0
        
        total = round(min(beauty + skin_b + tone_b + geo_b, 10.0), 2)
        
        results.append({
            'pref_name': pref_name,
            'score': total,
            'det_score': round(beauty, 2),
            'skin_bonus': skin_b,
            'tone_bonus': tone_b,
            'geo_bonus': round(geo_b, 3),
        })
    
    results.sort(key=lambda r: r['score'], reverse=True)
    return results


# ═══════════════════════════════════════════════
#  环境稀释补偿 & 人脸遮罩 (v38)
# ═══════════════════════════════════════════════

def _compute_env_dilution(img: np.ndarray) -> float:  # pyright: ignore[reportUnusedFunction]
    """v13: 检测面部与背景颜色接近程度，返回特征稀释补偿因子。

    当面与背景颜色过于接近时, 边缘/纹理/对比度特征会被背景像素稀释,
    导致评分偏低。此函数计算面-背景欧氏颜色距离, 返回补偿系数。

    Returns:
        float: [1.0, 1.30], 1.0 = 面/背景分明无需补偿, >1.0 = 需放大特征信号
    """
    h, w = img.shape[:2]
    if h < 100 or w < 100:
        return 1.0

    # 中心区域（面部）vs 边缘区域（背景）
    center = img[h//4:3*h//4, w//4:3*w//4]
    top_strip = img[0:h//8, :]
    bot_strip = img[7*h//8:h, :]
    left_strip = img[:, 0:w//8]
    right_strip = img[:, 7*w//8:w]

    center_mean = np.mean(center, axis=(0, 1))
    bg_regions = [top_strip, bot_strip, left_strip, right_strip]
    bg_means = [np.mean(r, axis=(0, 1)) for r in bg_regions if r.size > 0]

    if not bg_means:
        return 1.0

    avg_bg = np.mean(bg_means, axis=0)
    color_dist = np.linalg.norm(center_mean - avg_bg)

    # 距离 < 40 → 高度接近, 补偿 1.30
    # 距离 > 80 → 无需补偿
    compensation = 1.0 + max(0, 0.30 * (1 - min(color_dist / 80.0, 1.0)))
    return float(compensation)


def _apply_face_mask(face_crop: np.ndarray) -> np.ndarray:  # pyright: ignore[reportUnusedFunction]
    """对人脸ROI应用椭圆遮罩，去除四角背景/衣物/头发区域。

    算法:
      1. 在矩形ROI内创建椭圆遮罩 (轴长=宽高×0.40/0.43, 覆盖核心面部)
      2. 对遮罩高斯模糊 (σ=10) 实现羽化过渡, 避免硬边伪影
      3. 遮罩外用椭圆内肤色均值填充
    """
    h, w = face_crop.shape[:2]
    if h < 30 or w < 30:
        return face_crop

    # 椭圆遮罩
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    axes = (int(w * 0.40), int(h * 0.43))
    _ = cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

    # 高斯羽化
    mask_f = cv2.GaussianBlur(mask, (21, 21), 10).astype(float) / 255.0

    # 填充色 = 椭圆内肤色均值
    mask_bool = mask > 128
    if np.sum(mask_bool) > 0:
        fill_color = np.mean(face_crop[mask_bool], axis=0)
    else:
        fill_color = np.mean(face_crop, axis=(0, 1))

    mask_3ch = np.expand_dims(mask_f, axis=2)
    result = face_crop.astype(float) * mask_3ch + fill_color * (1 - mask_3ch)
    return result.astype(np.uint8)


# ═══════════════════════════════════════════════
#  v38: 偏好驱动的颜值提升建议 (v35)
# ═══════════════════════════════════════════════

_ADVICE_DB = {
    'symmetry': [
        ('high', '面部对称度良好，继续保持日常体态平衡。'),
        ('medium', '可通过表情管理训练让面部更对称。'),
        ('low', '尝试发型修饰（侧分刘海）来视觉平衡面部不对称。'),
    ],
    'proportion': [
        ('high', '三庭五眼比例优秀，黄金分割感的天然优势。'),
        ('medium', '可通过眉形调整/发型比例来微调面部视觉比例。'),
        ('low', '建议通过化妆（高光/阴影）来优化面部比例感知。'),
    ],
    'youth_index': [
        ('high', '面部年轻指数很高，保持防晒+保湿日常护理。'),
        ('medium', '保持规律作息和补水可延缓皮肤老化迹象。'),
        ('low', '加强精华/眼霜护理 + 充足睡眠可改善皮肤状态。'),
    ],
    'uniqueness': [
        ('high', '你拥有很高的面部辨识度，这是超模级别的特质。'),
        ('medium', '适当的个人风格可以进一步提升辨识度。'),
        ('low', '找到你的标志性特征（如眼睛/嘴唇），放大它。'),
    ],
    'harmony': [
        ('high', '五官和谐度高，整体给人舒服自然的感觉。'),
        ('medium', '整体协调感不错，可微调单一元素来进一步提升。'),
        ('low', '建议从整体色调统一（发色/眉色/唇色）开始优化。'),
    ],
}


def generate_beauty_advice(feats: FaceFeatures, pref_name: str, top_n: int = 3) -> list[dict[str, Any]]:
    """v35: 偏好驱动的颜值提升建议。

    策略: 找出'偏好权重高但实际得分低'的维度,
    按加权差距排序, 从建议库匹配偏好标签输出针对性建议.
    """
    pref_raw = PREFERENCE_PRESETS.get(pref_name, [1, 1, 1, 1, 1])
    f_vec = feats.to_list()
    dim_map = ['symmetry', 'proportion', 'youth_index', 'uniqueness', 'harmony']

    # 计算加权差距: weight * (10 - score) / 10
    gaps: list[tuple[str, float, float, float]] = []
    for _i, (val, w, dim) in enumerate(zip(f_vec[:5], pref_raw[:5], dim_map)):
        gap = w * (10 - val) / 10
        gaps.append((dim, gap, val, w))

    gaps.sort(key=lambda x: x[1], reverse=True)

    advice_list: list[dict[str, Any]] = []
    for dim, gap, val, w in gaps[:top_n]:
        label = 'low' if val < 5 else 'medium' if val < 7.5 else 'high'
        if dim in _ADVICE_DB:
            for lvl, txt in _ADVICE_DB[dim]:
                if lvl == label:
                    advice_list.append({
                        'dimension': dim,
                        'score': round(val, 1),
                        'weight': round(w, 2),
                        'gap': round(gap, 2),
                        'level': label,
                        'text': txt,
                    })
                    break

    # 通用建议
    advice_list.append({
        'dimension': '_general',
        'score': 0,
        'weight': 0,
        'gap': 0,
        'level': 'all',
        'text': '保持每日面部清洁+保湿基础护理，规律作息（7~8小时睡眠）促进皮肤自我修复。',
    })
    return advice_list


# ═══════════════════════════════════════════════
#  非人脸图片场景描述 (v38)
# ═══════════════════════════════════════════════

@dataclass
class SceneDescription:
    """非人脸图片的场景描述"""
    category: str = "未知"
    sub_category: str = ""
    color_palette: str = ""
    edge_density: float = 0.0
    texture_level: str = ""
    brightness: float = 0.0
    warmth: str = ""
    natural_text: str = ""


def describe_non_face_image(img: np.ndarray) -> SceneDescription:
    """分析非人脸图片的场景特征"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img

    if w < 100 or h < 100:
        return SceneDescription(category='过小图片', natural_text='图片尺寸过小，无法分析。')

    # 亮度
    brightness = float(np.mean(gray)) / 255.0

    # 暖色调/冷色调
    if img.shape[2] == 3:
        r_mean = float(np.mean(img[:, :, 0]))
        b_mean = float(np.mean(img[:, :, 2]))
        if r_mean > b_mean * 1.15:
            warmth = '暖色调'
        elif b_mean > r_mean * 1.15:
            warmth = '冷色调'
        else:
            warmth = '中性色调'
    else:
        warmth = '中性色调'

    # 边缘密度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.sum(edges > 0)) / max(edges.size, 1)

    # 纹理
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture_level = '复杂/细节丰富' if lap_var > 200 else '平滑/简洁'

    # 颜色调色板
    if img.shape[2] == 3:
        r, g, b = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
        max_c = max(r, g, b)
        if max_c == r:
            color_palette = '暖色系'
        elif max_c == g:
            color_palette = '自然系'
        else:
            color_palette = '冷色系'
    else:
        color_palette = '灰阶'

    # 简单场景分类
    dominant_green = False
    dominant_blue = False
    if img.shape[2] == 3:
        dominant_green = np.mean(img[:, :, 1]) > np.mean(img[:, :, 0]) * 1.2 and np.mean(img[:, :, 1]) > 100
        dominant_blue = np.mean(img[:, :, 2]) > np.mean(img[:, :, 0]) * 1.2 and np.mean(img[:, :, 2]) > 100

    if dominant_green:
        category = '自然风景'
        sub_category = '植被/森林/田野'
    elif dominant_blue:
        category = '天空/水域/海岸'
        sub_category = '蓝色系自然'
    elif brightness > 0.85:
        category = '纯色/简约'
        sub_category = '纯色背景/极简构图'
    elif edge_density > 0.15:
        category = '城市/建筑'
        sub_category = '街道/建筑室内外'
    elif edge_density > 0.05:
        category = '生活场景'
        sub_category = '日常/室内/物品'
    else:
        category = '文档/平面'
        sub_category = '文字/图表/抽象设计'

    # 自然语言描述
    natural_text = (f'这是一张{category}构图的图片，整体呈{warmth}。'
                    f'画面以{color_palette}为主，细节{texture_level}。'
                    f'推测为一幅「{sub_category}」类图像')

    return SceneDescription(
        category=category,
        sub_category=sub_category,
        color_palette=color_palette,
        edge_density=round(edge_density, 3),
        texture_level=texture_level,
        brightness=round(brightness, 3),
        warmth=warmth,
        natural_text=natural_text,
    )


# ═══════════════════════════════════════════════
#  批量分析 (v38)
# ═══════════════════════════════════════════════

def batch_analyze_images(
    folder_path: str,
    progress_callback: Callable[[int, int, str, dict[str, Any] | None], None] | None = None,
    cancel_event: object | None = None,
    pref_name: str = '均衡审美',
    soft_detect: bool = True,
) -> list[dict[str, Any]]:
    """v44: 批量分析文件夹中所有图片, 返回结果字典列表.

    Args:
        folder_path: 图片文件夹路径
        progress_callback: 可选, 签名为 callback(idx, total, fname, result_or_none)
        cancel_event:   可选, threading.Event, set() 后中断遍历
        pref_name:      审美偏好名称
        soft_detect:    是否使用软检测 (v44: 默认True, 硬检测+软兜底)

    Returns:
        [{filename, has_face, face_count, score, grade, skin_tone, ...}, ...]
    """
    import os as _os
    from image_utils import (
        load_and_normalize_image, detect_face, detect_face_soft,
        precheck_texture_quality, classify_skin_tone,
        detect_skin_blemishes, extract_face_landmarks,
    )

    supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    files = sorted([
        f for f in _os.listdir(folder_path)
        if _os.path.splitext(f)[1].lower() in supported_exts
        and _os.path.isfile(_os.path.join(folder_path, f))
    ])

    results: list[dict[str, Any]] = []
    total = len(files)
    pref_weights = PREFERENCE_PRESETS.get(pref_name, [1, 1, 1, 1, 1])

    for idx, fname in enumerate(files):
        if cancel_event and cancel_event.is_set():
            break

        filepath = _os.path.join(folder_path, fname)
        try:
            img = load_and_normalize_image(filepath, source_type='path')
        except Exception:
            if progress_callback:
                progress_callback(idx, total, fname, None)
            continue

        # 纹理预检
        tex_result = precheck_texture_quality(img)

        # v44: 硬检测优先, 软检测兜底
        face_rects = detect_face(img)
        if soft_detect and len(face_rects) == 0:
            soft_result = detect_face_soft(img)
            face_rects = soft_result['candidates']

        has_face = len(face_rects) > 0

        if has_face:
            # 肤色分类
            roi_bgr = cv2.cvtColor(img[face_rects[0][1]:face_rects[0][1]+face_rects[0][3],
                                       face_rects[0][0]:face_rects[0][0]+face_rects[0][2]],
                                   cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
            skin_tone = classify_skin_tone(roi_bgr)

            # v49: 瑕疵检测 (痘印/色斑)
            blemish_result = detect_skin_blemishes(roi_bgr)

            # 提取特征 + 评分
            feats = extract_face_roi_features(img, face_rects, _remove_bg=True)
            feats.blemish_score = blemish_result['blemish_score']
            # v53.5: 提取关键点用于解剖学面宽比
            landmarks_img = extract_face_landmarks(img, face_rects[0])
            geo_dims = compute_geo_dimensions(img, face_rects, landmarks=landmarks_img)
            a_mat = features_to_matrix(feats, pref_raw=pref_weights)
            det_val = float(np.linalg.det(a_mat))
            quality = features_quality(feats, pref_raw=pref_weights)
            beauty = raw_to_beauty(det_val, quality=quality)

            # v49 完整加减分链: 肤质 + 肤色 + 几何 - 瑕疵
            skin_w = SKIN_CLARITY_WEIGHTS.get(pref_name, 1.0)
            tone_w = SKIN_TONE_WEIGHTS.get(pref_name, 0.0)
            geo_w = GEO_CLARITY_WEIGHTS.get(pref_name, 1.0)
            skin_b = skin_clarity_bonus(feats.skin_clarity, pref_skin=skin_w)
            tone_b = skin_tone_affinity_bonus(feats.skin_tone_label, tone_w)
            geo_b = geo_clarity_bonus(geo_dims, pref_weight=geo_w)
            blemish_p = blemish_penalty(feats.blemish_score)
            total_score = round(min(beauty + skin_b + tone_b + geo_b - blemish_p, 10.0), 2)

            grade = get_grade(total_score)

            results.append({
                'filename': fname,
                'filepath': filepath,
                'has_face': True,
                'face_count': len(face_rects),
                'skin_tone': skin_tone,
                'feats': feats.as_dict(),
                'det_val': round(det_val, 2),
                'quality': round(quality, 2),
                'score': total_score,
                'grade': grade['label'],
                'tex_quality': tex_result['quality_label'],
                'skin_bonus': skin_b,
                'tone_bonus': tone_b,
                'geo_bonus': round(geo_b, 3),
                'blemish_score': blemish_result['blemish_score'],
                'blemish_details': blemish_result['details'],
                'blemish_penalty': blemish_p,
            })
        else:
            scene = describe_non_face_image(img)
            results.append({
                'filename': fname,
                'filepath': filepath,
                'has_face': False,
                'face_count': 0,
                'scene': scene,
                'tex_quality': tex_result['quality_label'],
                'score': 0,
                'grade': '无',
            })

        if progress_callback:
            progress_callback(idx, total, fname, results[-1])

    return results
