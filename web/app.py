"""
颜值矩阵分析系统 — Web API
基于 Flask，支持图片上传分析、偏好问卷
"""
import os
import sys
import traceback
import time

# 确保路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import cv2

from beauty_core import (
    FaceFeatures,
    GeoDimensions,
    PREFERENCE_PRESETS,
    SKIN_CLARITY_WEIGHTS,
    SKIN_TONE_WEIGHTS,
    extract_face_roi_features,
    features_to_matrix,
    features_quality,
    features_to_style,
    raw_to_beauty,
    skin_clarity_bonus,
    skin_tone_affinity_bonus,
    compute_geo_dimensions,
    geo_clarity_bonus,
    blemish_penalty,
    get_grade,
    simulate_makeup,
    compute_all_preference_scores,
)
from image_utils import (
    load_image,
    resize_for_analysis,
    remove_background,
    detect_faces,
    texture_precheck,
    estimate_skin_tone,
    detect_skin_blemishes,
)
from preference_questionnaire import (
    get_all_questions,
    compute_user_vector,
    match_best_preset,
)
from gender_inference import infer_gender_v5

app = Flask(__name__)
CORS(app)

# 上传文件存储
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ═══════════════════════════════════════════════
#  API 路由
# ═══════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'service': '颜值矩阵分析系统',
        'version': 'v53.1',
    })


@app.route('/api/preferences', methods=['GET'])
def get_preferences():
    """获取所有审美偏好预设"""
    prefs = []
    for name, weights in PREFERENCE_PRESETS.items():
        skin_w = SKIN_CLARITY_WEIGHTS.get(name, 1.0)
        tone_w = SKIN_TONE_WEIGHTS.get(name, 0.0)
        prefs.append({
            'name': name,
            'weights': weights,
            'skin_clarity_weight': skin_w,
            'skin_tone_weight': tone_w,
        })
    return jsonify({'preferences': prefs})


@app.route('/api/questionnaire', methods=['GET'])
def get_questionnaire():
    """获取问卷题目"""
    questions = get_all_questions()
    return jsonify({'questions': questions})


@app.route('/api/questionnaire/compute', methods=['POST'])
def compute_questionnaire():
    """提交问卷答案并计算审美偏好"""
    data = request.get_json() or {}
    answers = data.get('answers', [])
    
    if len(answers) != 12:
        return jsonify({'error': '需要12题全部答案'}), 400
    
    try:
        user_vector = compute_user_vector(answers)
        result = match_best_preset(user_vector)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/precheck', methods=['POST'])
def precheck_image():
    """图片纹理预检"""
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片'}), 400
    
    try:
        file = request.files['image']
        img_data = file.read()
        img = load_image(img_data)
        
        precheck_result = texture_precheck(img)
        
        return jsonify({
            'precheck': precheck_result,
            'image_shape': list(img.shape),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """核心分析接口
    
    参数:
        image: 图片文件
        pref_name: 审美偏好名称 (可选，默认'均衡审美')
        remove_bg: 是否去除背景 (可选, 'true'/'false', 默认'true')
        quick_mode: 快速模式 (可选, 'true'/'false', 默认'false')
        enhance_side: 侧脸增强 (可选)
        enhance_large: 大脸增强 (可选)
        
    返回:
        完整的颜值分析结果
    """
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片'}), 400
    
    t0 = time.time()
    
    try:
        file = request.files['image']
        img_data = file.read()
        img = load_image(img_data)
        
        # 参数解析
        pref_name = (request.form.get('pref_name') or '均衡审美').strip()
        if pref_name not in PREFERENCE_PRESETS:
            pref_name = '均衡审美'
        
        remove_bg = request.form.get('remove_bg', 'true').lower() == 'true'
        quick_mode = request.form.get('quick_mode', 'false').lower() == 'true'
        enhance_side = request.form.get('enhance_side', 'false').lower() == 'true'
        enhance_large = request.form.get('enhance_large', 'false').lower() == 'true'
        
        # 缩放
        img = resize_for_analysis(img, quick_mode=quick_mode)
        
        # 纹理预检
        precheck = texture_precheck(img)
        
        # 人脸检测
        face_rects = detect_faces(
            img,
            enhance_side=enhance_side,
            enhance_large=enhance_large
        )
        
        has_face = len(face_rects) > 0
        num_faces = len(face_rects)
        
        # 肤色估算
        skin_tone = estimate_skin_tone(img, face_rects) if has_face else '未知'
        
        if has_face:
            # ── 人脸流程 ──
            feats = extract_face_roi_features(img, face_rects, remove_bg=True)
            
            # v53.1: 瑕疵检测 (与桌面GUI一致, RGB→BGR转换)
            x, y, w, h = face_rects[0]
            face_crop_rgb = img[y:y+h, x:x+w]
            face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR) if len(face_crop_rgb.shape) == 3 else face_crop_rgb
            blemish_result = detect_skin_blemishes(face_crop_bgr)
            feats.blemish_score = blemish_result['blemish_score']
            
            a_mat = features_to_matrix(feats, pref_raw=PREFERENCE_PRESETS[pref_name])
            det_val = float(np.linalg.det(a_mat))
            quality = features_quality(feats, pref_raw=PREFERENCE_PRESETS[pref_name])
            beauty = raw_to_beauty(det_val, quality=quality)
            
            # 肤质白净透亮加分 (按偏好权重)
            skin_clarity_weight = SKIN_CLARITY_WEIGHTS.get(pref_name, 1.0)
            skin_bonus = skin_clarity_bonus(feats.skin_clarity, pref_skin=skin_clarity_weight)
            
            # 肤色偏好亲和加分 (按偏好权重)
            tone_weight = SKIN_TONE_WEIGHTS.get(pref_name, 0.0)
            tone_bonus = skin_tone_affinity_bonus(skin_tone, tone_weight)
            
            # 瑕疵扣分
            blemish_p = blemish_penalty(feats.blemish_score)
            
            # 几何美学分析 + 加分 (v53.5: MediaPipe 关键点解剖学面宽比)
            from image_utils import extract_face_landmarks
            landmarks_img = extract_face_landmarks(img, face_rects[0])
            geo_dims = compute_geo_dimensions(img, face_rects, landmarks=landmarks_img)
            geo_bonus = geo_clarity_bonus(geo_dims, pref_weight=skin_clarity_weight)
            
            # v53.1: 统一最终clamp (避免逐步clamp与最终clamp不一致)
            total_score = round(min(beauty + skin_bonus + tone_bonus + geo_bonus - blemish_p, 10.0), 2)
            
            # v53: ML性别分类器 (优先使用 feats.gender ML模型输出; 其次 fallback 到 infer_gender_v5)
            primary_gender = feats.gender if feats.gender != 'unknown' else None
            primary_conf = feats.gender_confidence if hasattr(feats, 'gender_confidence') else 0.0
            gender_detail = infer_gender_v5(
                face_features=feats.as_dict(),
                geo_dims=geo_dims.as_dict() if geo_dims.available else None,
                skin_texture_label=precheck.get('texture_label', 'medium'),
            )
            gender = {
                'gender': primary_gender or gender_detail.get('gender', 'unknown'),
                'gender_label': {'male': '男性', 'female': '女性', 'androgynous': '中性柔和', 'unknown': '未识别'}.get(primary_gender or gender_detail.get('gender', 'unknown'), '未识别'),
                'confidence': primary_conf if primary_gender else gender_detail.get('confidence', 0.5),
                'method': 'ml_logistic_cv' if primary_gender else ('geometric_heuristic_fallback' if gender_detail else 'unknown'),
            }
            
            # 风格推断
            style = features_to_style(feats)
            
            # 等级
            grade = get_grade(total_score)
            
            # 所有偏好得分对比
            all_pref_scores = compute_all_preference_scores(feats)
            
            result = {
                'type': 'face',
                'has_face': True,
                'num_faces': num_faces,
                'pref_name': pref_name,
                'skin_tone': skin_tone,
                'beauty_score': total_score,
                'grade': grade,
                'features': feats.as_dict(),
                'style': style,
                'gender': gender,
                'geo_dimensions': geo_dims.as_dict(),
                'matrix_det': round(det_val, 4),
                'quality': quality,
                'det_score': round(beauty, 2),
                'skin_bonus': skin_bonus,
                'tone_bonus': tone_bonus,
                'blemish_penalty': blemish_p,
                'blemish_score': feats.blemish_score,
                'blemish_details': blemish_result['details'],
                'geo_bonus': geo_bonus,
                'decomposition': {
                    'hill_formula': 'S = 10 × Hill(det(A), k=3, n=1.5) × (0.5 + 0.5×Hill(Q, k=5, n=2))',
                    'det_a': round(det_val, 4),
                    'det_score': round(raw_to_beauty(det_val, quality=quality), 2),
                    'quality': quality,
                    'skin_clarity_bonus': skin_bonus,
                    'skin_tone_bonus': tone_bonus,
                    'blemish_penalty': blemish_p,
                    'geo_bonus': geo_bonus,
                    'total': total_score,
                },
                'all_preference_scores': all_pref_scores,
                'precheck': precheck,
                'face_count': num_faces,
                'elapsed_ms': round((time.time() - t0) * 1000),
            }
        else:
            # 非人脸：场景分析
            result = {
                'type': 'scene',
                'has_face': False,
                'num_faces': 0,
                'precheck': precheck,
                'message': '未检测到人脸，这是一张风景/物品图片',
                'elapsed_ms': round((time.time() - t0) * 1000),
            }
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'elapsed_ms': round((time.time() - t0) * 1000),
        }), 500


@app.route('/api/makeup/simulate', methods=['POST'])
def simulate_makeup_api():
    """化妆模拟"""
    data = request.get_json() or {}
    base_score = data.get('base_score', 5.0)
    delta = data.get('delta', 1.0)
    
    if not isinstance(base_score, (int, float)) or not isinstance(delta, (int, float)):
        return jsonify({'error': '参数格式错误'}), 400
    
    delta = max(0.0, min(2.0, delta))
    result = simulate_makeup(base_score, delta)
    return jsonify(result)


@app.route('/api/compare', methods=['POST'])
def compare_all_preferences():
    """获取所有审美偏好的得分对比"""
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片'}), 400
    
    try:
        file = request.files['image']
        img_data = file.read()
        img = load_image(img_data)
        img = resize_for_analysis(img)
        
        face_rects = detect_faces(img)
        if not face_rects:
            return jsonify({'error': '未检测到人脸'}), 400
        
        feats = extract_face_roi_features(img, face_rects)
        scores = compute_all_preference_scores(feats)
        
        return jsonify({
            'has_face': True,
            'comparisons': scores,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    
    print(f'🚀 颜值矩阵分析系统 API 启动')
    print(f'   地址: http://0.0.0.0:{port}')
    print(f'   调试: {debug}')
    
    app.run(host='0.0.0.0', port=port, debug=debug)
