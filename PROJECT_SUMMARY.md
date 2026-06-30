# 颜值矩阵分析系统 — 项目摘要

**工作目录**: `d:\CodeBuddy\history\workspaces\20260609121427\`

## 项目目标

构建一个完整的颜值分析系统，包含：
- 颜值评分
- 性别判定
- 肤色分类（白皙/红润/自然/小麦/橄榄）

## 核心文件

| 文件 | 用途 |
|------|------|
| `beauty_core.py` | 核心推理引擎（人脸检测、特征提取、性别CNN、肤色CNN） |
| `image_utils.py` | 图像加载/归一化/人脸检测/ITA°肤色公式 |
| `beauty_gui_desktop.py` | 桌面GUI |
| `gender_inference.py` | 性别推理独立脚本 |

## 已完成模型

### 1. 性别 CNN（97% 准确率）

- 骨架：MobileNetV3-Small（二分类 male/female）
- 标签：FFHQ 真值标注（非伪标签）
- ONNX：`stats_output/gender_cnn_v2.onnx`（4.4 MB）
- 训练脚本：`train_gender_cnn.py`
- beauty_core 接口：`predict_gender_cnn(img, face_rect)`

### 2. 肤色 CNN v3（85.64% 准确率 vs ITA° 标签）

- 骨架：**EfficientNet-B0**（5.3M 参数）
- 分类：3 大类（深/中/浅），R/B 后处理 → 5 小类（白皙/红润/自然/小麦/橄榄）
- 标签：ITA° CIE Lab 伪标签（非真值，67%→86% 是 CNN 对 ITA° 的一致性）
- 训练改进：Mixup α=0.2 + Warmup 3 epochs + CosineAnnealing + 梯度裁剪
- 训练耗时：275 min（CPU）、37 epochs（Early Stopping）
- ONNX：`stats_output/skin_tone_cnn_v3.onnx`（18.3 MB）← **当前使用**
- 旧版：`stats_output/skin_tone_cnn.onnx`（v2, MobileNetV3-Small, 67%, 已淘汰）
- 训练脚本：`train_skin_tone_cnn_v3.py`（v3 版本）
- beauty_core 接口：`predict_skin_tone_cnn(face_bgr, skin_r_mean, skin_b_mean)`

### 3. 颜值评分模型

- 训练脚本：`train_beauty_model.py`
- 报告：`stats_output/training_report_full.json`

## 数据集

- **SCUT-FBP5500**：5500 张人脸（颜值评分，非肤色数据集）
  - 路径：`SCUT-FBP5500_v2/SCUT-FBP5500_v2/Images/`
  - 训练/测试分割：60/40，见 `train.txt` / `test.txt`
- **人脸裁剪缓存**：`stats_output/gender_cnn_cache/`（5500 张 face ROI，肤色训练用）
- **肤色标签缓存**：`stats_output/skin_tone_3class_labels.json`（ITA° 生成的 3 分类伪标签）
- **FFHQ**：仅 5 张（不足以扩展）
- **Reverse_Hill**：37 张（不足以扩展）

## 模型接口（beauty_core.py 中的关键函数）

```python
# 性别推理
predict_gender_cnn(img, face_rect) → {gender, confidence, prob_male, method}

# 肤色推理（v3 - 高效 Net-B0）
predict_skin_tone_cnn(face_bgr, skin_r_mean, skin_b_mean) → '白皙'|'红润'|'自然'|'小麦'|'橄榄'

# 完整特征提取（在 extract_face_roi_features 中调用上述两个函数）
```

## 技术栈

- PyTorch + ONNX Runtime（推理）
- OpenCV（人脸检测 + 图像处理）
- sklearn（评估指标）

## 已知限制

- 肤色标签是 ITA° 伪标签，非人工标注真值
- SCUT-FBP5500 是亚洲人为主的颜值数据集，肤色多样性有限
- Windows CPU 训练（无 GPU），训练耗时较长
- 人脸检测用 Haar Cascade（非 DNN），小脸/侧脸可能漏检

---

*生成时间：2026-06-30，用于新会话快速恢复上下文*
