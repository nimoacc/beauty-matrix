# Changelog — 颜值矩阵分析系统

本文档记录颜值矩阵分析系统的所有重要变更。

---

## [v53.6] — 2026-06-30 基础分上界下调 + 肤色光照鲁棒

### 📐 评分参数
- **VMAX 7.0→6.5**: 基础分上界 9.0→8.5，释放 0.5 分额度给加分项
  - 避免基础分 + 加分项频繁撞 10.0 天花板
  - 加分项（肤质/肤色/几何）在高端区恢复区分力

### 🔬 肤色判定
- **ITA° 改用 Top-30% 最亮肤色像素**: 原来全肤色像素均值 → 取亮度 Top-30%
  - 侧光/阴影不再压低 ITA°，光照鲁棒性显著提升
  - 与 `chroma_ratio` 统一使用同一组 Top-30% 像素

### 📦 打包更新
- `beauty_v53.spec`: 新增 `skin_tone_cnn_v3.onnx` + `onnxruntime` 依赖
- exe 名称: v53.1 → v53.6

---

## [v53.4] — 2026-06-30 几何双斜率衰减增强

### 📐 几何加分参数
- **penalty_slope 1.0→3.0**: 偏差衰减斜率增大
  - SCUT 验证: 增大 face_ratio 偏离最优区后的区分度
  - 零交点从 ~0.62/0.88 收紧至 ~0.62/0.85

---

## [v53.3] — 2026-06-30 geo_bonus 双向偏差罚分

### 🔴 P0 评分逻辑修正
- **单向偏差 → 双向罚分**: v53.2 使用 `(1-w/h)` 单向打分 (窄脸加分、宽脸扣分), 存在偏见
  - **修复**: 改为 `MAX_BOOST - |face_ratio - 0.75|` 双向偏差罚分
  - 过窄 (w/h<0.65) 和过宽 (w/h>0.85) **同等扣分**
  - 最优比例 0.75 获得最大加分 +0.27
- **去除居中偏移**: 人脸框在画面中的位置不再参与评分 (构图 ≠ 颜值)

### 📊 效果对比 (GEO_WEIGHT=0.5, k=4.0)
| face_ratio | v53.2 geo_b | v53.3 geo_b | 说明 |
|-----------|------------|------------|------|
| 0.75 (最优) | +0.19 | **+0.27** | 最优拿满分 |
| 0.65/0.85 | +0.19/+0.11 | **+0.19** | 对称扣分 |
| 0.55/0.95 | -0.05/+0.10 | **+0.10** | 对称扣分 |
| 0.45/1.05 | -0.15/+0.15 | **-0.05** | 开始扣分 |

---

## [v53.2] — 2026-06-30 geo_bonus评分膨胀修复

### 🔴 P0 评分校准
- **geo_bonus 膨胀修复**: v53.2 前, 所有人脸 geo_bonus≈+1.9~2.0 (常量), 导致普通人脸跑出 9.5+ 高分
  - **根因**: `compute_geo_dimensions.bonus` 全域仅 [0.47,0.63], Sigmoid(x0=0.25) 对其无区分力
  - **修复**: bonus 改为 `(1-w/h)` 主线 + 居中偏移, 范围 [-0.3, +0.3], 可正可负
  - **激活函数**: 普通Sigmoid → 对称Sigmoid `2*sigmoid(x)-1`, 映射到 [-1, +1]
  - **权重**: `geo_weight 1.5→0.5`, `k 6→4`

### 📊 效果对比
| 场景 | v53.1 geo_b | v53.2 geo_b | 总分变化 |
|------|------------|------------|----------|
| 典型居中脸 | +2.00 | **+0.19** | 9.59→7.77 |
| 宽脸居中 | +1.85 | **+0.11** | — |
| 宽脸偏移 | +1.78 | **−0.15** | 首次实现扣分 |

---

## [v53.1] — 2026-06-30 PM审查问题修复 (热修复)

### 🔴 P0 阻断性Bug修复
- **P0-1**: Web API `style` NameError — `analyze_image()` 中 `features_to_style(feats)` 未被调用, 现已补全
- **P0-2**: 桌面GUI版本号 v50→v53.1 — 窗口标题/Logo/导出报告统一为 v53.1
- **P0-3**: Web API health 端点版本号 v52→v53.1

### 🟠 P1 评分一致性修复
- **P1-1**: GUI 评分遗漏 geo_bonus — 分析流程中补全 `compute_geo_dimensions()` + `geo_clarity_bonus()`, 多人脸每张独立计算
- **P1-2**: Web API 缺失瑕疵检测链 — 补全 `detect_skin_blemishes()` + `blemish_penalty()`
- **P1-3**: Web API 评分逐步clamp vs 最终clamp不一致 — 统一为一次性 `min(sum, 10.0)` 最终clamp

### 🟡 P2 代码质量修复
- **P2-1**: `beauty_core.py:319` `dir()` 冗余检查 — 移除 `if 'np' in dir()` 防御性代码
- **P2-2**: `compute_geo_dimensions()` 重复 `import numpy` — 移除函数内部重复导入
- **P2-3**: GUI多人脸 geo_dims — 每张脸独立计算并存入 face_results
- **P2-4**: 微信小程序无独立版本号 — `app.js` 新增 `version: 'v53.1'`, `project.config.json` 同步
- **P2-5**: 三端 infer_gender_v5 调用方式不一致 — v53 已统一通过 `extract_face_roi_features` 内部调用, Web API 仅作 fallback

### 修改文件
- `beauty_core.py` — P2-1/P2-2 代码清理 + 版本号 v53→v53.1
- `beauty_gui_desktop.py` — P0-2 版本号统一 + P1-1 几何加分补全 + 评分分解显示更新
- `web/app.py` — P0-1 style修复 + P0-3 版本号 + P1-2 瑕疵链 + P1-3 clamp统一
- `wechat_miniapp/app.js` — P2-4 版本号
- `wechat_miniapp/project.config.json` — P2-4 描述更新
- `CHANGELOG.md` — 本文档

---

## [v53] — 2026-06-29 性别检测校准 (修复女性误检为男性)

### 🐛 关键Bug修复: 女性被系统性误判为"男性"

**根因**: v52 的几何启发式公式三个分量全部系统性偏向男性:

| 分量 | v52 中性点 | 实际典型范围 | 偏差方向 |
|---|---|---|---|
| `face_aspect` (脸宽/脸高) | 0.78 | 0.65~0.85 | 宽脸女性 < 0.78 → **偏男** |
| `topbot_ratio` (上半区/下半区边缘密度) | 1.05 | 1.2~1.8 | 眉毛+眼睛边缘多 → **几乎恒偏男** |
| `face_sharpness` (log1p(Laplacian方差)) | 3.5 | 3.9~8.5 | 纹理清晰 → **几乎恒偏男** |

三个正向偏置叠加 → 女性面孔轻松超过 0.2 男性判定阈值 → 误检。

### 🔧 修复内容

1. **零中心化校准** (`beauty_core.py` 性别启发式):
   - `face_aspect` 中性点: 0.78→0.76, 权重: 4.0→2.5
   - `topbot_ratio` 中性点: 1.05→1.25, 权重: 2.0→1.0
   - `face_sharpness` 中性点: 3.5→4.5, 权重: 0.5→0.3
   - male/female 判定阈值: ±0.2→±0.3 (更宽的中性区)

2. **确定性几何维度** (`compute_geo_dimensions()`):
   - 替代 `random.uniform()` 随机模拟
   - 从 face_rect 和图像特征中确定性估算 jaw_angle/symmetry_index/face_ratio 等

3. **纹理标签修复** (`texture_precheck()`):
   - 新增返回 `texture_label` 字段 (fine/medium/coarse)
   - 基于拉普拉斯方差阈值 (>200粗 / >80中 / ≤80细)

4. **Web API 修复** (`web/app.py`):
   - `geo_dims` 仅在有数据时传入 `infer_gender_v5`
   - 版本号更新为 v53

---

## [v53] — 2026-06-29 ML性别分类器 (替代手工启发式公式)

### 🤖 新增: 基于 SCUT-FBP5500 训练的 ML 性别分类器

**问题**: v52 手工公式 (v53 校准后) 仍不可靠, 无法达到生产级精度.

**v53 方案**: 在 SCUT-FBP5500_v2 5500张标注人脸上训练 LogisticRegressionCV 分类器.

### 训练结果

| 指标 | 值 |
|------|-----|
| 数据集 | SCUT-FBP5500_v2 (5500张: AF/AM/CF/CM) |
| 模型 | LogisticRegressionCV (L2, 5-fold CV, 629维多项式特征) |
| 测试集准确率 | **80.9%** |
| ROC-AUC | **0.879** |
| 亚洲人脸准确率 | **82.7%** |
| 女性误检为男性 | **18.0%** (v52 手工公式 ~100%) |
| 模型文件大小 | 255 KB |

### 📁 新增文件
- `train_gender_model.py` — 性别分类器训练脚本
- `stats_output/gender_model_v1.pkl` — 训练好的模型
- `stats_output/gender_features_cache.pkl` — 特征缓存
- `beauty_v53.spec` — PyInstaller spec (含性别模型)

### 🔧 修改文件
- `beauty_core.py`:
  - 新增 `predict_gender_v53()` — ML 模型预测函数
  - 新增 `_load_gender_model()` — 模型懒加载
  - 新增 `_fallback_gender_heuristic()` — 模型不可用时降级
  - `_extract_32d_features()` 新增 `jaw_edge_density`/`cheek_sharpness` 两个性别特征
  - `FaceFeatures` 新增 `gender_confidence` 字段
  - `extract_face_roi_features()` 改为调用 ML 模型
- `image_utils.py`: `texture_precheck()` 新增 `texture_label` 返回
- `web/app.py`: 使用 ML 模型输出 + 置信度
- `beauty_gui_desktop.py`: 显示性别置信度

---

## [v52] — 2026-06-28 性别感知风格推断 + 多平台同步 + exe 打包

### ⭐ 新功能: 男性/女性分套美学标签

**问题**: 所有风格名称均为女性化描述 (甜美少女/高级超模/古典东方...)，男性照片也被分配给女性风格。

**v52 修复**:
- `FaceFeatures` 新增 `gender` 字段 (male/female/unknown)
- `extract_face_roi_features()` 内置几何启发式性别检测:
  - 使用面宽比 (face_aspect)、眉骨突出度 (topbot_edge_ratio)、皮肤锐度 (face_sharpness) 三指标
  - 宽脸 + 高眉骨 + 粗纹理 → 男性；椭圆脸 + 平滑 → 女性
- `features_to_style()` 双性别美学标签:
  - 男性: 阳光少年 / 硬朗型男 / 儒雅绅士 / 韩系俊朗 / 日系清新 / 硬朗雕塑
  - 女性: 甜美少女 / 高级超模 / 古典东方 / 韩系精致 / 日系自然 / 建模标杆
  - 中性: 瓷肌白肤 / 均衡风格
- GUI 显示 `👤 性别: 男性/女性/未识别` + 导出含性别字段

### 🛠 配套修复
- 修复 bat 文件 UTF-8 编码导致的启动闪退 (中文括号冲突)
- `FaceFeatures.as_dict()` 补充 gender 字段

### 📦 exe 打包
- PyInstaller 6.19.0 打包 → `颜值矩阵分析系统 v52.exe` (130MB)
- 覆盖旧版 v38, 存放在根目录和 `dist/`

### 🌐 多平台同步
- **Web API** (`web/app.py`): 
  - 性别检测优先用 v52 几何启发式 `feats.gender`，降级用 `infer_gender_v5`
  - 版本号更新为 v52
- **微信小程序** (`wechat_miniapp/pages/result/`): 
  - 结果页新增性别显示 (👨/👩/⚧ + 性别标签)
  - 新增风格显示 (primary_style)
  - 保留原审美视角显示

---

## [v51] — 2026-06-28 肤色判定重写 (标准CIE ITA° + 肤色像素过滤)

### 🐛 关键Bug修复: 白皮肤被误判为"深肤色"

**根因**: `classify_skin_tone()` 中 ITA° 公式存在三个问题:

| 问题 | 旧版(v49) | v51 修复 |
|---|---|---|
| **色彩空间** | 直接使用 OpenCV Lab 原生值 (L:0-255, b:0-255) | 转换为标准 CIE Lab (L*:0-100, b*:-128~127) |
| **像素过滤** | 整个人脸 ROI 取均值 (混合头发/眉毛/阴影) | HSV+YCrCb 双空间肤色掩码过滤, 仅对肤色像素计算 |
| **阈值** | 基于错误公式的 SCUT 经验阈值 | 临床皮肤科 ITA 标准 + 东亚人群微调 |

**效果**: 白皮用户不会再被显示为 🎭 肤色: 深肤色.

### 技术细节

- 新增 `_compute_skin_pixel_mask()` 辅助函数
- `classify_skin_tone()` v51 重写: 肤色掩码 + 标准CIE公式 + 新阈值(>48浅 / >20中 / ≤20深)
- 同步修正 `calibrate_skin_tone.py` 的 `compute_ita()` 公式
- 旧版 SCUT 校准数据(均值40.3等)基于错误公式, 需重新运行校准脚本

---

## [v50] — 2026-06-27 瑕疵检测完全重写 (SCUT数据驱动校准)

### 🔬 核心变更: 统计自适应瑕疵检测器

v49 的硬阈值检测器在 SCUT 数据集上表现失控 (Mean=5.97, 100% >4.0), 几乎把所有正常脸判为瑕疵。

**v50 完全重写, 三大改进:**

1. **面部比例掩码**: 自动排除眼(12-45%)/鼻(40-65%)/唇(65-92%)区域, 柔和边缘过渡, 杜绝正常面部结构误检
2. **肤色统计自适应**: 每张脸独立计算皮肤 HSV S / Lab L 中位数+MAD, 痘印阈值=S_median+2.5MAD, 色斑=L_median-2.5MAD, 不受光照/肤色/图片质量影响
3. **圆形度过滤**: 连通组件筛选 circularity>0.25(痘印)/0.2(色斑), 排除不规则纹理/毛发/阴影

### 📊 SCUT 2300张校准结果

| 指标 | v49 (旧) | **v50 (新)** |
|---|---|---|
| **Mean** | 5.97 | **1.76** |
| **Median** | 5.65 | **1.70** |
| **P90** | 7.90 | **3.90** |
| **P95** | 8.11 | **4.50** |
| **>4.0%** | 100% | **8.5%** |
| **>6.0%** | ~80% | **0.4%** |

### 📐 扣分阶梯重新标定

| 瑕疵分 | 旧减分 | **新减分** | 分布占比 |
|---|---|---|---|
| ≤2.5 | 0 | **0** | ~61% |
| 2.5~4.0 | 0→0.4 | **0→0.3** | ~30% |
| 4.0~6.0 | 0.4→1.5 | **0.3→1.0** | ~8% |
| 6.0+ | 1.5→3.0 | **1.0→2.0** | ~0.4% |

### 🔧 缺陷判定阈值更新
- `diagnose_defects()` 触发门槛: 2.0 → **2.5**
- 轻微: 2.5~4.0, 明显: 4.0~6.0, 严重: 6.0+

### 📦 新增数据集
- **ISIC Archive**: 50张皮肤镜病变图像 (nevus/melanoma), 用于验证检测器对真实色斑的敏感度
- 安装 `gdown` 用于未来 FFHQ 下载

### 修改文件
- `image_utils.py` — `detect_skin_blemishes()` 完全重写 + 新增 `_create_skin_mask()`
- `beauty_core.py` — `blemish_penalty()` 阈值重新标定 + `diagnose_defects()` 瑕疵门槛调整
- `calibrate_blemish.py` — 新增校准脚本 (SCUT 2300张 + ISIC 50张)
- `download_calibration_data.py` — 新增数据集下载脚本
- `stats_output/blemish_calibration_v50.json` — 校准参数存档
- `CHANGELOG.md` — 本文档

---

## [v49] — 2026-06-27 肤色校准 + 瑕疵检测 + 启动脚本修复

### 🔬 肤色分类 SCUT 校准
- **阈值数据驱动**: 基于 5500 张 SCUT-FBP5500_v2 人脸 ITA° 分布重新标定
  - Asian ITA 均值=40.3, Caucasian ITA 均值=35.9, 整体集中在 [20, 50]
  - 原临床阈值 (>55/28/-30) 对东亚人群过宽→新阈值 >43/40/28/10
- **Cr/Y 色度比辅助**: 高 ITA+低色度=白皙, 高 ITA+高色度=红润, 细分浅肤色子类型
- **新增校准脚本**: `calibrate_skin_tone.py` 输出 `stats_output/skin_tone_calibration.json`

### 🔍 面部瑕疵检测 (痘/色斑)
- **`detect_skin_blemishes()`** (新增 ~120行): HSV红色痘印 + Lab暗区 + DoG斑点三层检测
  - 形态学处理: 开运算去噪→闭运算连接→连通组件面积/圆形度筛选
  - 返回 0-10 瑕疵评分 + 痘/斑计数 + 诊断详情
- **`blemish_penalty()`** (新增): 瑕疵分映射减分量
  - ≤2 分: 不减, 2→5: 0→0.4, 5→8: 0.4→1.5(加速), 8→10: 1.5→3.0
- **评分公式更新**: `total = min(beauty + skin_b + tone_b + geo_b - blemish_p, 10.0)`
- **缺陷诊断增强**: `diagnose_defects()` 瑕疵≥4 强制加入缺陷列表, 不受总分门限限制
- **GUI 显示**: 建议 Tab 展示瑕疵评分/痘数/斑数/减分值

### 📐 参数调整
- **图片最小尺寸**: 500×500 → **200×200**, 降低准入门槛

### 🔧 bat 启动脚本修复
- **`__pycache__` 自动清理**: 启动前删除所有 `__pycache__` 目录, 确保源码改动立即生效
- **全量依赖检查**: 从仅检查 `customtkinter` 扩展为检查 `numpy`/`cv2`/`PIL`/`sklearn`
- **版本号同步**: 硬编码 v48→v49, 与源码一致

### 修改文件
- `image_utils.py` — 肤色阈值校准 + 瑕疵检测 + 尺寸限制
- `beauty_core.py` — `FaceFeatures.blemish_score` + `blemish_penalty()` + 评分管线 + 缺陷诊断
- `beauty_gui_desktop.py` — 瑕疵检测集成 + GUI 显示 + 版本号
- `beauty_system/config.yaml` — v49 参数
- `启动颜值分析系统.bat` — 清理缓存 + 依赖检查 + 版本同步
- `calibrate_skin_tone.py` — 新增校准脚本
- `CHANGELOG.md` — 本文档

---

## [v48] — 2026-06-25 SCUT-FBP5500 训练校准

### 🧠 模型训练 (核心变更)
- **基于 SCUT-FBP5500_v2 全量训练**: 5500张人脸, 60人评分均值
- **32维真实视觉特征提取**: 替代随机模拟, 包含全局/人脸亮度对比度锐度熵/颜色矩/对称性/边缘密度等
- **RidgeCV 回归**: 32维→560维多项式→Ridge(alpha=162), 官方60/40 split
- **SCUT 1-5 → 2-9 tanh 非线性映射**: 两端稀有, 中间集中, std=1.12
- **Pearson r = 0.6135** (SCUT scale), **r = 0.6310** (2-9 scale) — 远超 0.3 目标
- **Spearman ρ = 0.5971** (p=1.18e-212)
- **MAE = 0.42** (on 1-5 scale), **R² = 0.374**

### 📐 评分管线 v48 参数校准
| 参数 | v44 | v48 | 说明 |
|:---|:---|:---|:---|
| vmax | 5.5 | 7.0 | 扩大动态范围 |
| k_half | 8.0 | 16.3 | 数据驱动 |
| q_weight | 22.0 | 5.0 | det贡献占比提升 |
| calib_offset | -0.2 | 2.0 | 底线=2分 |
| det_cap | 15.0 | 13.3 | 自适应 |
| geo_weight | 3.2 | 1.5 | 减少随机干扰 |

### 🔧 代码变更
- `beauty_core.py`: 嵌入 Ridge 模型 + `predict_calibrated_score()` 函数
- `beauty_core.py`: `extract_face_roi_features` 改为真实特征提取
- `beauty_system/config.yaml`: v48 参数写入
- 新增 `train_beauty_model.py`: 完整训练脚本
- 新增 `stats_output/beauty_model_full.pkl`: 可部署模型

### 📦 编译
- `颜值矩阵分析系统 v48.exe` (~130MB, 含 sklearn + 模型)

---

## [v47] — 2026-06-24 多项修复 & 显示优化

### 🐛 修复
- **人脸框精准对齐**: `_apply_results` 用 `Image.fromarray(self.current_image)` 替代 `self.current_display`，确保坐标与 Canvas 尺寸一致
- **维度 tab 增强**: 添加 `update_idletasks` 强制布局、`try/except` 异常捕获、`_draw_feature_bars_fallback` 降级渲染

### ✨ 显示优化
- **矩阵 tab 彩色化**: 正值🟢/负值🔴 着色，强值(>2)粗体高亮，增强可读性
- **对比条形图**: 从固定10字宽改为35字宽相对比例缩放，分数相近时差异仍明显可见，末尾追加差距数值

### 📦 编译
- `颜值矩阵分析系统 v47.exe` (~71MB)

---

## [v46] — 2026-06-24 Bug 修复

### 🐛 修复
- **CTkTextbox `tag_config` font 崩溃**: 改用底层 `_textbox.tag_config()` 绕过 CustomTkinter 对 font 参数的限制
- **人脸框位置偏移**: `resize_for_analysis` 缩放后检测框坐标未还原到原图尺寸，导致 Canvas 上人脸框错位 — 新增 `scale_x/scale_y` 比例还原

### 📦 编译
- PyInstaller 6.19.0 → `颜值矩阵分析系统 v46.exe` (~71MB)

---

## [v45] — 2026-06-24 UI 美学重构

### 🎨 视觉升级
- **全新配色体系**: 深紫靛蓝主题 (BG_ROOT #08081a → BG_CARD #1a1a3e → ACCENT #7c3aed)
- **头部标题栏重设计**: Logo ✦ v45 标识 + AI 引擎描述 + 状态指示圆点
- **Tab 标签图标化**: 🔍 分析 / 📋 批量 / 🎨 偏好 / 📝 问卷
- **图片 Canvas 欢迎引导**: 虚线边框 + 摄像头 Emoji + 格式提示
- **Canvas 特征柱状图**: 维度 Tab 从纯文本条升级为 Canvas 渐变柱 + 数值标签
- **ASCII 环形评分仪表**: 评分 Tab 用 █░ 色条 + 大号彩色分数 + 等级标签
- **卡片边框体系**: 所有面板统一圆角12 + BORDER_COLOR 1px 边框
- **按钮药丸风格**: 统一 corner_radius=8 + 悬停变色
- **CheckBox 紫色主题**: fg_color=ACCENT 统一紫色选中态
- **进度条紫色**: batch_progress 紫色进度
- **滑块紫色**: weight_sliders 紫色进度条 + 按钮
- **状态指示系统**: 标题栏圆点实时变色 + `_set_status()` 统一管理

### 🔧 代码整理
- `_set_status()` 统一状态栏 + 状态指示灯更新
- 移除废弃的 `feats_text` 类型声明
- 修复 `_fs(11,)` 多余逗号语法
- 所有状态消息添加 Emoji 图标

### 📦 编译
- PyInstaller 6.19.0 → `颜值矩阵分析系统 v45.exe` (~71MB)

---

## [v44] — 2026-06-24 天花板修复 & 分布范围释放

### v44.1 重磅修复 (2026-06-24)
- **侧脸增强GUI开关**: 恢复分析标签页的「侧脸增强」复选框，支持旋转±15°/±30°检测
- **大脸增强GUI开关**: 恢复分析标签页的「大脸增强」复选框，支持宽松参数组
- **纹理预检状态卡片**: 在结果面板顶部新增独立状态卡片，实时显示清晰度/噪点/光照/分辨率
- **det(A) 非零修复**: 矩阵对角改回乘法，添加缩放身份矩阵打破秩-1结构，确保 det(A) 显著非零
- **5×5矩阵标签修复**: 行→审美维度(C1-C5)，列→器官(眼/鼻/唇/轮廓/肤质)
- **矩阵展示Tab**: 独立"矩阵"标签页显示5×5美学矩阵+行列式+含义解释
- **版本号统一**: 标题栏/窗口/v44 everywhere
- **偏好传递修复**: `compute_all_preference_scores` 和批量分析传递 pref_raw 到矩阵
- **detect_faces 增强**: 支持 `enhance_side`/`enhance_large` 动态开关

### 问题诊断
v43 的 `CALIB_OFFSET=-1.35` 一刀切给所有人减了 1.35 分，天花板被压死在 7.74，`惊艳(8+)`和`绝世(9.5+)` 完全不可达。

### 参数变更

| 参数 | v43 | **v44** | 作用 |
|------|:--:|:--:|------|
| `VMAX` | 5.0 | **5.5** | 扩展 Hill 量程 |
| `K_HALF` | 6.0 | **8.0** | 抬高半饱和点, 补偿均值 |
| `Q_WEIGHT` | 30.0 | **22.0** | 降低品质权重 |
| `DET_CAP` | 15.0 | 15.0 | 不变 |
| `CALIB_OFFSET` | -1.35 | **-0.2** | 削掉 85% 偏移, 释放天花板 |
| `GEO_WEIGHT` | 3.0 | **3.2** | 略增几何空间 |
| `GEO_K` | 4.0 | **6.0** | 加大 sigmoid 陡峭度 |
| `GEO_X0` | 0.15 | **0.25** | 右移拐点, 拉开中高分差距 |
| `GEO_BMAX` | 1.5 | 1.5 | 不变 |
| `SKIN_BMAX` | 0.4 | **0.5** | 增加肤质加分上限 |
| `SKIN_K` | 3.0 | 3.0 | 不变 |
| `SKIN_X0` | 0.50 | 0.50 | 不变 |

### 效果对比

| 指标 | v43 | **v44** |
|------|:--:|:--:|
| **天花板 (理论最大)** | 7.74 | **9.48** |
| **惊艳(8+) 图片数** | 0 张 | **45 张** |
| 总分标准差 | ~1.3 | 2.13 |
| 分值跨度 | 4.23 | 5.90 |

### 修改文件
- `beauty_system/config.yaml` — 添加 scoring/geo_bonus_cfg/skin_clarity_bonus_cfg 三段配置
- `beauty_core.py` — raw_to_beauty/skin_clarity_bonus/geo_clarity_bonus 全部升级为 v44 公式
- `gender_inference.py` — 升级到 v5 级算法 (6维几何 + sigmoid校准)
- `beauty_gui_desktop.py` — 版本号 v38→v44

---

## [v43] — 2026-06-23 方差优化 & 硬+软兜底检测

### 参数变更

| 参数 | v42 | **v43** | 作用 |
|------|:--:|:--:|------|
| `_GEO_BONUS_K` | 6.0 | **4.0** | ↓sigmoid 陡峭度 → geo 加分差异化 |
| `_GEO_BONUS_B_MAX` | — | **1.5** | 新增模块级变量 |
| `GEO_WEIGHT` | 1.5 | **3.0** | 放大几何加分量程 |
| `CALIB_OFFSET` | 0.0 | **-1.35** | 补偿 bonus 抬升的 μ |
| `_SKIN_BONUS_K` | 8.0 | **3.0** | ↓肤质 sigmoid 陡峭度 |
| `_B_MAX_SKIN_CLARITY` | 0.5 | **0.4** | 微调 |

### SCUT 评估 (2155 对)

| 指标 | v42 | **v43** | 变化 |
|------|:--:|:--:|------|
| **Pearson r** | 0.445 | **0.490** | +10% |
| **MAE** | 0.98 | **1.06** | +8%（方差代价可接受） |
| **model_mean** | 6.0 | **6.0** | 不变 |
| **model_std** | 0.74 | **1.25** | +69% (接近目标 1.35) |

### 硬+软兜底检测
- 硬检测 (Haar 5策略) 优先，漏检时软检测兜底
- 45 张无人脸 → 4 张 (收回 41 张)
- `batch_analyze_images` 默认 `soft_detect=True`

---

## [v42] — 2026-06-22 模块化架构重构

### 架构变更
- `beauty_core.py` 65KB 单体 → `beauty_system/` 6子模块拆分
- `beauty_system/config.yaml` — 统一配置中心
- `beauty_system/beauty_scoring.py` — Hill评分+加分链
- `beauty_system/face_detection.py` — 人脸检测+性别推断+几何美学
- `beauty_system/feature_extraction.py` — 特征提取+肤质计算
- `beauty_system/data_types.py` — 共享数据类定义

---

## [v41] — 2026-06-20 性别判定 v5: 域泛化+概率校准

### 训练改进
- **混合训练集**: SCUT + UTKFace (285+ 人)
- **域泛化**: 跨数据集迁移，避免 SCUT 过拟合
- **概率校准**: IsotonicRegression 曲线校准
- **Stacking 集成**: XGBoost + LogisticRegression → 最终分类器
- **UTKFace 准确率**: **96.3%** (152/4/7/134)

---

## [v40] — 2026-06-19 性别判定 v4: MediaPipe + LBP 纹理

### 模型升级
- **MediaPipe 面部地标**: 468点 → 85维特征向量
- **LBP 纹理特征**: 皮肤纹理粗糙度编码
- **SCUT 准确率**: **89.75%**, AUC=0.9665
- 训练数据: SCUT-FBP5500 前400张

---

## [v39] — 2026-06-18 性别判定 v1: Haar 级联 + 15维 SVM

### 基础架构
- Haar 级联人脸检测 (正面+侧脸旋转)
- 15维几何特征提取 (眉骨/下颌/鼻梁/面宽...)
- SVM 分类器 + 5-fold 交叉验证
- 初始准确率: 56.0% → 经过 v40/v41 迭代至 96.3%

---

## [v38-recovery] — 2026-06-24 (从打包 exe 恢复 & 完整 debug)

### Bug 修复
- **`beauty_core.py` 缺少 `import cv2`**：类 `_apply_face_mask`、`describe_non_face_image`、`batch_analyze_images` 运行时崩溃 → 已添加顶行导入
- **`beauty_gui_desktop.py` 中文引号语法错误**：第512/1362行使用 `"..."` 导致 Python 解析器报 SyntaxError → 改为 `「...」`
- **`generate_beauty_advice` 位置错误**：定义在 `beauty_gui_desktop.py` 中导致 `import customtkinter` 失败时无法导入 → 移至 `beauty_core.py`，GUI 通过 `from beauty_core import *` 获取

### GUI 完整重构 — v38 特性
| 功能 | 描述 |
|------|------|
| **DPI 感知** | `_init_dpi_awareness()` / `_init_screen_scale()` — Windows/macOS 自适应 |
| **多人脸导航** | Canvas 绘制人脸框 + 点击切换 + 上一张/下一张按钮 |
| **批量分析 Tab** | 文件夹选择 → 异步分析 → 进度条更新 → 结果排序/CSV+JSON导出 |
| **偏好建议 Tab** | `generate_beauty_advice()` — 加权差距排序 + 5维建议库匹配 |
| **自定义偏好存储** | `custom_preferences.json` 持久化 → 保存/恢复/删除已存偏好 |
| **化妆模拟弹窗** | δ∈[0,2] 滑块 → `simulate_makeup()` → 实时预览效果 |
| **导出报告** | 人脸 → TXT/JSON；批量 → CSV/JSON/TXT |
| **4标签页** | 分析(评分/维度/对比/建议) / 批量 / 偏好 / 问卷 |
| **多人脸全脸分析** | 每张脸独立评分 + 全偏好对比 |

### 测试覆盖 (15项)
```
模块导入: beauty_core ✅ image_utils ✅ preference_questionnaire ✅ gender_inference ✅
功能测试: Hill函数 ✅ 场景描述 ✅ 环境稀释 ✅ 人脸遮罩 ✅ 
         特征提取 ✅ 评分计算 ✅ 化妆模拟 ✅ 偏好对比 ✅ 
         美化建议 ✅ 问卷计算 ✅ 性别推断 ✅ 纹理预检 ✅ 
         旧接口兼容 ✅ 批量分析 ✅ 肤色分类 ✅
```

### 文件状态
```
beauty_core.py           776行 ← 新增 cv2 导入 + generate_beauty_advice + _ADVICE_DB
image_utils.py           541行 ← v38 多策略Haar + 软检测 + 肤色分类
preference_questionnaire.py 416行 ← GUI模式 + CLI模式
gender_inference.py       69行 ← 未修改
beauty_gui_desktop.py    ~900行 ← 完整v38重构 (DPI/多人脸/批量/建议/导出)
CHANGELOG.md              ← 更新
```

---

## [v38] — 2026-06 (从打包 exe 恢复)

### 人脸检测
- **v38 多策略 Haar 检测**（`detect_face()`）：3组级联参数 + CLAHE增强 + 侧脸旋转重试 + 大脸自适应预处理，minSize绝对值上限防止参数爆炸
- **v38 全图纹理预检**（`precheck_texture_quality()`）：Laplacian方差三级分级（≥80正常 / 30~80偏低 / <30极低）
- **v36 软人脸检测**（`detect_face_soft()`）：6维软特征加权合成概率（0~1），标记所有疑似区域
- **v37 肤色分类**（`classify_skin_tone()`）：Top-30%亮度基准 + Cr/Y归一化色度比 + ITA°仲裁，消除双向误判
- **多层验证**：NMS去重 → 宽高比 → 肤色比例 → 边缘结构 → 纹理方差 → 眼睛/鼻子特征

### 图片处理
- **v38 格式白名单加载**（`load_and_normalize_image()`）：支持 path/bytes，EXIF自动旋转，拒绝GIF/SVG/RAW，500px最小尺寸
- **级联分类器懒加载**（`_resolve_cascade_path` / `_safe_load_cascade`）：PyInstaller `_MEIPASS` 兼容
- **v13 面/背景稀释补偿**（`_compute_env_dilution()`）：颜色距离 → [1.0, 1.30] 补偿系数
- **椭圆遮罩**（`_apply_face_mask()`）：高斯羽化 + 肤色均值填充，去除四角干扰

### 场景分析
- **v38 非人脸场景描述**（`describe_non_face_image()` / `SceneDescription`）：类别推断 / 颜色调色板 / 自然语言描述
- **v38 批量分析**（`batch_analyze_images()`）：文件夹遍历 + 进度回调 + 取消支持 + 偏好感知

### GUI
- **DPI 感知**（`_init_dpi_awareness` / `_init_screen_scale`）：Windows/macOS 自适应缩放
- **v35 偏好驱动建议**（`generate_beauty_advice()`）：加权差距排序 + 标签匹配建议库
- **多人脸导航**：Canvas 点击切换 + 人脸概览 Tab + 全脸分析
- **批量处理 Tab**：文件夹选择 → 缩略图加载 → 异步分析 → 进度更新 → 导出报告
- **问卷弹窗集成**：GUI内嵌12题问答 → 偏好向量 → 应用自定义偏好
- **自定义偏好管理**：保存/加载/恢复/清除 self.custom_pref_store

### 问卷
- **GUI 模式**（`PreferenceQuestionnaireGUI`）：独立/嵌入双模式，`gui_mode()` / `cli_mode()` / `main()` 入口
- **结果页面**：5维进度条 + 审美风格匹配 + 审美哲学解释 + 重新测试/应用偏好

---

## [v2.1] — 2026-06

### 新增
- **微信小程序**：完整的移动端体验
  - 首页：拍照/选图 + 审美偏好选择器（11种预设）+ 纹理预检
  - 结果页：综合评分、化妆模拟（δ=0~2.0）、评分公式透明化、偏好对比表、Canvas分享图
  - 历史页：分析次数/平均分/最高分统计、时间排序列表、单条删除/清空
  - 问卷页：12题互动问答 → 5维审美向量 → 匹配最佳预设
- **API 升级**：`/api/analyze` 接受 `pref_name` 参数，按审美偏好动态计算 skin_clarity、skin_tone、geo 加分
- **分享功能**：Canvas 生成分享卡片（分数+等级+偏好视角）
- **偏好多维度对比**：同一张脸在11种审美下的得分排序

### 优化
- 历史记录存储完整性（增加 `features` 和 `geo_dimensions` 字段）
- 小程序 UI 采用暗色赛博朋克主题，适配移动端交互

---

## [v2.0] — 2026-05

### 新增
- **桌面GUI**（`beauty_gui_desktop.py`）：基于 customtkinter
  - 三栏布局：图片区 / 分析结果区 / 偏好设置区
  - 19种预设审美偏好切换
  - 偏好对比表（Tab3）：同一张脸不同审美得分对比
  - 多人脸支持、批量分析、导出报告
  - 化妆模拟滑块、背景去除/快速模式/检测增强开关
  - 分析耗时/停止分析
- **Web API**（`web/app.py`）：基于 Flask + CORS
  - `/api/health` — 健康检查
  - `/api/preferences` — 获取偏好列表
  - `/api/questionnaire` — 获取问卷题目
  - `/api/questionnaire/compute` — 提交问卷答案
  - `/api/precheck` — 图片纹理预检
  - `/api/analyze` — 核心分析接口
  - `/api/makeup/simulate` — 化妆模拟
  - `/api/compare` — 多偏好对比
- **审美偏好系统**（`beauty_system/config.yaml`）：
  - 11种预设审美视角
  - 每种预设配置 5 维权重 + skin_clarity 权重 + skin_tone 权重
- **偏好问卷**（`preference_questionnaire.py`）：12道选择题计算用户个人审美向量

### 核心算法升级
- **v30 肤质白净透亮加分**：`skin_clarity_bonus()` 按偏好权重动态加分
- **v31 MediaPipe 6 维几何美学**：`compute_geo_dimensions()` → `geo_clarity_bonus()` 几何加分
- **v40 肤色偏好亲和加分**：`skin_tone_affinity_bonus()` 按肤色类型/偏好计算亲和度
- **评分公式透明化**：`S = 10 × Hill(det(A), k=3, n=1.5) × (0.5 + 0.5 × Hill(Q, k=5, n=2))` + 3项加分分解

---

## [v1.2] — 2026-04

### 新增
- **性别推断**（`gender_inference.py`）：基于面部特征参数推断性别置信度

### 优化
- 图片纹理预检增强：清晰度/噪点/压缩伪影三维度评估
- 人脸检测支持侧脸增强/大脸增强选项

---

## [v1.1] — 2026-03

### 新增
- **图片工具模块**（`image_utils.py`）：load/resize/remove_bg/预检/肤色估算

### 优化
- 背景去除算法：肤色掩码 → 形态学 → 最大连通区域 → 高斯羽化边缘

---

## [v1.0] — 2026-02

### 新增
- **核心引擎**（`beauty_core.py`）：Hill 函数 + 5×5 矩阵 + FaceFeatures 10维 + GeoDimensions 6维
- 评分系统、等级划分、风格推断、化妆模拟、全偏好对比
- **初始数据集集成**：SCUT-FBP5500_v2
