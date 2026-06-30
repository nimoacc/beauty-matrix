// 颜值矩阵分析 — 结果页
const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    // 分析结果
    result: null,
    score: 0,
    grade: '',
    gradeLabel: '',

    // 偏好对比
    allPrefScores: [],
    showPrefCompare: false,

    // 化妆模拟
    showMakeup: false,
    makeupDelta: 1.0,
    makeupResult: null,

    // 公式展开
    showFormula: false,

    // Canvas 分享
    canvasReady: false,
  },

  onLoad(options) {
    if (options.data) {
      try {
        const result = JSON.parse(decodeURIComponent(options.data));
        this.setResult(result);
      } catch (e) {
        wx.showToast({ title: '数据解析失败', icon: 'error' });
        setTimeout(() => wx.navigateBack(), 1500);
      }
    }
  },

  setResult(result) {
    const gradeLabels = {
      'S': '倾国倾城', 'A': '美若天仙', 'B': '秀丽端庄',
      'C': '清秀可人', 'D': '自然本色'
    };

    // v52: 性别数据
    const genderData = result.gender || {};
    const genderIcon = { 'male': '👨', 'female': '👩', 'androgynous': '⚧', 'unknown': '👤' }[genderData.gender || 'unknown'];

    let allPrefScores = [];
    if (result.all_preference_scores) {
      allPrefScores = Object.entries(result.all_preference_scores)
        .map(([name, score]) => ({ name, score: Math.round(score * 100) / 100 }))
        .sort((a, b) => b.score - a.score);
    }

    this.setData({
      result,
      score: result.beauty_score || 0,
      grade: result.grade || '-',
      gradeLabel: gradeLabels[result.grade] || '',
      genderData,
      genderIcon,
      allPrefScores,
    });

    // 绘制分享 Canvas
    setTimeout(() => this.drawShareCanvas(), 500);
  },

  /** 化妆模拟 */
  onMakeupSliderChange(e) {
    const delta = parseFloat(e.detail.value);
    this.setData({ makeupDelta: delta });
    this.simulateMakeup(delta);
  },

  async simulateMakeup(delta) {
    try {
      const res = await api.simulateMakeup(this.data.score, delta);
      this.setData({ makeupResult: res });
    } catch {
      // 本地估算
      const baseScore = this.data.score;
      const bonus = Math.log2(1 + delta) * 2.0;
      const newScore = Math.min(10, Math.round((baseScore + bonus) * 100) / 100);
      const diff = Math.round((newScore - baseScore) * 100) / 100;
      this.setData({
        makeupResult: {
          base_score: baseScore,
          delta: delta,
          adjusted_score: newScore,
          delta_score: diff,
          grade: this.computeGrade(newScore),
        }
      });
    }
  },

  computeGrade(score) {
    if (score >= 9.0) return 'S';
    if (score >= 7.5) return 'A';
    if (score >= 6.0) return 'B';
    if (score >= 4.5) return 'C';
    return 'D';
  },

  /** 偏好对比 */
  onTogglePrefCompare() {
    this.setData({ showPrefCompare: !this.data.showPrefCompare });
  },

  /** 公式展开 */
  onToggleFormula() {
    this.setData({ showFormula: !this.data.showFormula });
  },

  /** 化妆面板 */
  onToggleMakeup() {
    const show = !this.data.showMakeup;
    this.setData({ showMakeup: show });
    if (show && !this.data.makeupResult) {
      this.simulateMakeup(this.data.makeupDelta);
    }
  },

  /** 绘制分享 Canvas */
  drawShareCanvas() {
    const query = wx.createSelectorQuery();
    query.select('#shareCanvas')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (!res[0]) return;
        const canvas = res[0].node;
        const ctx = canvas.getContext('2d');
        const w = res[0].width;
        const h = res[0].height;
        const dpr = wx.getSystemInfoSync().pixelRatio;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        // 背景
        ctx.fillStyle = '#0a0a1a';
        ctx.fillRect(0, 0, w, h);

        // 渐变标题
        const grad = ctx.createLinearGradient(w / 2, 0, w / 2, h);
        grad.addColorStop(0, '#e94560');
        grad.addColorStop(1, '#ffd700');
        ctx.fillStyle = grad;
        ctx.font = 'bold 24px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('颜值矩阵分析报告', w / 2, 60);

        // 分数
        ctx.font = 'bold 72px sans-serif';
        ctx.fillStyle = '#fff';
        ctx.fillText(this.data.score.toFixed(1), w / 2, 160);

        // 等级
        ctx.font = 'bold 28px sans-serif';
        ctx.fillStyle = '#e94560';
        ctx.fillText(`${this.data.grade}级 · ${this.data.gradeLabel}`, w / 2, 210);

        // 偏好
        ctx.font = '18px sans-serif';
        ctx.fillStyle = '#8888aa';
        ctx.fillText(`审美视角: ${this.data.result.pref_name || '均衡审美'}`, w / 2, 250);

        // 底部品牌
        ctx.font = '14px sans-serif';
        ctx.fillStyle = '#555577';
        ctx.fillText('颜值矩阵分析系统 · FaceMatriX', w / 2, h - 30);

        this.setData({ canvasReady: true });
      });
  },

  /** 保存分享图 */
  onSaveShareImage() {
    if (!this.data.canvasReady) return;
    const query = wx.createSelectorQuery();
    query.select('#shareCanvas')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (!res[0]) return;
        wx.canvasToTempFilePath({
          canvas: res[0].node,
          success: (result) => {
            wx.saveImageToPhotosAlbum({
              filePath: result.tempFilePath,
              success: () => wx.showToast({ title: '已保存到相册', icon: 'success' }),
              fail: () => wx.showToast({ title: '保存失败', icon: 'error' }),
            });
          },
          fail: () => wx.showToast({ title: '生成失败', icon: 'error' }),
        });
      });
  },

  /** 查看历史 */
  goHistory() {
    wx.switchTab({ url: '/pages/history/history' });
  },

  /** 分享 */
  onShareAppMessage() {
    return {
      title: `颜值得分 ${this.data.score} · ${this.data.grade}级`,
      path: '/pages/index/index',
      imageUrl: '',
    };
  },
});
