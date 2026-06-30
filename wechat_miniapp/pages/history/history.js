// 颜值矩阵分析 — 历史记录页
const app = getApp();

Page({
  data: {
    history: [],
    stats: {
      total: 0,
      avgScore: 0,
      maxScore: 0,
    },
    showClearConfirm: false,
  },

  onShow() {
    this.loadHistory();
  },

  loadHistory() {
    try {
      const history = wx.getStorageSync('beauty_history') || [];
      const stats = this.computeStats(history);
      this.setData({ history, stats });
    } catch (_) {
      this.setData({ history: [], stats: { total: 0, avgScore: 0, maxScore: 0 } });
    }
  },

  computeStats(history) {
    const valid = history.filter(h => h.hasFace && h.score != null);
    const scores = valid.map(h => h.score);
    return {
      total: history.length,
      avgScore: scores.length ? Math.round(scores.reduce((a, b) => a + b, 0) / scores.length * 100) / 100 : 0,
      maxScore: scores.length ? Math.max(...scores) : 0,
    };
  },

  /** 点击查看结果 */
  onTapItem(e) {
    const item = e.currentTarget.dataset.item;
    if (!item.features) {
      wx.showToast({ title: '数据不完整', icon: 'none' });
      return;
    }

    // 从缓存中取完整数据
    const history = wx.getStorageSync('beauty_history') || [];
    const full = history.find(h => h.id === item.id);
    if (!full) {
      wx.showToast({ title: '记录已失效', icon: 'none' });
      return;
    }

    // 构建结果数据
    const result = {
      type: 'face',
      has_face: full.hasFace,
      pref_name: app.globalData.currentPref || '均衡审美',
      beauty_score: full.score,
      grade: full.grade,
      features: full.features,
      geo_dimensions: full.geo_dimensions,
      decomposition: null,
      all_preference_scores: null,
      elapsed_ms: null,
    };

    const encoded = encodeURIComponent(JSON.stringify(result));
    wx.navigateTo({
      url: `/pages/result/result?data=${encoded}`,
    });
  },

  /** 删除单条 */
  onDeleteItem(e) {
    const id = e.currentTarget.dataset.id;
    wx.showModal({
      title: '确认删除',
      content: '删除后不可恢复',
      confirmColor: '#e94560',
      success: (res) => {
        if (res.confirm) {
          try {
            let history = wx.getStorageSync('beauty_history') || [];
            history = history.filter(h => h.id !== id);
            wx.setStorageSync('beauty_history', history);
            this.loadHistory();
            wx.showToast({ title: '已删除', icon: 'success' });
          } catch (_) {
            wx.showToast({ title: '删除失败', icon: 'error' });
          }
        }
      }
    });
  },

  /** 清空确认 */
  onClearAll() {
    this.setData({ showClearConfirm: true });
  },

  /** 确认清空 */
  onConfirmClear() {
    wx.setStorageSync('beauty_history', []);
    this.setData({ showClearConfirm: false, history: [], stats: { total: 0, avgScore: 0, maxScore: 0 } });
    wx.showToast({ title: '已清空', icon: 'success' });
  },

  /** 取消清空 */
  onCancelClear() {
    this.setData({ showClearConfirm: false });
  },

  /** 格式化时间 */
  formatTime(ts) {
    const d = new Date(ts);
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}/${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
  },
});
