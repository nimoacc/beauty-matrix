// 颜值矩阵分析 — 首页
const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    // 图片
    imagePath: '',
    imageWidth: 0,
    imageHeight: 0,

    // 偏好选择器
    prefList: app.globalData.PREF_LIST,
    currentPref: app.globalData.currentPref,
    prefIndex: 0,
    showPrefPicker: false,

    // 状态
    status: 'idle', // idle | prechecking | precheck_done | analyzing | done | error
    precheckResult: null,
    errorMsg: '',

    // 选项
    removeBg: true,
    quickMode: false,
  },

  onLoad() {
    const prefList = app.globalData.PREF_LIST;
    const currentPref = app.globalData.currentPref;
    const idx = prefList.indexOf(currentPref);
    this.setData({
      prefList,
      currentPref,
      prefIndex: idx >= 0 ? idx : 0,
    });
  },

  onShow() {
    // 同步偏好（从问卷页返回可能修改）
    const currentPref = app.globalData.currentPref;
    const idx = this.data.prefList.indexOf(currentPref);
    if (currentPref !== this.data.currentPref) {
      this.setData({ currentPref, prefIndex: idx >= 0 ? idx : 0 });
    }
  },

  /** 选择图片 */
  chooseImage() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album', 'camera'],
      sizeType: ['compressed'],
      success: (res) => {
        const file = res.tempFiles[0];
        this.setData({
          imagePath: file.tempFilePath,
          imageWidth: file.width || 300,
          imageHeight: file.height || 400,
          status: 'idle',
          errorMsg: '',
          precheckResult: null,
        });
        // 自动预检
        this.doPrecheck();
      },
      fail: (err) => {
        if (err.errMsg.indexOf('cancel') === -1) {
          this.setData({ errorMsg: '选择图片失败' });
        }
      }
    });
  },

  /** 纹理预检 */
  async doPrecheck() {
    if (!this.data.imagePath) return;
    this.setData({ status: 'prechecking' });
    try {
      const res = await api.precheckImage(this.data.imagePath);
      this.setData({
        precheckResult: res.precheck,
        status: 'precheck_done',
      });
      // 纹理通过则自动分析
      if (res.precheck && res.precheck.level === 'good') {
        this.doAnalyze();
      }
    } catch (e) {
      this.setData({
        precheckResult: { level: 'warn', detail: e.message || '预检失败' },
        status: 'precheck_done',
      });
    }
  },

  /** 开始分析 */
  async doAnalyze() {
    if (!this.data.imagePath) return;

    this.setData({ status: 'analyzing', errorMsg: '' });

    wx.showLoading({ title: 'AI分析中...', mask: true });

    try {
      const result = await api.analyzeImage(this.data.imagePath, {
        pref_name: this.data.currentPref,
        remove_bg: this.data.removeBg,
        quick_mode: this.data.quickMode,
      });

      wx.hideLoading();

      // 存入历史
      this.saveToHistory(result);

      // 跳转结果页
      const encoded = encodeURIComponent(JSON.stringify(result));
      wx.navigateTo({
        url: `/pages/result/result?data=${encoded}`,
      });

      this.setData({ status: 'done' });
    } catch (e) {
      wx.hideLoading();
      this.setData({
        status: 'error',
        errorMsg: e.message || '分析失败，请重试',
      });
    }
  },

  /** 存入历史 */
  saveToHistory(result) {
    try {
      const history = wx.getStorageSync('beauty_history') || [];
      const entry = {
        id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
        timestamp: Date.now(),
        score: result.beauty_score,
        grade: result.grade,
        hasFace: result.has_face,
        imagePath: this.data.imagePath,
        features: result.features || {},
        geo_dimensions: result.geo_dimensions || { available: false },
      };
      history.unshift(entry);
      if (history.length > app.globalData.maxHistory) {
        history.length = app.globalData.maxHistory;
      }
      wx.setStorageSync('beauty_history', history);
    } catch (_) { /* ignore */ }
  },

  /** 偏好选择器 */
  onPrefChange(e) {
    const idx = parseInt(e.detail.value);
    const name = this.data.prefList[idx];
    this.setData({ prefIndex: idx, currentPref: name });
    app.setPreference(name);
  },

  onPrefPickerOpen() {
    this.setData({ showPrefPicker: true });
  },

  onPrefPickerClose() {
    this.setData({ showPrefPicker: false });
  },

  onPrefSelect(e) {
    const idx = parseInt(e.currentTarget.dataset.index);
    const name = this.data.prefList[idx];
    this.setData({ prefIndex: idx, currentPref: name, showPrefPicker: false });
    app.setPreference(name);
  },

  /** 选项开关 */
  onRemoveBgToggle() {
    this.setData({ removeBg: !this.data.removeBg });
  },

  onQuickModeToggle() {
    this.setData({ quickMode: !this.data.quickMode });
  },

  /** 前往问卷 */
  goQuestionnaire() {
    wx.navigateTo({ url: '/pages/questionnaire/questionnaire' });
  },

  /** 重试 */
  onRetry() {
    this.doAnalyze();
  },

  /** 重新选图 */
  onRechoose() {
    this.setData({
      imagePath: '',
      status: 'idle',
      errorMsg: '',
      precheckResult: null,
    });
  },
});
