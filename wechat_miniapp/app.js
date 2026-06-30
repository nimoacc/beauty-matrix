// 颜值矩阵分析系统 — 微信小程序全局入口 v53.1
App({
  globalData: {
    // 版本号
    version: 'v53.1',
    // Flask 后端 API 地址 (部署后修改)
    apiBase: 'http://localhost:5000',
    // 当前审美偏好
    currentPref: '均衡审美',
    // 历史记录上限
    maxHistory: 100,
    // 偏好预设缓存
    preferencesCache: null,
    // 偏好预设列表
    PREF_LIST: [
      '均衡审美', '对称至上', '比例至上', '青春至上',
      '独特至上', '和谐至上', '成熟魅力', '韩系精致',
      '日系可爱', '欧美大气', '自然清新'
    ],
  },

  onLaunch() {
    // 恢复用户偏好设置
    const savedPref = wx.getStorageSync('current_pref');
    if (savedPref) {
      this.globalData.currentPref = savedPref;
    }

    // 预加载偏好列表
    this.loadPreferences();

    console.log('颜值矩阵分析系统 小程序启动');
    console.log('API 地址:', this.globalData.apiBase);
  },

  /** 加载审美偏好预设列表 */
  loadPreferences() {
    if (this.globalData.preferencesCache) return;
    wx.request({
      url: `${this.globalData.apiBase}/api/preferences`,
      method: 'GET',
      success: (res) => {
        if (res.statusCode === 200 && res.data.preferences) {
          this.globalData.preferencesCache = res.data.preferences;
          this.globalData.PREF_LIST = res.data.preferences.map(p => p.name);
        }
      },
      fail: () => {
        console.warn('无法连接后端，使用默认偏好列表');
      }
    });
  },

  /** 切换审美偏好 */
  setPreference(name) {
    this.globalData.currentPref = name;
    wx.setStorageSync('current_pref', name);
  },
});
