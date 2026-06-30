// 颜值矩阵分析系统 — API 工具模块
const app = getApp();

/**
 * 封装的请求方法
 */
function request(url, options = {}) {
  const app = getApp();
  const baseUrl = app.globalData.apiBase;
  
  return new Promise((resolve, reject) => {
    wx.request({
      url: `${baseUrl}${url}`,
      method: options.method || 'GET',
      data: options.data,
      header: options.header || {},
      timeout: options.timeout || 30000,
      success: (res) => {
        if (res.statusCode === 200) {
          resolve(res.data);
        } else {
          reject({ message: res.data.error || '请求失败', status: res.statusCode, data: res.data });
        }
      },
      fail: (err) => {
        reject({ message: '网络请求失败', err });
      }
    });
  });
}

/**
 * 上传图片分析
 * @param {string} filePath - 图片临时路径
 * @param {object} options - { pref_name, remove_bg, quick_mode }
 */
function analyzeImage(filePath, options = {}) {
  return new Promise((resolve, reject) => {
    const app = getApp();
    const prefName = options.pref_name || app.globalData.currentPref || '均衡审美';

    wx.uploadFile({
      url: `${app.globalData.apiBase}/api/analyze`,
      filePath: filePath,
      name: 'image',
      formData: {
        pref_name: prefName,
        remove_bg: options.remove_bg !== false ? 'true' : 'false',
        quick_mode: options.quick_mode ? 'true' : 'false',
        enhance_side: options.enhance_side ? 'true' : 'false',
        enhance_large: options.enhance_large ? 'true' : 'false',
      },
      timeout: 60000,
      success: (res) => {
        try {
          const data = JSON.parse(res.data);
          if (res.statusCode === 200 && !data.error) {
            resolve(data);
          } else {
            reject({ message: data.error || '分析失败' });
          }
        } catch (e) {
          reject({ message: '解析结果失败' });
        }
      },
      fail: (err) => {
        reject({ message: '上传失败，请检查网络连接' });
      }
    });
  });
}

/**
 * 图片纹理预检
 */
function precheckImage(filePath) {
  return new Promise((resolve, reject) => {
    const app = getApp();
    wx.uploadFile({
      url: `${app.globalData.apiBase}/api/precheck`,
      filePath: filePath,
      name: 'image',
      timeout: 30000,
      success: (res) => {
        try {
          const data = JSON.parse(res.data);
          if (res.statusCode === 200) {
            resolve(data);
          } else {
            reject({ message: data.error || '预检失败' });
          }
        } catch (e) {
          reject({ message: '解析结果失败' });
        }
      },
      fail: () => reject({ message: '预检请求失败' })
    });
  });
}

/**
 * 获取所有审美偏好预设
 */
function getPreferences() {
  return request('/api/preferences');
}

/**
 * 获取问卷题目
 */
function getQuestionnaire() {
  return request('/api/questionnaire');
}

/**
 * 提交问卷答案
 */
function submitQuestionnaire(answers) {
  return request('/api/questionnaire/compute', {
    method: 'POST',
    header: { 'Content-Type': 'application/json' },
    data: { answers }
  });
}

/**
 * 化妆模拟
 */
function simulateMakeup(baseScore, delta) {
  return request('/api/makeup/simulate', {
    method: 'POST',
    header: { 'Content-Type': 'application/json' },
    data: { base_score: baseScore, delta: delta }
  });
}

/**
 * 多偏好对比
 */
function comparePreferences(filePath) {
  return new Promise((resolve, reject) => {
    const app = getApp();
    wx.uploadFile({
      url: `${app.globalData.apiBase}/api/compare`,
      filePath: filePath,
      name: 'image',
      timeout: 30000,
      success: (res) => {
        try {
          const data = JSON.parse(res.data);
          if (res.statusCode === 200) {
            resolve(data);
          } else {
            reject({ message: data.error || '对比失败' });
          }
        } catch (e) {
          reject({ message: '解析结果失败' });
        }
      },
      fail: () => reject({ message: '网络请求失败' })
    });
  });
}

/**
 * 健康检查
 */
function healthCheck() {
  return request('/api/health');
}

module.exports = {
  request,
  analyzeImage,
  precheckImage,
  getPreferences,
  getQuestionnaire,
  submitQuestionnaire,
  simulateMakeup,
  comparePreferences,
  healthCheck,
};
