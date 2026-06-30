// 颜值矩阵分析 — 审美偏好问卷
const api = require('../../utils/api');
const app = getApp();

Page({
  data: {
    // 问卷
    questions: [],
    currentIndex: 0,
    answers: [], // 存储每题的选项索引

    // 状态
    status: 'loading', // loading | answering | result
    result: null,

    // 进度
    progress: 0,
  },

  onLoad() {
    this.loadQuestions();
  },

  async loadQuestions() {
    this.setData({ status: 'loading' });
    try {
      const res = await api.getQuestionnaire();
      const questions = res.questions || [];
      this.setData({
        questions,
        answers: new Array(questions.length).fill(-1),
        currentIndex: 0,
        status: 'answering',
        progress: 0,
      });
    } catch (e) {
      // 加载失败的话降级使用本地问卷
      this.setData({
        questions: this.getLocalQuestions(),
        answers: new Array(12).fill(-1),
        currentIndex: 0,
        status: 'answering',
        progress: 0,
      });
    }
  },

  /** 选择题 */
  onSelectOption(e) {
    const optionIdx = parseInt(e.currentTarget.dataset.index);
    const answers = [...this.data.answers];
    answers[this.data.currentIndex] = optionIdx;

    const progress = Math.round((answers.filter(a => a >= 0).length / answers.length) * 100);

    this.setData({ answers, progress });

    // 自动跳下一题
    setTimeout(() => {
      if (this.data.currentIndex < this.data.questions.length - 1) {
        this.setData({ currentIndex: this.data.currentIndex + 1 });
      } else {
        // 全部完成，提交
        this.submitAnswers(answers);
      }
    }, 400);
  },

  /** 上一题 */
  onPrev() {
    if (this.data.currentIndex > 0) {
      this.setData({ currentIndex: this.data.currentIndex - 1 });
    }
  },

  /** 下一题 */
  onNext() {
    if (this.data.answers[this.data.currentIndex] >= 0 &&
        this.data.currentIndex < this.data.questions.length - 1) {
      this.setData({ currentIndex: this.data.currentIndex + 1 });
    }
  },

  /** 提交所有答案 */
  async submitAnswers(answers) {
    this.setData({ status: 'loading' });
    wx.showLoading({ title: '分析中...', mask: true });

    try {
      const result = await api.submitQuestionnaire(answers);
      this.setData({ status: 'result', result });
    } catch (e) {
      // 本地计算降级方案
      const result = this.computeLocalResult(answers);
      this.setData({ status: 'result', result });
    }

    wx.hideLoading();
  },

  /** 应用偏好 */
  onApplyPreference() {
    if (this.data.result && this.data.result.best_preset) {
      app.setPreference(this.data.result.best_preset.name);
      wx.showToast({ title: `已切换为: ${this.data.result.best_preset.name}`, icon: 'success' });
      setTimeout(() => wx.switchTab({ url: '/pages/index/index' }), 1200);
    }
  },

  /** 重新答题 */
  onRetry() {
    this.setData({
      answers: new Array(this.data.questions.length).fill(-1),
      currentIndex: 0,
      status: 'answering',
      result: null,
      progress: 0,
    });
  },

  /** 返回首页 */
  goHome() {
    wx.switchTab({ url: '/pages/index/index' });
  },

  /** 进度百分比 */
  get progressPercent() {
    const total = this.data.questions.length || 1;
    const answered = this.data.answers.filter(a => a >= 0).length;
    return Math.round(answered / total * 100);
  },

  // ================= 本地问卷降级 =================

  getLocalQuestions() {
    return [
      { id: 1, text: '你更看重面部的哪一方面？', options: ['对称协调', '比例结构', '年轻活力', '独特辨识度', '整体和谐'] },
      { id: 2, text: '哪种脸型对你更有吸引力？', options: ['鹅蛋脸', '瓜子脸', '圆脸', '方脸', '菱形脸'] },
      { id: 3, text: '你偏好怎样的眼睛？', options: ['大而有神', '眼睛形状美', '眉眼间距好', '眼角线条柔', '整体搭配和谐'] },
      { id: 4, text: '对鼻子的审美偏好？', options: ['鼻梁挺拔', '鼻头精致', '鼻子小巧', '有特色', '与脸型协调'] },
      { id: 5, text: '理想嘴唇的印象？', options: ['唇形对称', '比例适中', '唇色红润', '有辨识度', '与五官和谐'] },
      { id: 6, text: '对肤质的看重程度？', options: ['非常看重白净', '比较看重细腻', '一般即可', '不太在意', '自然就好'] },
      { id: 7, text: '你更欣赏什么类型的美？', options: ['精致完美型', '比例标准型', '青春可爱型', '个性独特型', '气质和谐型'] },
      { id: 8, text: '你认为"耐看"最重要的因素是？', options: ['面部对称', '五官比例', '显年轻', '有特色', '整体和谐感'] },
      { id: 9, text: '对"高级脸"的理解？', options: ['完美对称', '黄金比例', '年轻态', '辨识度极高', '越看越舒服'] },
      { id: 10, text: '哪种风格更让你心动？', options: ['完美女神/男神', '标准美人/帅哥', '可爱元气', '酷飒个性', '温柔耐看'] },
      { id: 11, text: '你认为美貌的"保鲜期"取决于？', options: ['骨架对称', '比例不随年龄变', '保持年轻感', '独特气质', '整体和谐度'] },
      { id: 12, text: '用一个词概括你的审美核心？', options: ['对称', '比例', '年轻', '独特', '和谐'] },
    ];
  },

  computeLocalResult(answers) {
    // 5维向量映射 C1对称 C2比例 C3年轻 C4独特 C5和谐
    const dimMap = {
      0: { weight: 1.0, dim: 0 }, // 对称
      1: { weight: 1.0, dim: 1 }, // 比例
      2: { weight: 1.0, dim: 2 }, // 年轻
      3: { weight: 1.0, dim: 3 }, // 独特
      4: { weight: 1.0, dim: 4 }, // 和谐
    };

    const dims = [0, 0, 0, 0, 0];
    answers.forEach((ans, i) => {
      if (ans >= 0 && ans < 5) {
        dims[dimMap[ans].dim] += 1;
      }
    });

    // 归一化
    const total = answers.length || 1;
    const vector = dims.map(v => Math.round(v / total * 100) / 100);

    // 匹配最近预设
    const presets = app.globalData.PREF_LIST || [
      '均衡审美', '对称至上', '比例至上', '青春至上',
      '独特至上', '和谐至上', '成熟魅力', '韩系精致',
      '日系可爱', '欧美大气', '自然清新'
    ];

    // 预设特征向量(简化匹配)
    const presetVectors = {
      '对称至上': [0.8, 0.3, 0.2, 0.2, 0.4],
      '比例至上': [0.3, 0.8, 0.2, 0.2, 0.4],
      '青春至上': [0.2, 0.3, 0.8, 0.3, 0.4],
      '独特至上': [0.2, 0.2, 0.3, 0.8, 0.3],
      '和谐至上': [0.4, 0.4, 0.3, 0.2, 0.8],
      '均衡审美': [0.5, 0.5, 0.5, 0.5, 0.5],
      '韩系精致': [0.7, 0.6, 0.5, 0.3, 0.5],
      '日系可爱': [0.3, 0.4, 0.8, 0.4, 0.5],
      '欧美大气': [0.4, 0.5, 0.2, 0.7, 0.4],
      '自然清新': [0.3, 0.3, 0.6, 0.3, 0.6],
      '成熟魅力': [0.5, 0.5, 0.2, 0.6, 0.5],
    };

    let bestScore = -1;
    let bestName = '均衡审美';

    for (const [name, pv] of Object.entries(presetVectors)) {
      const dist = Math.sqrt(pv.reduce((sum, v, i) => sum + (v - vector[i]) ** 2, 0));
      const score = 1 - dist / Math.sqrt(5);
      if (score > bestScore) {
        bestScore = score;
        bestName = name;
      }
    }

    const allScores = Object.entries(presetVectors).map(([name, pv]) => {
      const dist = Math.sqrt(pv.reduce((sum, v, i) => sum + (v - vector[i]) ** 2, 0));
      return { name, score: Math.round((1 - dist / Math.sqrt(5)) * 100) / 100 };
    }).sort((a, b) => b.score - a.score);

    return {
      user_vector: vector,
      best_preset: { name: bestName, score: Math.round(bestScore * 100) / 100 },
      all_scores: allScores.slice(0, 5),
    };
  },
});
