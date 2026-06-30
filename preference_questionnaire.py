"""
审美偏好问卷模块
12 道选择题 → 计算用户5维审美向量 → 匹配最近预设
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from typing import Callable


# ═══════════════════════════════════════════════
#  问卷题目定义
# ═══════════════════════════════════════════════

QUESTIONS: list[dict[str, Any]] = [
    {
        'id': 1,
        'text': '看到一张脸，你最先注意到什么？',
        'options': [
            {'label': 'A. 左右脸是否对称', 'scores': [2, 0, 0, 0, 0]},
            {'label': 'B. 眼睛鼻子比例是否协调', 'scores': [0, 2, 0, 0, 0]},
            {'label': 'C. 皮肤好不好、年不年轻', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 有没有辨识度、特别不特别', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 整体舒不舒服、顺不顺眼', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 2,
        'text': '以下哪种描述最接近你的审美偏好？',
        'options': [
            {'label': 'A. 左右对称的建模脸才是标准美', 'scores': [2, 0, 0, 0, 1]},
            {'label': 'B. 三庭五眼黄金比例最重要', 'scores': [0, 2, 0, 0, 1]},
            {'label': 'C. 年轻感、元气感是王道', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 有特色、让人记住才是真美', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 舒服自然的整体感觉最打动人', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 3,
        'text': '你更欣赏哪种长相风格？',
        'options': [
            {'label': 'A. Angelababy 式的精致标准', 'scores': [1, 1, 0, 0, 1]},
            {'label': 'B. 刘亦菲 式的古典比例', 'scores': [0, 2, 0, 1, 1]},
            {'label': 'C. 赵露思 式的甜美年轻', 'scores': [0, 0, 2, 0, 1]},
            {'label': 'D. 舒淇 式的独特高级', 'scores': [0, 0, 0, 2, 1]},
            {'label': 'E. 高圆圆 式的温柔治愈', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 4,
        'text': '对于"瑕疵脸"（如不对称、小缺陷），你的态度是？',
        'options': [
            {'label': 'A. 不太能接受，美感打折明显', 'scores': [2, 0, 0, 0, 0]},
            {'label': 'B. 如果比例好可以忽略', 'scores': [0, 2, 0, 0, 0]},
            {'label': 'C. 年轻活力可以掩盖一切', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 瑕疵反而增添了独特魅力', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 整体协调的话小缺陷无所谓', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 5,
        'text': '你觉得整容能提升颜值吗？',
        'options': [
            {'label': 'A. 能，标准化变美', 'scores': [2, 0, 0, -1, 0]},
            {'label': 'B. 能改善比例问题就好', 'scores': [0, 2, 0, 0, 0]},
            {'label': 'C. 能，显年轻最重要', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 不一定，容易千篇一律失去特色', 'scores': [0, 0, -1, 2, 0]},
            {'label': 'E. 适度就好，自然最美', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 6,
        'text': '看到素颜和精致妆容的对比，你更关注？',
        'options': [
            {'label': 'A. 化妆让脸部更对称了', 'scores': [2, 0, 0, 0, 0]},
            {'label': 'B. 化妆改善了五官比例', 'scores': [0, 2, 0, 0, 0]},
            {'label': 'C. 化妆让皮肤显得更年轻', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 化妆后变得没辨识度了', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 整体气色和协调感提升了', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 7,
        'text': '一张评分9分的"完美脸" vs 一张评分6分但气质独特的脸，你更愿意多看？',
        'options': [
            {'label': 'A. 当然是9分的，客观美是硬道理', 'scores': [1, 1, 1, 0, 0]},
            {'label': 'B. 6分的，独特比标准更吸引我', 'scores': [0, 0, 0, 2, 1]},
            {'label': 'C. 看情况，因人而异', 'scores': [0, 0, 0, 0, 1]},
            {'label': 'D. 如果6分的特别年轻有活力就选它', 'scores': [0, 0, 2, 0, 0]},
        ]
    },
    {
        'id': 8,
        'text': '选一张你最喜欢的脸型：',
        'options': [
            {'label': 'A. 标准鹅蛋脸（对称优雅）', 'scores': [2, 0, 0, 0, 1]},
            {'label': 'B. 瓜子脸（比例精致）', 'scores': [1, 2, 0, 0, 0]},
            {'label': 'C. 圆脸（显嫩显小）', 'scores': [0, 0, 2, 0, 1]},
            {'label': 'D. 方脸/高级脸（有气场）', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 心形脸（柔和自然）', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 9,
        'text': '对于皮肤，你更看重？',
        'options': [
            {'label': 'A. 白就是王道，一白遮百丑', 'scores': [0, 0, 1, 0, 0]},
            {'label': 'B. 细腻无瑕比颜色重要', 'scores': [0, 1, 1, 0, 1]},
            {'label': 'C. 光滑有弹性显年轻', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 小麦色/健康色更有魅力', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 肤色不影响对美的判断', 'scores': [0, 0, 0, 0, 1]},
        ]
    },
    {
        'id': 10,
        'text': '哪种特质最能让你觉得"好看"？',
        'options': [
            {'label': 'A. 精致——没有瑕疵的完美', 'scores': [2, 1, 0, 0, 0]},
            {'label': 'B. 比例——像画出来的一样标准', 'scores': [1, 2, 0, 0, 0]},
            {'label': 'C. 青春——扑面而来的少女感', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 记忆点——过目不忘的辨识度', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 舒服——如沐春风的和谐感', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 11,
        'text': '看到一位长相"不标准"但魅力四射的人，你觉得？',
        'options': [
            {'label': 'A. 虽然魅力强，但客观上说还是不够美', 'scores': [1, 0, 0, 0, 0]},
            {'label': 'B. 魅力可以弥补比例上的缺陷', 'scores': [0, 0, 0, 1, 1]},
            {'label': 'C. 年轻活力的人总是更吸引眼球', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 独特的美才是真正的高级美', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 整体协调的话什么都好说', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
    {
        'id': 12,
        'text': '总结一下你的审美哲学：',
        'options': [
            {'label': 'A. 美是客观的，有绝对标准', 'scores': [1, 1, 0, 0, 0]},
            {'label': 'B. 美在比例，数学不会骗人', 'scores': [0, 2, 0, 0, 1]},
            {'label': 'C. 青春即正义，年轻就是美', 'scores': [0, 0, 2, 0, 0]},
            {'label': 'D. 美在独特，千篇一律不叫美', 'scores': [0, 0, 0, 2, 0]},
            {'label': 'E. 情人眼里出西施，感觉最重要', 'scores': [0, 0, 0, 0, 2]},
        ]
    },
]


# ═══════════════════════════════════════════════
#  问卷计算
# ═══════════════════════════════════════════════

# 预设权重向量 (C1=对称, C2=比例, C3=年轻, C4=独特, C5=和谐)
PRESET_VECTORS: dict[str, tuple[float, ...]] = {
    '均衡审美': (1.0, 1.0, 1.0, 1.0, 1.0),
    '对称至上': (2.0, 1.0, 0.8, 0.6, 1.0),
    '比例至上': (1.0, 2.0, 0.8, 0.6, 1.0),
    '青春至上': (0.8, 0.8, 2.0, 0.6, 1.0),
    '韩系精致': (1.2, 1.2, 1.5, 0.8, 1.2),
    '日系自然': (0.9, 0.9, 1.0, 1.5, 1.2),
    '欧美飒爽': (1.0, 1.0, 0.7, 2.0, 0.9),
    '古典东方': (1.2, 1.3, 0.9, 0.8, 1.3),
    '温柔治愈': (0.9, 1.0, 1.2, 0.7, 1.4),
    '英气俊朗': (1.3, 1.1, 0.8, 1.8, 0.9),
    '高级超模': (1.0, 1.0, 0.6, 2.0, 0.7),
}


def compute_user_vector(answers: list[int]) -> list[float]:
    """
    根据12题答案计算用户5维审美向量
    
    Args:
        answers: 每题选择索引 (0-based, 对应 A/B/C/D/E)
    
    Returns:
        [C1, C2, C3, C4, C5] 5维权重向量
    """
    if len(answers) != len(QUESTIONS):
        raise ValueError(f'需要 {len(QUESTIONS)} 题答案，只提供了 {len(answers)} 题')
    
    # 累计5维得分
    total = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    for i, choice_idx in enumerate(answers):
        if choice_idx < 0 or choice_idx >= len(QUESTIONS[i]['options']):
            continue
        scores = QUESTIONS[i]['options'][choice_idx]['scores']
        for j in range(5):
            total[j] += scores[j]
    
    # 归一化到 [0, 2] 范围
    max_possible = 2.0 * len(QUESTIONS)
    normalized = [max(0.2, min(2.0, (v / max_possible * 4 + 1))) for v in total]
    
    return [round(v, 2) for v in normalized]


def match_best_preset(user_vector: list[float]) -> dict[str, Any]:
    """
    将用户向量匹配到最近的预设审美
    
    Returns:
        { 'name': '', 'distance': float, 'all_matches': [...] }
    """
    best_name = '均衡审美'
    best_dist = float('inf')
    all_matches = []
    
    for name, preset_vec in PRESET_VECTORS.items():
        # 余弦距离
        dot = sum(a * b for a, b in zip(user_vector, preset_vec))
        norm_u = math.sqrt(sum(a * a for a in user_vector))
        norm_p = math.sqrt(sum(a * a for a in preset_vec))
        
        if norm_u > 0 and norm_p > 0:
            cos_sim = dot / (norm_u * norm_p)
            distance = 1.0 - cos_sim
        else:
            distance = 1.0
        
        all_matches.append({
            'name': name,
            'distance': round(distance, 4),
            'similarity': round(1 - distance, 4),
        })
        
        if distance < best_dist:
            best_dist = distance
            best_name = name
    
    all_matches.sort(key=lambda m: m['distance'])
    
    return {
        'matched_preset': best_name,
        'distance': round(best_dist, 4),
        'user_vector': user_vector,
        'all_matches': all_matches[:5],  # 前5名
    }


def get_all_questions() -> list[dict[str, Any]]:
    """获取所有问卷题目（不含分数信息）"""
    return [
        {
            'id': q['id'],
            'text': q['text'],
            'options': [opt['label'] for opt in q['options']]
        }
        for q in QUESTIONS
    ]


# ═══════════════════════════════════════════════
#  GUI 模式 (v38: 集成到桌面应用中)
# ═══════════════════════════════════════════════

def _has_tk() -> bool:
    """检查 tkinter 是否可用"""
    try:
        import tkinter  # noqa: F811
        _ = tkinter.Tk
        return True
    except ImportError:
        return False


class PreferenceQuestionnaireGUI:
    """审美偏好问卷 GUI 窗口 (v38 集成版)"""

    tk: Any
    ttk: Any
    messagebox: Any
    on_complete_callback: Any
    current_q: int
    answers: list[int]
    result_vector: list[float] | None
    closest_name: str | None
    similarity: float
    root: Any
    title_label: Any
    progress_var: Any
    progress: Any
    progress_label: Any
    question_label: Any
    options_frame: Any
    option_var: Any
    option_radios: list[Any]
    btn_frame: Any
    prev_btn: Any
    next_btn: Any
    complete_btn: Any

    def __init__(self, root: Any = None, on_complete: Any = None):
        import tkinter as tk
        from tkinter import ttk, messagebox

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.on_complete_callback = on_complete
        self.current_q = 0
        self.answers = [-1] * len(QUESTIONS)
        self.result_vector = None
        self.closest_name = None
        self.similarity = 0.0

        if root is None:
            self.root = tk.Toplevel()
        else:
            self.root = tk.Toplevel(root)

        self.root.title('审美偏好问卷')
        self.root.geometry('560x460')
        self.root.resizable(True, True)
        self._build_ui()

    def _build_ui(self):
        """构建 UI"""
        # 标题
        self.title_label = self.ttk.Label(
            self.root,
            text='发现你的审美DNA',
            font=('Microsoft YaHei', 16, 'bold'),
        )
        self.title_label.pack(pady=(15, 5))

        # 进度条
        self.progress_var = self.tk.DoubleVar(value=0)
        self.progress = self.ttk.Progressbar(
            self.root, variable=self.progress_var, maximum=len(QUESTIONS),
        )
        self.progress.pack(fill='x', padx=30, pady=(0, 10))

        # 进度文字
        self.progress_label = self.ttk.Label(
            self.root, text='', font=('Microsoft YaHei', 10),
        )
        self.progress_label.pack()

        # 问题文字
        self.question_label = self.ttk.Label(
            self.root, text='', font=('Microsoft YaHei', 13),
            wraplength=500, justify='left',
        )
        self.question_label.pack(pady=(15, 10), padx=30)

        # 选项框架
        self.options_frame = self.ttk.Frame(self.root)
        self.options_frame.pack(fill='both', expand=True, padx=30, pady=10)

        # 选项变量
        self.option_var = self.tk.IntVar(value=-1)
        self.option_radios = []

        # 按钮框架
        self.btn_frame = self.ttk.Frame(self.root)
        self.btn_frame.pack(fill='x', padx=30, pady=(5, 15))

        self.prev_btn = self.ttk.Button(
            self.btn_frame, text='← 上一题', command=self._prev_question,
        )
        self.prev_btn.pack(side='left')

        self.next_btn = self.ttk.Button(
            self.btn_frame, text='下一题 →', command=self._next_question,
        )
        self.next_btn.pack(side='right')

        self.complete_btn = self.ttk.Button(
            self.btn_frame, text='✓ 完成', command=self._on_complete,
            state='disabled',
        )
        self.complete_btn.pack(side='right', padx=(0, 10))

        self._show_question()

    def _show_question(self):
        """显示当前问题"""
        q = QUESTIONS[self.current_q]

        # 更新进度
        self.progress_var.set(self.current_q + 1)
        answered_count = sum(1 for a in self.answers if a >= 0)
        self.progress_label.config(
            text=f'第 {self.current_q + 1} / {len(QUESTIONS)} 题  ({answered_count} 题已答)',
        )

        # 问题文字
        self.question_label.config(text=f'Q{self.current_q + 1}. {q["text"]}')

        # 清空选项
        for widget in self.options_frame.winfo_children():
            widget.destroy()
        self.option_radios.clear()

        # 显示选项
        self.option_var.set(self.answers[self.current_q])
        for j, opt in enumerate(q['options']):
            rb = self.ttk.Radiobutton(
                self.options_frame,
                text=opt['label'],
                variable=self.option_var,
                value=j,
                command=self._on_option_selected,
            )
            rb.pack(anchor='w', pady=4)
            self.option_radios.append(rb)

        # 按钮状态
        self.prev_btn.config(state='normal' if self.current_q > 0 else 'disabled')
        is_last = self.current_q == len(QUESTIONS) - 1
        self.next_btn.pack_forget() if is_last else self.next_btn.pack(side='right')
        if is_last:
            self.complete_btn.config(state='normal' if answered_count >= len(QUESTIONS) else 'disabled')

    def _on_option_selected(self):
        """选项被选中"""
        choice = self.option_var.get()
        if choice >= 0:
            self.answers[self.current_q] = choice

        answered_count = sum(1 for a in self.answers if a >= 0)
        self.progress_label.config(
            text=f'第 {self.current_q + 1} / {len(QUESTIONS)} 题  ({answered_count} 题已答)',
        )

        if self.current_q == len(QUESTIONS) - 1:
            self.complete_btn.config(state='normal' if answered_count >= len(QUESTIONS) else 'disabled')

    def _prev_question(self):
        """上一题"""
        if self.current_q > 0:
            self.current_q -= 1
            self._show_question()

    def _next_question(self):
        """下一题"""
        if self.current_q < len(QUESTIONS) - 1:
            self.current_q += 1
            self._show_question()

    def _on_complete(self):
        """完成问卷"""
        if any(a < 0 for a in self.answers):
            self.messagebox.showwarning('提示', '请回答所有问题后再完成。')
            return

        # 计算结果
        user_vec = compute_user_vector(self.answers)
        result = match_best_preset(user_vec)
        self.result_vector = user_vec
        self.closest_name = result['matched_preset']
        self.similarity = result['all_matches'][0]['similarity']

        self._show_result()

    def _show_result(self):
        """显示结果"""
        for widget in self.root.winfo_children():
            widget.destroy()

        self.root.title('你的审美DNA')

        # 结果展示
        title = self.ttk.Label(
            self.root,
            text='你的审美DNA分析结果',
            font=('Microsoft YaHei', 16, 'bold'),
        )
        title.pack(pady=(20, 10))

        # 匹配预设
        preset_label = self.ttk.Label(
            self.root,
            text=f'最匹配审美风格: {self.closest_name}',
            font=('Microsoft YaHei', 14),
            foreground='#e94560',
        )
        preset_label.pack(pady=5)

        sim_label = self.ttk.Label(
            self.root,
            text=f'匹配度: {self.similarity:.1%}',
            font=('Microsoft YaHei', 12),
        )
        sim_label.pack(pady=5)

        # 维度展示
        dim_names = ['对称 (C1)', '比例 (C2)', '年轻 (C3)', '独特 (C4)', '和谐 (C5)']
        dim_frame = self.ttk.Frame(self.root)
        dim_frame.pack(pady=15, padx=30, fill='x')

        for _i, (name, val) in enumerate(zip(dim_names, self.result_vector or [])):
            row = self.ttk.Frame(dim_frame)
            row.pack(fill='x', pady=2)
            self.ttk.Label(row, text=name, width=12, anchor='e').pack(side='left', padx=(0, 10))

            bar = self.ttk.Progressbar(row, value=val * 50, maximum=100, length=200)
            bar.pack(side='left')

            self.ttk.Label(row, text=f'{val:.2f}', width=5).pack(side='left', padx=(5, 0))

        # 解释
        if self.closest_name == '独特至上':
            interpret = '你是一位「个性派」审美者 — 独特比标准更重要。'
        elif self.closest_name == '和谐至上':
            interpret = '你是一位「和谐派」审美者 — 整体协调是第一要义。'
        elif self.closest_name == '对称至上' or self.closest_name == '比例至上':
            interpret = '你是一位「经典派」审美者 — 相信数学般的精确美。'
        elif self.closest_name == '青春至上':
            interpret = '你是一位「青春派」审美者 — 年轻朝气是美的核心。'
        else:
            interpret = '你拥有独特的审美视角，不囿于单一标准。'

        interp_label = self.ttk.Label(
            self.root, text=interpret,
            font=('Microsoft YaHei', 11),
            wraplength=500,
        )
        interp_label.pack(pady=15)

        # 按钮
        btn_frame = self.ttk.Frame(self.root)
        btn_frame.pack(pady=(10, 20))

        self.ttk.Button(
            btn_frame, text='重新测试', command=self._reset,
        ).pack(side='left', padx=5)

        self.ttk.Button(
            btn_frame, text='应用偏好', command=self._apply_result,
        ).pack(side='left', padx=5)

    def _apply_result(self):
        """应用结果到主窗口"""
        if self.on_complete_callback:
            self.on_complete_callback(self)
        self.root.destroy()

    def _reset(self):
        """重新测试"""
        self.current_q = 0
        self.answers = [-1] * len(QUESTIONS)
        self.result_vector = None
        self.closest_name = None

        for widget in self.root.winfo_children():
            widget.destroy()

        self._build_ui()


def gui_mode(root: object | None = None, on_complete: Callable[..., None] | None = None) -> PreferenceQuestionnaireGUI | None:
    """启动 GUI 问卷模式"""
    if not _has_tk():
        print('tkinter 不可用，无法启动 GUI 模式')
        return None
    return PreferenceQuestionnaireGUI(root=root, on_complete=on_complete)


def cli_mode() -> dict[str, Any]:
    """命令行交互模式"""
    print('\n  审美偏好问卷系统')
    print('  ' + '-' * 40)

    answers = []
    for q in QUESTIONS:
        print(f'\nQ{q["id"]}. {q["text"]}')
        for opt in q['options']:
            print(f'  {opt["label"]}')
        while True:
            try:
                choice = input('  请选择 (A-E): ').strip().upper()
                idx = ord(choice) - ord('A')
                if 0 <= idx < len(q['options']):
                    answers.append(idx)
                    break
                print('  请输入 A-E')
            except (ValueError, IndexError):
                print('  请输入 A-E')

    user_vec = compute_user_vector(answers)
    result = match_best_preset(user_vec)

    print('\n' + '=' * 40)
    print('  分析结果')
    print('=' * 40)
    print(f'  最匹配审美风格: {result["matched_preset"]}')
    print(f'  用户向量: {user_vec}')
    print(f'\n  排名前3匹配:')
    for m in result['all_matches'][:3]:
        print(f'    {m["name"]}: 相似度 {m["similarity"]:.1%}')

    return result


def main():
    """主入口"""
    import sys

    if _has_tk() and '--cli' not in sys.argv:
        # 默认启动 GUI
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        app = gui_mode()
        if app:
            app.root.wait_window()
    else:
        _ = cli_mode()


if __name__ == '__main__':
    main()
