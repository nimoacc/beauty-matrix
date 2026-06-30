"""
颜值矩阵分析系统 — 桌面GUI版 v53.3
基于 tkinter + customtkinter, 集成: DPI感知/多人脸/批量/偏好建议/自定义偏好/化妆模拟/导出
v52: 性别感知风格推断 (几何启发式性别检测 + 男/女双套美学标签)
v53: 集成ML性别分类模型 + 报告时间戳 + 调试增强
v53.1: PM审查热修复 (批量geo_bonus补全 + 三端评分管线一致性)
v53.2: geo_bonus评分膨胀修复 (对称Sigmoid + 允许负数扣分)
v53.3: geo_bonus去除居中 + 双向偏差罚分 (过窄/过宽均扣分, 最优≈0.75)
"""
from __future__ import annotations
import os
import sys
import json
import time
import threading
import ctypes
import platform
from tkinter import filedialog, messagebox
from typing import Any

# pyright: reportUninitializedInstanceVariable=false, reportMissingTypeStubs=false

try:
    import customtkinter as ctk
except ImportError:
    print("请安装 customtkinter: pip install customtkinter")
    sys.exit(1)

import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beauty_core import *  # noqa: F403
from image_utils import *  # noqa: F403
from preference_questionnaire import *  # noqa: F403


# ═══════════════════════════════════════════════
#  DPI 感知 & 屏幕自适应 (v38)
# ═══════════════════════════════════════════════

_SCREEN_W = 1920
_SCREEN_H = 1080
_SCALE_BASE = 1.0
_SCALE_FONT = 1.0

# pyright: reportConstantRedefinition=false

def _init_dpi_awareness():
    """初始化各平台 DPI 感知, 返回 (w, h, scale)"""
    global _SCREEN_W, _SCREEN_H, _SCALE_BASE
    try:
        # Windows
        if platform.system() == 'Windows':
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        # macOS
        elif platform.system() == 'Darwin':
            pass  # macOS 原生 HiDPI

        # 获取屏幕分辨率
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        _SCREEN_W = root.winfo_screenwidth()
        _SCREEN_H = root.winfo_screenheight()
        root.destroy()

        _SCALE_BASE = min(_SCREEN_W / 1920, _SCREEN_H / 1080)
    except Exception:
        _SCREEN_W, _SCREEN_H, _SCALE_BASE = 1920, 1080, 1.0


def _init_screen_scale():
    """初始化完成后调用, 基于屏幕尺寸计算布局缩放"""
    global _SCALE_FONT
    _SCALE_FONT = _SCALE_BASE * max(0.85, min(1.1, (_SCREEN_W / 1920)))


_init_dpi_awareness()
_init_screen_scale()


def _sp(base_px: int) -> int:
    """缩放像素值"""
    return max(1, int(base_px * _SCALE_BASE))


def _fs(base_pt: int) -> int:
    """缩放字体大小"""
    return max(8, int(base_pt * _SCALE_FONT))


# ═══════════════════════════════════════════════
#  主题 & 自定义偏好存储路径
# ═══════════════════════════════════════════════

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ══════ v45 配色体系: 深紫靛蓝 ─═════
BG_ROOT    = "#08081a"   # 根窗口背景 (最深)
BG_DARK    = "#0f0f2e"   # 画布/控件暗背景
BG_CARD    = "#1a1a3e"   # 卡片/面板背景
BG_CARD_H  = "#252550"   # 卡片悬停
ACCENT     = "#7c3aed"   # 主强调色 (紫)
ACCENT_H   = "#a78bfa"   # 悬停亮紫
ACCENT2    = "#4c1d95"   # 辅色 (深紫)
ACCENT2_H  = "#6d28d9"   # 辅色悬停
GRADIENT_1 = "#7c3aed"   # 渐变起 (紫)
GRADIENT_2 = "#ec4899"   # 渐变终 (粉)
SUCCESS    = "#10b981"   # 高分绿
WARNING    = "#f59e0b"   # 中分黄
DANGER     = "#ef4444"   # 低分红
GOLD       = "#fbbf24"   # 金色高亮
GOLD2      = "#f59e0b"   # 暗金
TEXT_PRIMARY   = "#f1f5f9"   # 主文字
TEXT_SECONDARY = "#94a3b8"   # 副文字
TEXT_MUTED     = "#64748b"   # 弱文字
BORDER_COLOR   = "#2d2d5e"   # 边框

# 矩阵五段渐变色 [0,5] → 5段
MATRIX_C1 = "#ef4444"  # [0,1) 红
MATRIX_C2 = "#f59e0b"  # [1,2) 橙
MATRIX_C3 = "#fbbf24"  # [2,3) 金
MATRIX_C4 = "#34d399"  # [3,4) 青绿
MATRIX_C5 = "#10b981"  # [4,5] 翠绿
MATRIX_GRID = "#4a4a7a"  # 表格网格线
RING_BG        = "#1e1e4a"   # 环形仪表底色

_CUSTOM_PREF_PATH = os.path.join(os.path.dirname(__file__), 'custom_preferences.json')


# ═══════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════

class BeautyGUI(ctk.CTk):
    """桌面端颜值分析主窗口 — v38 增强版"""

    current_image: np.ndarray | None
    current_image_path: str | None
    current_display: Image.Image | None
    current_result: dict[str, Any] | None
    current_pref: str
    is_analyzing: bool
    stop_flag: bool
    face_rects: list[tuple[int, int, int, int]]
    face_results: list[dict[str, Any]]
    active_face_idx: int
    batch_folder: str | None
    batch_results: list[dict[str, Any]]
    batch_cancel: threading.Event
    custom_pref_store: dict[str, list[float]]
    tabview: ctk.CTkTabview
    tab_analyze: ctk.CTkFrame
    tab_batch: ctk.CTkFrame
    tab_pref: ctk.CTkFrame
    tab_questionnaire: ctk.CTkFrame
    status_label: ctk.CTkLabel
    img_canvas: ctk.CTkCanvas
    btn_select: ctk.CTkButton
    btn_analyze: ctk.CTkButton
    btn_stop: ctk.CTkButton
    check_remove_bg: ctk.CTkCheckBox
    check_quick: ctk.CTkCheckBox
    check_soft: ctk.CTkCheckBox
    check_side: ctk.CTkCheckBox
    check_large: ctk.CTkCheckBox
    precheck_card: ctk.CTkFrame
    precheck_label: ctk.CTkLabel
    precheck_detail: ctk.CTkLabel
    result_tabs: ctk.CTkTabview
    rtab_score: Any
    rtab_feats: Any
    rtab_matrix: Any
    rtab_compare: Any
    rtab_advice: Any
    score_text: ctk.CTkTextbox
    matrix_text: ctk.CTkTextbox
    compare_text: ctk.CTkTextbox
    advice_text: ctk.CTkTextbox
    face_nav_frame: ctk.CTkFrame
    face_nav_label: ctk.CTkLabel
    btn_prev_face: ctk.CTkButton
    btn_next_face: ctk.CTkButton
    btn_makeup: ctk.CTkButton
    btn_export: ctk.CTkButton
    btn_batch_folder: ctk.CTkButton
    batch_folder_label: ctk.CTkLabel
    btn_batch_start: ctk.CTkButton
    btn_batch_stop: ctk.CTkButton
    btn_batch_export: ctk.CTkButton
    batch_progress: ctk.CTkProgressBar
    batch_progress_label: ctk.CTkLabel
    batch_result_text: ctk.CTkTextbox
    pref_buttons: dict[str, ctk.CTkButton]
    weight_sliders: dict[str, ctk.CTkSlider]
    saved_pref_frame: ctk.CTkScrollableFrame
    q_answers: list[Any]
    q_widgets: list[Any]
    btn_submit: ctk.CTkButton
    q_result_label: ctk.CTkLabel
    photo_img: ImageTk.PhotoImage | None
    _img_offset: tuple[int, int]
    _img_scale: float
    _status_dot: ctk.CTkFrame
    _status_dot_label: ctk.CTkLabel
    _beauty_gauge: ctk.CTkCanvas | None
    _feat_bars: ctk.CTkCanvas | None

    def __init__(self):
        super().__init__()

        self.title("颜值矩阵分析系统 v53.1")
        w = _sp(1400)
        h = _sp(900)
        self.geometry(f"{w}x{h}")
        self.minsize(_sp(1000), _sp(700))
        self.configure(fg_color=BG_ROOT)

        # 状态变量
        self.current_image = None      # 原始 numpy 图片
        self.current_image_path = None
        self.current_display = None    # 显示用 PIL Image
        self.current_result = None     # 当前分析结果
        self.current_pref = "均衡审美"
        self.is_analyzing = False
        self.stop_flag = False

        # 多人脸状态
        self.face_rects = []           # [(x,y,w,h), ...]
        self.face_results = []         # [{score, grade, skin_tone, ...}, ...]
        self.active_face_idx = 0       # 当前选中的人脸索引

        # 批量分析状态
        self.batch_folder = None
        self.batch_results = []
        self.batch_cancel = threading.Event()

        # 自定义偏好存储
        self.custom_pref_store = {}
        self._load_custom_pref_store()

        # Canvas 坐标/缩放 (懒初始化)
        self._img_offset = (0, 0)
        self._img_scale = 1.0

        # 构建 UI
        self._build_ui()

        # Canvas 画布控件 (延迟引用)
        self._beauty_gauge = None
        self._feat_bars = None

    # ── DPI 辅助 ──
    def _dp(self, px: int) -> int:
        return _sp(px)

    def _font(self, family: str, base_pt: int, weight: str | None = None) -> tuple[str, int] | tuple[str, int, str]:
        if weight:
            return (family, _fs(base_pt), weight)
        return (family, _fs(base_pt))

    # ── 状态指示 ──
    def _set_status(self, text: str, color: str = SUCCESS, status_text: str = "就绪") -> None:
        """统一设置状态指示灯 + 状态栏文字"""
        self._status_dot.configure(fg_color=color)
        self._status_dot_label.configure(text=status_text, text_color=color)
        self.status_label.configure(text=text)

    # ── 构建界面 ──
    def _build_ui(self) -> None:
        """构建完整界面"""
        # ═══ 顶部标题栏 ═══
        title_frame = ctk.CTkFrame(
            self, fg_color=BG_CARD, corner_radius=0, height=_sp(56),
            border_width=1, border_color=BORDER_COLOR,
        )
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        # 左侧 Logo 区
        logo_frame = ctk.CTkFrame(title_frame, fg_color="transparent")
        logo_frame.pack(side="left", padx=_sp(16), pady=_sp(6))

        # 渐变 Logo 文字 (用两段不同色模拟)
        ctk.CTkLabel(
            logo_frame, text="✦",
            font=self._font("Segoe UI Symbol", 22),
            text_color=GRADIENT_2,
        ).pack(side="left", padx=(0, _sp(6)))

        ctk.CTkLabel(
            logo_frame, text="颜值矩阵",
            font=self._font("Microsoft YaHei", 18, "bold"),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        ctk.CTkLabel(
            logo_frame, text="v53.1",
            font=self._font("Consolas", 10),
            text_color=TEXT_MUTED,
        ).pack(side="left", padx=(_sp(6), _sp(12)))

        # 分隔线
        ctk.CTkFrame(
            logo_frame, width=1, height=_sp(22), fg_color=BORDER_COLOR,
        ).pack(side="left", padx=_sp(4))

        ctk.CTkLabel(
            logo_frame, text="AI 美学评估引擎",
            font=self._font("Microsoft YaHei", 10),
            text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(_sp(8), 0))

        # 右侧状态指示
        self._status_dot = ctk.CTkFrame(
            title_frame, width=_sp(8), height=_sp(8),
            fg_color=SUCCESS, corner_radius=4,
        )
        self._status_dot.pack(side="right", padx=(0, _sp(12)))
        self._status_dot_label = ctk.CTkLabel(
            title_frame, text="就绪",
            font=self._font("Microsoft YaHei", 10),
            text_color=TEXT_MUTED,
        )
        self._status_dot_label.pack(side="right", padx=(0, _sp(4)))

        # ═══ 主内容区域 (TabView) ═══
        self.tabview = ctk.CTkTabview(
            self, fg_color=BG_ROOT,
            segmented_button_fg_color=BG_DARK,
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color=ACCENT_H,
            segmented_button_unselected_color=BG_DARK,
            segmented_button_unselected_hover_color=BG_CARD_H,
        )
        self.tabview.pack(fill="both", expand=True, padx=_sp(10), pady=(_sp(6), _sp(2)))

        self.tab_analyze = self.tabview.add(" 🔍 分析 ")
        self.tab_batch = self.tabview.add(" 📋 批量 ")
        self.tab_pref = self.tabview.add(" 🎨 偏好 ")
        self.tab_questionnaire = self.tabview.add(" 📝 问卷 ")

        self._build_analyze_tab()
        self._build_batch_tab()
        self._build_pref_tab()
        self._build_questionnaire_tab()

        # ═══ 底部状态栏 ═══
        status_bar = ctk.CTkFrame(
            self, fg_color=BG_CARD, corner_radius=0, height=_sp(30),
            border_width=1, border_color=BORDER_COLOR,
        )
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            status_bar,
            text="💡 就绪 — 点击「选择图片」开始颜值分析",
            font=self._font("Microsoft YaHei", 10),
            text_color=TEXT_SECONDARY,
        )
        self.status_label.pack(side="left", padx=_sp(14), pady=_sp(3))

    # ═══════════════════════════════════════════
    #  Tab 0: 分析 (图片 + 结果 + 多人脸)
    # ═══════════════════════════════════════════
    def _build_analyze_tab(self):
        """构建分析主Tab: 左侧图片, 右侧结果"""
        tab = self.tab_analyze
        tab.configure(fg_color=BG_ROOT)

        # ═══ 左侧面板 (图片区) ═══
        left_panel = ctk.CTkFrame(
            tab, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=BORDER_COLOR,
        )
        left_panel.pack(side="left", fill="both", expand=True, padx=(_sp(6), _sp(4)), pady=_sp(6))

        # 图片标题栏
        img_header = ctk.CTkFrame(left_panel, fg_color="transparent")
        img_header.pack(fill="x", padx=_sp(12), pady=(_sp(10), _sp(4)))

        ctk.CTkLabel(
            img_header, text="📷",
            font=self._font("Segoe UI Emoji", 14),
        ).pack(side="left", padx=(0, _sp(4)))

        ctk.CTkLabel(
            img_header, text="图片预览",
            font=self._font("Microsoft YaHei", 12, "bold"),
            text_color=TEXT_PRIMARY,
        ).pack(side="left")

        ctk.CTkLabel(
            img_header, text="点击人脸框切换",
            font=self._font("Microsoft YaHei", 9),
            text_color=TEXT_MUTED,
        ).pack(side="right")

        # 图片展示 Canvas (带内边框效果)
        canvas_holder = ctk.CTkFrame(
            left_panel, fg_color=BG_DARK, corner_radius=8,
            border_width=1, border_color=BORDER_COLOR,
        )
        canvas_holder.pack(fill="both", expand=True, padx=_sp(12), pady=(0, _sp(8)))

        self.img_canvas = ctk.CTkCanvas(
            canvas_holder,
            bg=BG_DARK, highlightthickness=0,
            width=_sp(500), height=_sp(550),
            bd=0, relief="flat",
        )
        self.img_canvas.pack(fill="both", expand=True, padx=_sp(2), pady=_sp(2))
        self.img_canvas.bind("<Button-1>", self._on_canvas_click)

        # 绘制欢迎引导
        self._draw_welcome_on_canvas()

        # 按钮区
        img_btn_frame = ctk.CTkFrame(left_panel, fg_color="transparent")
        img_btn_frame.pack(fill="x", padx=_sp(12), pady=(0, _sp(4)))

        self.btn_select = ctk.CTkButton(
            img_btn_frame, text="📂 选择图片",
            command=self._select_image,
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            font=self._font("Microsoft YaHei", 11),
            corner_radius=8, border_width=0,
        )
        self.btn_select.pack(side="left", padx=(0, _sp(6)))

        self.btn_analyze = ctk.CTkButton(
            img_btn_frame, text="🔍 开始分析",
            command=self._start_analysis,
            fg_color=ACCENT, hover_color=ACCENT_H,
            font=self._font("Microsoft YaHei", 11),
            corner_radius=8, border_width=0,
        )
        self.btn_analyze.pack(side="left", padx=(0, _sp(6)))

        self.btn_stop = ctk.CTkButton(
            img_btn_frame, text="⏹ 停止",
            command=self._stop_analysis,
            fg_color="#3f3f5e", hover_color=DANGER,
            state="disabled",
            font=self._font("Microsoft YaHei", 11),
            corner_radius=8,
        )
        self.btn_stop.pack(side="left")

        # 选项区
        opt_card = ctk.CTkFrame(
            left_panel, fg_color=BG_DARK, corner_radius=8,
            border_width=1, border_color=BORDER_COLOR,
        )
        opt_card.pack(fill="x", padx=_sp(12), pady=(_sp(2), _sp(8)))

        opt_row1 = ctk.CTkFrame(opt_card, fg_color="transparent")
        opt_row1.pack(fill="x", padx=_sp(8), pady=(_sp(6), _sp(2)))

        self.check_remove_bg = ctk.CTkCheckBox(
            opt_row1, text="去背景", text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_H,
            border_color=BORDER_COLOR, checkmark_color="#fff",
        )
        self.check_remove_bg.pack(side="left", padx=(0, _sp(12)))
        self.check_remove_bg.select()

        self.check_quick = ctk.CTkCheckBox(
            opt_row1, text="快速模式", text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_H,
            border_color=BORDER_COLOR, checkmark_color="#fff",
        )
        self.check_quick.pack(side="left", padx=(0, _sp(12)))

        self.check_soft = ctk.CTkCheckBox(
            opt_row1, text="软检测", text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_H,
            border_color=BORDER_COLOR, checkmark_color="#fff",
        )
        self.check_soft.pack(side="left")

        opt_row2 = ctk.CTkFrame(opt_card, fg_color="transparent")
        opt_row2.pack(fill="x", padx=_sp(8), pady=(_sp(2), _sp(6)))

        self.check_side = ctk.CTkCheckBox(
            opt_row2, text="侧脸增强", text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_H,
            border_color=BORDER_COLOR, checkmark_color="#fff",
        )
        self.check_side.pack(side="left", padx=(0, _sp(12)))
        self.check_side.select()

        self.check_large = ctk.CTkCheckBox(
            opt_row2, text="大脸增强", text_color=TEXT_PRIMARY,
            fg_color=ACCENT, hover_color=ACCENT_H,
            border_color=BORDER_COLOR, checkmark_color="#fff",
        )
        self.check_large.pack(side="left", padx=(0, _sp(12)))
        self.check_large.select()

        # ═══ 右侧结果面板 ═══
        right_panel = ctk.CTkFrame(
            tab, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=BORDER_COLOR,
        )
        right_panel.pack(side="right", fill="both", expand=True, padx=(_sp(4), _sp(6)), pady=_sp(6))

        # 纹理预检状态卡片
        self.precheck_card = ctk.CTkFrame(
            right_panel, fg_color=BG_DARK, corner_radius=8,
            border_width=1, border_color=BORDER_COLOR,
        )
        self.precheck_card.pack(fill="x", padx=_sp(8), pady=(_sp(8), _sp(4)))

        self.precheck_label = ctk.CTkLabel(
            self.precheck_card,
            text="🔬 纹理预检: 等待图片加载",
            font=self._font("Microsoft YaHei", 10, "bold"),
            text_color=TEXT_SECONDARY,
        )
        self.precheck_label.pack(side="left", padx=_sp(10), pady=_sp(5))

        self.precheck_detail = ctk.CTkLabel(
            self.precheck_card,
            text="",
            font=self._font("Consolas", 9),
            text_color=TEXT_MUTED,
        )
        self.precheck_detail.pack(side="right", padx=_sp(10), pady=_sp(5))

        # 结果子标签
        self.result_tabs = ctk.CTkTabview(
            right_panel, fg_color=BG_CARD,
            segmented_button_fg_color=BG_DARK,
            segmented_button_selected_color=ACCENT2,
            segmented_button_selected_hover_color=ACCENT2_H,
            segmented_button_unselected_color=BG_DARK,
            segmented_button_unselected_hover_color=BG_CARD_H,
        )
        self.result_tabs.pack(fill="both", expand=True, padx=_sp(6), pady=(_sp(2), _sp(6)))

        self.rtab_score  = self.result_tabs.add(" 💎 评分 ")
        self.rtab_feats  = self.result_tabs.add(" 📐 维度 ")
        self.rtab_matrix = self.result_tabs.add(" 🧮 矩阵 ")
        self.rtab_compare= self.result_tabs.add(" 📊 对比 ")
        self.rtab_advice = self.result_tabs.add(" 💡 建议 ")

        # ── 评分Tab: 环形仪表 + 文字 ──
        self.score_text = ctk.CTkTextbox(
            self.rtab_score, font=("Consolas", _fs(11)),
            fg_color=BG_DARK, text_color=TEXT_PRIMARY, wrap="word",
            border_width=1, border_color=BORDER_COLOR, corner_radius=8,
        )
        self.score_text.pack(fill="both", expand=True, padx=_sp(4), pady=_sp(4))
        self.score_text.insert("1.0", "等待分析…\n\n💎 选择图片后点击「开始分析」")
        self.score_text.configure(state="disabled")

        # ── 维度Tab: Canvas柱状图 ──
        self._feat_bars = ctk.CTkCanvas(
            self.rtab_feats, bg=BG_DARK, highlightthickness=0,
            height=_sp(280),
        )
        self._feat_bars.pack(fill="both", expand=True, padx=_sp(4), pady=_sp(4))

        # ── 矩阵Tab ──
        self._matrix_canvas = ctk.CTkCanvas(
            self.rtab_matrix, bg=BG_DARK, highlightthickness=0,
            height=_sp(380),
        )
        self._matrix_canvas.pack(fill="both", expand=True, padx=_sp(4), pady=_sp(4))
        self._draw_matrix_placeholder()

        # ── 对比Tab ──
        self.compare_text = ctk.CTkTextbox(
            self.rtab_compare, font=("Consolas", _fs(11)),
            fg_color=BG_DARK, text_color=TEXT_PRIMARY, wrap="word",
            border_width=1, border_color=BORDER_COLOR, corner_radius=8,
        )
        self.compare_text.pack(fill="both", expand=True, padx=_sp(4), pady=_sp(4))
        self.compare_text.insert("1.0", "📊 分析后显示11种审美视角得分对比")
        self.compare_text.configure(state="disabled")

        # ── 建议Tab ──
        self.advice_text = ctk.CTkTextbox(
            self.rtab_advice, font=("Microsoft YaHei", _fs(11)),
            fg_color=BG_DARK, text_color=TEXT_PRIMARY, wrap="word",
            border_width=1, border_color=BORDER_COLOR, corner_radius=8,
        )
        self.advice_text.pack(fill="both", expand=True, padx=_sp(4), pady=_sp(4))
        self.advice_text.insert("1.0", "💡 分析后显示偏好驱动的颜值提升建议…")
        self.advice_text.configure(state="disabled")

        # ── 多人脸导航 ──
        self.face_nav_frame = ctk.CTkFrame(
            right_panel, fg_color=BG_DARK, corner_radius=8,
        )
        self.face_nav_frame.pack(fill="x", padx=_sp(6), pady=(0, _sp(2)))
        self.face_nav_frame.pack_forget()

        self.face_nav_label = ctk.CTkLabel(
            self.face_nav_frame, text="",
            font=self._font("Microsoft YaHei", 10, "bold"),
            text_color=TEXT_PRIMARY,
        )
        self.face_nav_label.pack(side="left", padx=_sp(8), pady=_sp(4))

        self.btn_prev_face = ctk.CTkButton(
            self.face_nav_frame, text="◀",
            command=self._prev_face,
            width=_sp(36), font=self._font("Microsoft YaHei", 12),
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            corner_radius=8,
        )
        self.btn_prev_face.pack(side="right", padx=(0, _sp(4)), pady=_sp(3))

        self.btn_next_face = ctk.CTkButton(
            self.face_nav_frame, text="▶",
            command=self._next_face,
            width=_sp(36), font=self._font("Microsoft YaHei", 12),
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            corner_radius=8,
        )
        self.btn_next_face.pack(side="right", padx=_sp(4), pady=_sp(3))

        # ── 操作按钮 ──
        action_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        action_frame.pack(fill="x", padx=_sp(6), pady=(0, _sp(6)))

        self.btn_makeup = ctk.CTkButton(
            action_frame, text="💄 化妆模拟",
            command=self._simulate_makeup,
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            state="disabled", corner_radius=8,
            font=self._font("Microsoft YaHei", 10),
        )
        self.btn_makeup.pack(side="left", padx=(0, _sp(6)))

        self.btn_export = ctk.CTkButton(
            action_frame, text="📥 导出报告",
            command=self._export_report,
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            state="disabled", corner_radius=8,
            font=self._font("Microsoft YaHei", 10),
        )
        self.btn_export.pack(side="left")

    # ═══════════════════════════════════════════
    #  Tab 1: 批量分析
    # ═══════════════════════════════════════════
    def _build_batch_tab(self) -> None:
        """构建批量分析Tab"""
        tab = self.tab_batch
        tab.configure(fg_color=BG_ROOT)

        # 顶部控制区
        ctrl_frame = ctk.CTkFrame(
            tab, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=BORDER_COLOR,
        )
        ctrl_frame.pack(fill="x", padx=_sp(10), pady=_sp(8))

        self.btn_batch_folder = ctk.CTkButton(
            ctrl_frame, text="📁 选择文件夹",
            command=self._batch_select_folder,
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            font=self._font("Microsoft YaHei", 11),
            corner_radius=8,
        )
        self.btn_batch_folder.pack(side="left", padx=_sp(12), pady=_sp(10))

        self.batch_folder_label = ctk.CTkLabel(
            ctrl_frame, text="未选择文件夹",
            font=self._font("Microsoft YaHei", 10),
            text_color=TEXT_MUTED,
        )
        self.batch_folder_label.pack(side="left", padx=_sp(10))

        self.btn_batch_export = ctk.CTkButton(
            ctrl_frame, text="📥 导出结果",
            command=self._batch_export_report,
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            state="disabled", corner_radius=8,
            font=self._font("Microsoft YaHei", 11),
        )
        self.btn_batch_export.pack(side="right", padx=_sp(12), pady=_sp(10))

        self.btn_batch_stop = ctk.CTkButton(
            ctrl_frame, text="⏹ 停止",
            command=self._batch_stop,
            fg_color="#3f3f5e", hover_color=DANGER,
            state="disabled", corner_radius=8,
            font=self._font("Microsoft YaHei", 11),
        )
        self.btn_batch_stop.pack(side="right", padx=(0, _sp(6)), pady=_sp(10))

        self.btn_batch_start = ctk.CTkButton(
            ctrl_frame, text="▶ 开始批量分析",
            command=self._batch_start_analysis,
            fg_color=ACCENT, hover_color=ACCENT_H,
            state="disabled", corner_radius=8,
            font=self._font("Microsoft YaHei", 11),
        )
        self.btn_batch_start.pack(side="right", padx=(0, _sp(6)), pady=_sp(10))

        # 进度条
        prog_frame = ctk.CTkFrame(tab, fg_color="transparent")
        prog_frame.pack(fill="x", padx=_sp(12), pady=_sp(2))

        self.batch_progress = ctk.CTkProgressBar(
            prog_frame, progress_color=ACCENT, fg_color=BG_CARD,
        )
        self.batch_progress.pack(fill="x", pady=_sp(2))
        self.batch_progress.set(0)

        self.batch_progress_label = ctk.CTkLabel(
            prog_frame, text="",
            font=self._font("Microsoft YaHei", 10),
            text_color=TEXT_MUTED,
        )
        self.batch_progress_label.pack()

        # 结果列表
        batch_content = ctk.CTkFrame(
            tab, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=BORDER_COLOR,
        )
        batch_content.pack(fill="both", expand=True, padx=_sp(10), pady=_sp(6))

        self.batch_result_text = ctk.CTkTextbox(
            batch_content,
            font=("Consolas", _fs(11)),
            fg_color=BG_DARK, text_color=TEXT_PRIMARY, wrap="word",
            border_width=0, corner_radius=8,
        )
        self.batch_result_text.pack(fill="both", expand=True, padx=_sp(8), pady=_sp(8))
        self.batch_result_text.insert("1.0", '📁 选择文件夹后点击「开始批量分析」…')
        self.batch_result_text.configure(state="disabled")

    # ═══════════════════════════════════════════
    #  Tab 2: 偏好设置
    # ═══════════════════════════════════════════
    def _build_pref_tab(self) -> None:
        """构建审美偏好Tab"""
        tab = self.tab_pref
        tab.configure(fg_color=BG_ROOT)

        # 左侧: 预设偏好
        left_col = ctk.CTkFrame(
            tab, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=BORDER_COLOR,
        )
        left_col.pack(side="left", fill="both", expand=True, padx=(_sp(6), _sp(4)), pady=_sp(6))

        ctk.CTkLabel(
            left_col, text="🎨 预设审美偏好",
            font=self._font("Microsoft YaHei", 13, "bold"),
            text_color=TEXT_PRIMARY,
        ).pack(pady=(_sp(10), _sp(8)))

        self.pref_buttons = {}
        pref_scroll = ctk.CTkScrollableFrame(left_col, fg_color="transparent")
        pref_scroll.pack(fill="both", expand=True, padx=_sp(8))

        for _, name in enumerate(PREFERENCE_PRESETS.keys()):
            is_default = name == '均衡审美'
            btn = ctk.CTkButton(
                pref_scroll,
                text=f"{'⭐ ' if is_default else '• '}{name}",
                font=self._font("Microsoft YaHei", 11),
                fg_color=ACCENT if is_default else ACCENT2,
                hover_color=ACCENT_H if is_default else ACCENT2_H,
                corner_radius=8,
                command=lambda n=name: self._select_pref(n),
            )
            btn.pack(fill="x", pady=_sp(2))
            self.pref_buttons[name] = btn

        # 右侧: 自定义权重
        right_col = ctk.CTkFrame(
            tab, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=BORDER_COLOR,
        )
        right_col.pack(side="right", fill="both", expand=True, padx=(_sp(4), _sp(6)), pady=_sp(6))

        ctk.CTkLabel(
            right_col,
            text="⚙ 自定义审美权重",
            font=self._font("Microsoft YaHei", 13, "bold"),
            text_color=TEXT_PRIMARY,
        ).pack(pady=(_sp(10), _sp(8)))

        self.weight_sliders = {}
        labels_cn = [
            ('C1: 对称性', 'symmetry'),
            ('C2: 比例感', 'proportion'),
            ('C3: 年轻度', 'youth_index'),
            ('C4: 独特性', 'uniqueness'),
            ('C5: 和谐度', 'harmony'),
        ]

        for label, _key in labels_cn:
            s_frame = ctk.CTkFrame(right_col, fg_color="transparent")
            s_frame.pack(fill="x", pady=_sp(3), padx=_sp(16))

            ctk.CTkLabel(
                s_frame, text=label,
                font=self._font("Microsoft YaHei", 10),
                text_color=TEXT_SECONDARY, width=_sp(85),
            ).pack(side="left")

            slider = ctk.CTkSlider(
                s_frame, from_=0, to=2, number_of_steps=15,
                progress_color=ACCENT, button_color=ACCENT,
                button_hover_color=ACCENT_H,
            )
            slider.set(1.0)
            slider.pack(side="left", fill="x", expand=True, padx=_sp(8))
            self.weight_sliders[label] = slider

        # 按钮组
        custom_btn_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        custom_btn_frame.pack(fill="x", padx=_sp(16), pady=_sp(10))

        ctk.CTkButton(
            custom_btn_frame, text="✓ 应用自定义",
            command=self._apply_custom_weights,
            fg_color=ACCENT, hover_color=ACCENT_H,
            corner_radius=8,
            font=self._font("Microsoft YaHei", 10),
        ).pack(side="left", padx=(0, _sp(6)))

        ctk.CTkButton(
            custom_btn_frame, text="💾 保存偏好",
            command=self._save_custom_pref,
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            corner_radius=8,
            font=self._font("Microsoft YaHei", 10),
        ).pack(side="left", padx=(0, _sp(6)))

        ctk.CTkButton(
            custom_btn_frame, text="↩ 恢复上次",
            command=self._restore_custom_pref,
            fg_color=ACCENT2, hover_color=ACCENT2_H,
            corner_radius=8,
            font=self._font("Microsoft YaHei", 10),
        ).pack(side="left")

        # 已保存偏好列表
        ctk.CTkLabel(
            right_col, text="📋 已保存的偏好",
            font=self._font("Microsoft YaHei", 11, "bold"),
            text_color=TEXT_SECONDARY,
        ).pack(anchor="w", padx=_sp(16), pady=(_sp(8), _sp(2)))

        self.saved_pref_frame = ctk.CTkScrollableFrame(
            right_col, fg_color=BG_DARK, corner_radius=8,
            border_width=1, border_color=BORDER_COLOR,
        )
        self.saved_pref_frame.pack(fill="both", expand=True, padx=_sp(16), pady=(0, _sp(10)))
        self._refresh_saved_pref_display()

    # ═══════════════════════════════════════════
    #  Tab 3: 问卷
    # ═══════════════════════════════════════════
    def _build_questionnaire_tab(self) -> None:
        """构建问卷Tab"""
        tab = self.tab_questionnaire
        tab.configure(fg_color=BG_ROOT)

        # 头部
        header_card = ctk.CTkFrame(
            tab, fg_color=BG_CARD, corner_radius=12,
            border_width=1, border_color=BORDER_COLOR,
        )
        header_card.pack(fill="x", padx=_sp(8), pady=(_sp(8), _sp(4)))

        ctk.CTkLabel(
            header_card,
            text="🧬 审美DNA发现",
            font=self._font("Microsoft YaHei", 18, "bold"),
            text_color=GRADIENT_2,
        ).pack(pady=(_sp(14), _sp(4)))

        ctk.CTkLabel(
            header_card,
            text="回答12道题，发现属于你的独特审美偏好类型",
            font=self._font("Microsoft YaHei", 11),
            text_color=TEXT_SECONDARY,
        ).pack(pady=(0, _sp(12)))

        # 问卷内容
        q_scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        q_scroll.pack(fill="both", expand=True, padx=_sp(8))

        self.q_answers = []
        self.q_widgets = []

        questions = get_all_questions()
        for q in questions:
            q_container = ctk.CTkFrame(
                q_scroll, fg_color=BG_CARD, corner_radius=8,
                border_width=1, border_color=BORDER_COLOR,
            )
            q_container.pack(fill="x", pady=_sp(3))

            ctk.CTkLabel(
                q_container,
                text=f"Q{q['id']}. {q['text']}",
                font=self._font("Microsoft YaHei", 11, "bold"),
                text_color=TEXT_PRIMARY,
                wraplength=_sp(400),
            ).pack(anchor="w", padx=_sp(8), pady=(_sp(8), _sp(3)))

            var = ctk.StringVar(value="")
            for opt in q['options']:
                rb = ctk.CTkRadioButton(
                    q_container, text=opt, variable=var, value=opt[0],
                    font=self._font("Microsoft YaHei", 10),
                    text_color=TEXT_PRIMARY,
                )
                rb.pack(anchor="w", padx=_sp(20), pady=_sp(1))

            self.q_answers.append(var)
            self.q_widgets.append(q_container)

        self.btn_submit = ctk.CTkButton(
            q_scroll,
            text="提交问卷，发现我的审美DNA",
            command=self._submit_questionnaire,
            fg_color=ACCENT, height=_sp(36),
            font=self._font("Microsoft YaHei", 12),
        )
        self.btn_submit.pack(pady=_sp(10))

        self.q_result_label = ctk.CTkLabel(
            q_scroll,
            text="",
            font=self._font("Microsoft YaHei", 12),
            text_color=ACCENT,
        )
        self.q_result_label.pack(pady=(0, _sp(10)))

    # ═══════════════════════════════════════════
    #  事件处理: 图片选择 & 预览
    # ═══════════════════════════════════════════
    def _select_image(self) -> None:
        """选择图片文件"""
        path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")],
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        """加载并显示图片"""
        try:
            # 用 v38 正常化加载
            img = load_and_normalize_image(path, source_type='path')
            self.current_image = img
            self.current_image_path = path
            self.current_result = None
            self.face_rects = []
            self.face_results = []
            self.active_face_idx = 0

            # 显示缩略图
            pil_img = Image.fromarray(img)
            self._update_canvas_preview(pil_img, face_rects=[])

            self._set_status(f"📷 已加载: {os.path.basename(path)}  ({img.shape[1]}×{img.shape[0]})")

            # 重置结果面板
            self._clear_results()

        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载图片: {e}")

    def _update_canvas_preview(self, pil_img: Image.Image, face_rects: list[tuple[int, int, int, int]] | None = None) -> None:
        """更新 Canvas 上的图片预览, 可选绘制人脸框"""
        canvas_w = self.img_canvas.winfo_width()
        canvas_h = self.img_canvas.winfo_height()
        if canvas_w < 50:
            canvas_w, canvas_h = _sp(500), _sp(550)

        # 缩放适配
        img_w, img_h = pil_img.size
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        dw, dh = int(img_w * scale), int(img_h * scale)

        resized = pil_img.resize((dw, dh), Image.Resampling.LANCZOS)
        self.current_display = resized
        self.photo_img = ImageTk.PhotoImage(resized)

        self.img_canvas.delete("all")
        x_off = (canvas_w - dw) // 2
        y_off = (canvas_h - dh) // 2
        self.img_canvas.create_image(x_off, y_off, anchor="nw", image=self.photo_img)

        # 存储偏移和缩放用于 canvas 点击计算
        self._img_offset = (x_off, y_off)
        self._img_scale = scale

        # 绘制人脸框
        if face_rects:
            for i, (x, y, w, h) in enumerate(face_rects):
                cx = int(x_off + x * scale)
                cy = int(y_off + y * scale)
                cw = int(w * scale)
                ch = int(h * scale)
                color = GOLD if i == self.active_face_idx else "#00ccff"
                self.img_canvas.create_rectangle(
                    cx, cy, cx + cw, cy + ch,
                    outline=color, width=2,
                )
                self.img_canvas.create_text(
                    cx + 5, cy - 10,
                    text=f"#{i + 1}", fill=color,
                    anchor="w", font=("Consolas", _fs(11), "bold"),
                )

    # ── Canvas 辅助绘制 ──
    def _draw_welcome_on_canvas(self) -> None:
        """在图片 Canvas 上绘制欢迎引导"""
        cw = int(self.img_canvas["width"]) if self.img_canvas["width"] else _sp(500)
        ch = int(self.img_canvas["height"]) if self.img_canvas["height"] else _sp(550)
        cx, cy = cw // 2, ch // 2

        self.img_canvas.delete("all")
        # 虚线边框
        for i in range(4):
            offset = i * 8
            self.img_canvas.create_rectangle(
                _sp(40) + offset, _sp(40) + offset,
                cw - _sp(40) - offset, ch - _sp(40) - offset,
                outline=BORDER_COLOR, dash=(6, 4), width=1,
            )
        self.img_canvas.create_text(
            cx, cy - _sp(20), text="📷",
            font=("Segoe UI Emoji", _fs(36)),
            fill=TEXT_MUTED,
        )
        self.img_canvas.create_text(
            cx, cy + _sp(30), text="点击「选择图片」加载照片",
            font=("Microsoft YaHei", _fs(12)),
            fill=TEXT_MUTED,
        )
        self.img_canvas.create_text(
            cx, cy + _sp(55), text="支持 JPG / PNG / BMP / WebP",
            font=("Microsoft YaHei", _fs(10)),
            fill=TEXT_MUTED,
        )

    @staticmethod
    def _score_color(score: float) -> str:
        """根据分数返回颜色"""
        if score >= 7.5:
            return SUCCESS
        elif score >= 5.0:
            return GOLD
        elif score >= 3.0:
            return WARNING
        else:
            return DANGER

    def _draw_score_gauge(self, total: float, grade: dict[str, Any]) -> None:
        """在评分Tab文本框中绘制环形仪表 (ASCII Art版本, 兼容所有字体)"""
        color = self._score_color(total)
        gauge_w = 30
        filled = int(total / 10 * (gauge_w - 2))

        lines = ["", ""]
        # 装饰线
        lines.append("  ┌" + "─" * (gauge_w + 2) + "┐")

        # 色彩条 (使用纯色条简化)
        bar = "█" * filled + "░" * (gauge_w - filled)
        lines.append(f"  │ {bar} │")

        lines.append("  └" + "─" * (gauge_w + 2) + "┘")
        lines.append("")
        lines.append(f"         {total:.1f}  / 10")
        lines.append(f"       {grade.get('label', '?')}")
        lines.append("")

        # Score decomposition
        lines.append("  ── 评分分解 ──")
        self.score_text.configure(state="normal")
        self.score_text.delete("1.0", "end")
        # 用 tag 上色
        for line in lines:
            self.score_text.insert("end", line + "\n")
        # Highlight score
        self.score_text.insert("end", f"\n  💎 颜值评分: ", "")
        idx = self.score_text.index("end-1c")
        self.score_text.insert("end", f"{total:.1f} / 10", "")
        self.score_text.tag_add("score_big", idx, "end-1c")
        self.score_text.tag_config("score_big", foreground=color,
                                    font=("Consolas", _fs(22), "bold"))
        self.score_text.insert("end", f"\n  🏷 等级: {grade.get('label', '?')}\n", "")
        self.score_text.configure(state="disabled")

    # ═══════════════════════════════════════════
    #  矩阵 Canvas 工具方法
    # ═══════════════════════════════════════════

    @staticmethod
    def _matrix_cell_color(v: float) -> str:
        """矩阵单元格五段渐变色 [0,5]"""
        if v < 1.0:
            return MATRIX_C1
        elif v < 2.0:
            return MATRIX_C2
        elif v < 3.0:
            return MATRIX_C3
        elif v < 4.0:
            return MATRIX_C4
        else:
            return MATRIX_C5

    @staticmethod
    def _matrix_text_color(bg: str) -> str:
        """根据背景亮度决定文字颜色"""
        # 红色/橙色用白色文字, 金黄/青绿/翠绿用深色文字
        if bg in (MATRIX_C1, MATRIX_C2):
            return "#ffffff"
        else:
            return "#1a1a2e"

    def _draw_matrix_placeholder(self, msg: str | None = None) -> None:
        """绘制矩阵占位提示 (未分析时)"""
        canvas = self._matrix_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 50:
            cw, ch = _sp(500), _sp(380)
        lines = [
            "🧮 5×5 美学矩阵",
            "",
            "行 (面部器官):  R1眼  R2鼻  R3唇  R4轮廓  R5肤质",
            "列 (审美维度):  C1对称性  C2比例  C3年轻  C4独特性  C5和谐度",
        ]
        if msg:
            lines = [msg]
        y = ch // 2 - len(lines) * _fs(10)
        for i, line in enumerate(lines):
            canvas.create_text(
                cw // 2, y + i * _fs(20),
                text=line, anchor="center",
                fill=TEXT_MUTED if msg else TEXT_SECONDARY,
                font=("Microsoft YaHei", _fs(11)),
            )

    def _draw_matrix_canvas(self, md: dict[str, Any]) -> None:
        """在矩阵Tab Canvas上绘制 Excel 风格表格"""
        canvas = self._matrix_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 50:
            cw, ch = _sp(500), _sp(380)

        # ── 布局参数 ──
        row_label_w = _sp(90)
        cell_w = _sp(76)
        cell_h = _sp(34)
        col_header_h = _sp(34)
        left_margin = _sp(30)
        top_start = _sp(58)  # 标题+副标题之后

        table_left = left_margin
        table_top = top_start
        table_right = table_left + row_label_w + 5 * cell_w
        table_bottom = table_top + col_header_h + 5 * cell_h

        n_rows = 5
        n_cols = 5

        col_headers_short = [n.replace('性', '').replace('度', '') for n in md['col_names']]
        row_names = md['row_names']
        matrix = md['matrix']

        # ── 标题 ──
        canvas.create_text(
            cw // 2, _sp(10),
            text="🧮 5×5 美学矩阵",
            anchor="n", fill=TEXT_PRIMARY,
            font=("Microsoft YaHei", _fs(13), "bold"),
        )
        canvas.create_text(
            cw // 2, _sp(36),
            text="行 → 面部器官 (R1~R5)    列 → 审美维度 (C1~C5)",
            anchor="n", fill=TEXT_SECONDARY,
            font=("Microsoft YaHei", _fs(9)),
        )

        # ── 绘制单元格背景 (先画背景, 再画线避免遮盖) ──
        # 列头行背景
        for j in range(5):
            x1 = table_left + row_label_w + j * cell_w
            canvas.create_rectangle(
                x1, table_top,
                x1 + cell_w, table_top + col_header_h,
                fill=BG_CARD_H, outline="", width=0,
            )
        # 左上角单元格背景
        canvas.create_rectangle(
            table_left, table_top,
            table_left + row_label_w, table_top + col_header_h,
            fill=BG_CARD, outline="", width=0,
        )
        # 行标签列背景
        for i in range(5):
            y1 = table_top + col_header_h + i * cell_h
            canvas.create_rectangle(
                table_left, y1,
                table_left + row_label_w, y1 + cell_h,
                fill=BG_CARD, outline="", width=0,
            )
        # 数据单元格背景 (五段渐变色)
        for i in range(5):
            for j in range(5):
                v = matrix[i][j]
                color = self._matrix_cell_color(v)
                x1 = table_left + row_label_w + j * cell_w
                y1 = table_top + col_header_h + i * cell_h
                canvas.create_rectangle(
                    x1, y1, x1 + cell_w, y1 + cell_h,
                    fill=color, outline="", width=0,
                )

        # ── 绘制网格线 ──
        # 外边框 (加粗)
        canvas.create_rectangle(
            table_left, table_top, table_right, table_bottom,
            outline=MATRIX_GRID, width=2,
        )
        # 水平内线
        for i in range(0, 6):
            y = table_top + col_header_h + i * cell_h
            w = 2 if i == 0 else 1  # 列头下加粗
            canvas.create_line(table_left, y, table_right, y, fill=MATRIX_GRID, width=w)
        # 垂直内线 (第一根在行标签右侧, 加粗)
        for j in range(0, 6):
            x = table_left + row_label_w + j * cell_w
            w = 2 if j == 0 else 1
            canvas.create_line(x, table_top, x, table_bottom, fill=MATRIX_GRID, width=w)

        # ── 绘制文字 ──
        # 列头
        for j in range(5):
            cx = table_left + row_label_w + j * cell_w + cell_w // 2
            cy = table_top + col_header_h // 2
            canvas.create_text(
                cx, cy,
                text=col_headers_short[j],
                fill=TEXT_PRIMARY,
                font=("Microsoft YaHei", _fs(10), "bold"),
            )
        # 行标签
        for i in range(5):
            cx = table_left + row_label_w // 2
            cy = table_top + col_header_h + i * cell_h + cell_h // 2
            canvas.create_text(
                cx, cy,
                text=row_names[i],
                fill=TEXT_SECONDARY,
                font=("Microsoft YaHei", _fs(10)),
            )
        # 数据单元格
        for i in range(5):
            for j in range(5):
                v = matrix[i][j]
                color = self._matrix_cell_color(v)
                txt_color = self._matrix_text_color(color)
                cx = table_left + row_label_w + j * cell_w + cell_w // 2
                cy = table_top + col_header_h + i * cell_h + cell_h // 2
                canvas.create_text(
                    cx, cy,
                    text=f"{v:+.2f}",
                    fill=txt_color,
                    font=("Consolas", _fs(11), "bold"),
                )

        # ── 底部统计信息 ──
        info_y = table_bottom + _sp(14)
        det = md['det']
        ds = f"{det:.6f}" if abs(det) < 100 else f"{det:.2f}"

        lines = [
            (f"det(矩阵) = {ds}", TEXT_PRIMARY),
            ("", TEXT_MUTED),
        ]
        if abs(det) < 1e-12:
            lines.append(("▸ 行列式≈0 → 各维度贡献均衡", TEXT_SECONDARY))
        elif det > 0:
            lines.append((f"▸ 审美秩 {abs(det):.2f} → 正定 / 和谐美感", SUCCESS))
        else:
            lines.append((f"▸ 审美秩 {abs(det):.2f} → 负定 / 冲突张力", DANGER))

        lines.append((f"▸ 范围: [{md['min_val']:.2f}, {md['max_val']:.2f}]", TEXT_SECONDARY))
        lines.append(("▸ 交叉值 = hill(器官)×hill(审美维度)×10", TEXT_SECONDARY))

        # 图例 (在 info 下方用色块表示)
        legend_y = info_y
        for i, (text, color) in enumerate(lines):
            if not text:
                legend_y += _fs(8)
                continue
            canvas.create_text(
                table_left + _sp(4), legend_y,
                text=text, anchor="w",
                fill=color, font=("Microsoft YaHei", _fs(9)),
            )
            legend_y += _fs(18)

        # 五段色块图例
        legend_y += _sp(4)
        legend_colors = [
            (MATRIX_C1, "0~1"), (MATRIX_C2, "1~2"),
            (MATRIX_C3, "2~3"), (MATRIX_C4, "3~4"), (MATRIX_C5, "4~5"),
        ]
        sw = _sp(28)
        sh = _sp(14)
        total_legend_w = len(legend_colors) * (sw + _sp(4)) - _sp(4)
        legend_start_x = table_left + (table_right - table_left - total_legend_w) // 2
        for k, (lcolor, llabel) in enumerate(legend_colors):
            lx = legend_start_x + k * (sw + _sp(4))
            canvas.create_rectangle(
                lx, legend_y, lx + sw, legend_y + sh,
                fill=lcolor, outline=MATRIX_GRID, width=1,
            )
            canvas.create_text(
                lx + sw // 2, legend_y + sh // 2,
                text=llabel,
                fill=self._matrix_text_color(lcolor),
                font=("Consolas", _fs(7), "bold"),
            )

    def _draw_feature_bars(self, feats: Any) -> None:
        """在维度Tab Canvas上绘制柱状图"""
        if self._feat_bars is None:
            return
        canvas = self._feat_bars
        canvas.delete("all")

        # 强制布局以确保 winfo 返回有效尺寸
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 50:
            cw, ch = _sp(420), _sp(300)

        if not hasattr(feats, 'as_dict'):
            canvas.create_text(
                cw // 2, ch // 2, text="无法获取特征数据",
                fill=TEXT_MUTED, font=("Microsoft YaHei", _fs(11)),
            )
            return

        try:
            feat_dict = feats.as_dict()
        except Exception:
            canvas.create_text(
                cw // 2, ch // 2, text="特征数据解析失败",
                fill=TEXT_MUTED, font=("Microsoft YaHei", _fs(11)),
            )
            return

        n = len(feat_dict)
        if n == 0:
            return

        pad_l, pad_r, pad_t, pad_b = _sp(90), _sp(40), _sp(20), _sp(30)
        bar_area_w = cw - pad_l - pad_r
        bar_area_h = ch - pad_t - pad_b
        bar_gap = bar_area_h / n

        labels_cn_map = {
            'symmetry': 'C1 对称性', 'proportion': 'C2 比例感',
            'youth_index': 'C3 年轻度', 'uniqueness': 'C4 独特性',
            'harmony': 'C5 和谐度',
            'eye_score': '眼部分值', 'nose_score': '鼻部分值',
            'lip_score': '唇部分值', 'contour_score': '轮廓分值',
            'skin_texture': '肤质分值',
        }

        max_val = max(max(feat_dict.values()), 0.1)

        for i, (key, val) in enumerate(feat_dict.items()):
            y_center = pad_t + bar_gap * i + bar_gap * 0.45
            bar_h = max(bar_gap * 0.55, _sp(8))

            # 标签
            label_text: str = labels_cn_map.get(key, "") or key[:12]
            canvas.create_text(
                pad_l - _sp(6), y_center,
                text=label_text, anchor="e",
                fill=TEXT_SECONDARY,
                font=("Microsoft YaHei", _fs(9)),
            )

            # 背景条
            canvas.create_rectangle(
                pad_l, y_center - bar_h / 2,
                pad_l + bar_area_w, y_center + bar_h / 2,
                fill=RING_BG, outline="", width=0,
            )

            # 数值条 (渐变色映射)
            ratio = val / max_val if max_val > 0 else 0
            bar_w = max(ratio * bar_area_w, _sp(2))
            bar_color = self._score_color(val) if key in ('symmetry', 'proportion', 'youth_index', 'uniqueness', 'harmony') else ACCENT

            canvas.create_rectangle(
                pad_l, y_center - bar_h / 2,
                pad_l + bar_w, y_center + bar_h / 2,
                fill=bar_color, outline="", width=0,
            )

            # 数值
            canvas.create_text(
                pad_l + bar_w + _sp(6), y_center,
                text=f"{val:.1f}", anchor="w",
                fill=TEXT_PRIMARY,
                font=("Consolas", _fs(10), "bold"),
            )

        # 标题
        canvas.create_text(
            cw // 2, pad_t - _sp(2),
            text="📐 10 维美学特征分布", anchor="n",
            fill=TEXT_PRIMARY,
            font=("Microsoft YaHei", _fs(11), "bold"),
        )

        # 底部分隔线
        canvas.create_line(
            pad_l, ch - pad_b + _sp(8),
            pad_l + bar_area_w, ch - pad_b + _sp(8),
            fill=BORDER_COLOR, width=1,
        )

    def _draw_feature_bars_fallback(self, feats: Any) -> None:
        """当 feats 不是 FaceFeatures 时的降级渲染"""
        if self._feat_bars is None:
            return
        canvas = self._feat_bars
        canvas.delete("all")
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 50:
            cw, ch = _sp(420), _sp(300)

        # 尝试从dict/list提取数据
        items = []
        if isinstance(feats, dict):
            items = list(feats.items())
        elif isinstance(feats, (list, tuple)):
            items = [(f"C{i+1}", v) for i, v in enumerate(feats[:10])]
        else:
            canvas.create_text(
                cw // 2, ch // 2,
                text=f"不支持的特征格式\n({type(feats).__name__})",
                fill=TEXT_MUTED, font=("Microsoft YaHei", _fs(11)),
            )
            return

        if not items:
            return

        n = len(items)
        pad_l, pad_r, pad_t, pad_b = _sp(90), _sp(40), _sp(20), _sp(30)
        bar_area_w = cw - pad_l - pad_r
        bar_area_h = ch - pad_t - pad_b
        bar_gap = bar_area_h / max(n, 1)
        max_val = max(max(v for _, v in items), 0.1)

        labels_cn_map = {
            'symmetry': 'C1 对称性', 'proportion': 'C2 比例感',
            'youth_index': 'C3 年轻度', 'uniqueness': 'C4 独特性',
            'harmony': 'C5 和谐度',
            'eye_score': '眼部分值', 'nose_score': '鼻部分值',
            'lip_score': '唇部分值', 'contour_score': '轮廓分值',
            'skin_texture': '肤质分值',
        }

        for i, (key, val) in enumerate(items):
            y_center = pad_t + bar_gap * i + bar_gap * 0.45
            bar_h = max(bar_gap * 0.55, _sp(8))
            label_text = labels_cn_map.get(key, str(key)[:12])
            canvas.create_text(
                pad_l - _sp(6), y_center,
                text=label_text, anchor="e",
                fill=TEXT_SECONDARY,
                font=("Microsoft YaHei", _fs(9)),
            )
            ratio = val / max_val if max_val > 0 else 0
            bar_w = max(ratio * bar_area_w, _sp(2))
            canvas.create_rectangle(
                pad_l, y_center - bar_h / 2,
                pad_l + bar_area_w, y_center + bar_h / 2,
                fill=RING_BG, outline="", width=0,
            )
            canvas.create_rectangle(
                pad_l, y_center - bar_h / 2,
                pad_l + bar_w, y_center + bar_h / 2,
                fill=ACCENT, outline="", width=0,
            )
            canvas.create_text(
                pad_l + bar_w + _sp(6), y_center,
                text=f"{float(val):.1f}", anchor="w",
                fill=TEXT_PRIMARY,
                font=("Consolas", _fs(10), "bold"),
            )

    def _on_canvas_click(self, event: Any) -> None:
        """Canvas 点击事件: 切换选中人脸"""
        if not self.face_rects or len(self.face_rects) <= 1:
            return

        x_off, y_off = self._img_offset
        scale = self._img_scale
        cx = (event.x - x_off) / scale
        cy = (event.y - y_off) / scale

        # 检查点击是否在某个人脸框内
        for i, (fx, fy, fw, fh) in enumerate(self.face_rects):
            if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
                if i != self.active_face_idx:
                    self.active_face_idx = i
                    self._update_face_display()
                return

    def _update_face_display(self):
        """更新当前选中人脸的显示"""
        if self.current_display:
            self._update_canvas_preview(self.current_display, self.face_rects)

        if self.face_results and self.active_face_idx < len(self.face_results):
            self._show_face_result(self.face_results[self.active_face_idx])

        # 更新导航
        n_faces = len(self.face_rects)
        if n_faces > 1:
            self.face_nav_label.configure(text=f"人脸 {self.active_face_idx + 1} / {n_faces}")
            self.face_nav_frame.pack(fill="x", padx=_sp(5), pady=(0, _sp(5)),
                                     after=self.btn_export.master if hasattr(self, 'btn_export') else None)
        else:
            self.face_nav_frame.pack_forget()

    def _prev_face(self):
        """上一张人脸"""
        if self.face_rects and self.active_face_idx > 0:
            self.active_face_idx -= 1
            self._update_face_display()

    def _next_face(self):
        """下一张人脸"""
        if self.face_rects and self.active_face_idx < len(self.face_rects) - 1:
            self.active_face_idx += 1
            self._update_face_display()

    def _clear_results(self):
        """清除所有结果显示"""
        # 文本控件
        rst_text_map = {
            self.score_text: "等待分析…\n\n💎 选择图片后点击「开始分析」",
            self.compare_text: "📊 分析后显示11种审美视角得分对比",
            self.advice_text: "💡 分析后显示偏好驱动的颜值提升建议…",
        }
        for text_widget, placeholder in rst_text_map.items():
            text_widget.configure(state="normal")
            text_widget.delete("1.0", "end")
            text_widget.insert("1.0", placeholder)
            text_widget.configure(state="disabled")

        # 清空矩阵 Canvas
        if self._matrix_canvas is not None:
            self._draw_matrix_placeholder()

        # 清空 Canvas 柱状图
        if self._feat_bars is not None:
            self._feat_bars.delete("all")
            cw = self._feat_bars.winfo_width()
            ch = self._feat_bars.winfo_height()
            if cw < 50:
                cw, ch = _sp(420), _sp(280)
            self._feat_bars.create_text(
                cw // 2, ch // 2,
                text="📐 分析后显示10维特征分布",
                fill=TEXT_MUTED,
                font=("Microsoft YaHei", _fs(11)),
            )

        self.btn_makeup.configure(state="disabled")
        self.btn_export.configure(state="disabled")
        self.face_nav_frame.pack_forget()

    # ═══════════════════════════════════════════
    #  分析核心
    # ═══════════════════════════════════════════
    def _start_analysis(self):
        """开始分析"""
        if self.current_image is None:
            messagebox.showwarning("提示", "请先选择一张图片")
            return

        self.is_analyzing = True
        self.stop_flag = False
        self.btn_analyze.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._set_status("🔍 分析中…", color=GOLD, status_text="分析中")

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _stop_analysis(self) -> None:
        """停止分析"""
        self.stop_flag = True
        self.is_analyzing = False
        self.after(0, lambda: self.btn_analyze.configure(state="normal"))
        self.after(0, lambda: self.btn_stop.configure(state="disabled"))

    def _run_analysis(self) -> None:
        """后台线程: 执行完整分析流程"""
        t0 = time.time()
        try:
            if self.current_image is None:
                self.after(0, self._on_error, "图片为空")
                return
            orig_h, orig_w = self.current_image.shape[:2]
            img = self.current_image.copy()
            quick = bool(self.check_quick.get())
            use_soft = bool(self.check_soft.get())

            # 缩放
            img = resize_for_analysis(img, quick_mode=quick)
            new_h, new_w = img.shape[:2]
            # 坐标还原比例（用于 Canvas 人脸框显示）
            scale_x = orig_w / new_w if new_w > 0 else 1.0
            scale_y = orig_h / new_h if new_h > 0 else 1.0
            if self.stop_flag:
                self.after(0, self._on_cancel, t0)
                return

            # 纹理预检
            precheck = texture_precheck(img)
            if self.stop_flag:
                self.after(0, self._on_cancel, t0)
                return

            # 人脸检测
            if use_soft:
                soft_result = detect_face_soft(img)
                face_rects = soft_result['candidates']
                face_probs = soft_result['probs']
            else:
                enhance_side = bool(self.check_side.get())
                enhance_large = bool(self.check_large.get())
                face_rects = detect_faces(img, enhance_side=enhance_side, enhance_large=enhance_large)
                face_probs = [1.0] * len(face_rects)

            if self.stop_flag:
                self.after(0, self._on_cancel, t0)
                return

            # 无面场景
            if not face_rects:
                scene = describe_non_face_image(img)
                self.after(0, self._show_no_face, img, precheck, scene, t0)
                return

            # 将检测框从缩放后坐标还原到原始图片坐标 (用于 Canvas 显示)
            if scale_x != 1.0 or scale_y != 1.0:
                display_rects = [(int(fx * scale_x), int(fy * scale_y),
                                  int(fw * scale_x), int(fh * scale_y))
                                 for (fx, fy, fw, fh) in face_rects]
            else:
                display_rects = face_rects

            # 分析每张脸
            face_results: list[dict[str, Any]] = []
            for f_idx, (fx, fy, fw, fh) in enumerate(face_rects):
                if self.stop_flag:
                    self.after(0, self._on_cancel, t0)
                    return

                face_crop = img[max(0, fy):min(fy + fh, img.shape[0]),
                                max(0, fx):min(fx + fw, img.shape[1])]
                if face_crop.size == 0:
                    continue

                # 肤色
                face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else face_crop
                skin_tone = classify_skin_tone(face_crop_bgr)

                # v49: 瑕疵检测
                blemish_result = detect_skin_blemishes(face_crop_bgr)

                # 特征提取 (只给当前脸)
                single_rect = [(fx, fy, fw, fh)]
                feats = extract_face_roi_features(img, single_rect, _remove_bg=bool(self.check_remove_bg.get()))
                feats.blemish_score = blemish_result['blemish_score']

                # 评分
                pref_raw = PREFERENCE_PRESETS.get(self.current_pref, [1, 1, 1, 1, 1])
                a_mat = features_to_matrix(feats, pref_raw=pref_raw)
                det_val = float(np.linalg.det(a_mat))
                quality = features_quality(feats, pref_raw=pref_raw)
                beauty = raw_to_beauty(det_val, quality=quality)

                skin_w = SKIN_CLARITY_WEIGHTS.get(self.current_pref, 1.0)
                tone_w = SKIN_TONE_WEIGHTS.get(self.current_pref, 0.0)
                skin_b = skin_clarity_bonus(feats.skin_clarity, pref_skin=skin_w)
                tone_b = skin_tone_affinity_bonus(feats.skin_tone_label, tone_w)
                blemish_p = blemish_penalty(feats.blemish_score)
                # v53.1: 几何美学加分 (此前遗漏)
                geo_dims = compute_geo_dimensions(img, single_rect)
                geo_b = geo_clarity_bonus(geo_dims, pref_weight=skin_w)
                total = round(min(beauty + skin_b + tone_b + geo_b - blemish_p, 10.0), 2)

                grade = get_grade(total)
                style = features_to_style(feats)
                all_scores = compute_all_preference_scores(feats)
                advice = generate_beauty_advice(feats, self.current_pref, top_n=3)
                # v48: 缺陷/增值判定
                defects = diagnose_defects(feats, total, self.current_pref)
                bonuses = diagnose_bonuses(feats, total, self.current_pref)
                # v48: Ridge校准分 (ML模型预测)
                calib_score = predict_calibrated_score(img, (fx, fy, fw, fh))

                face_results.append({
                    'feats': feats,
                    'det_val': round(det_val, 4),
                    'quality': quality,
                    'beauty': round(beauty, 2),
                    'skin_bonus': skin_b,
                    'tone_bonus': tone_b,
                    'geo_bonus': geo_b,                    # v53.1: 几何加分
                    'geo_dimensions': geo_dims,             # v53.1: 几何维度数据
                    'blemish_score': feats.blemish_score,
                    'blemish_details': blemish_result['details'],
                    'blemish_penalty': blemish_p,
                    'total': total,
                    'grade': grade,
                    'style': style,
                    'skin_tone': skin_tone,
                    'all_scores': all_scores,
                    'advice': advice,
                    'defects': defects,        # v48: 缺陷判定
                    'bonuses': bonuses,        # v48: 增值判定
                    'calib_score': calib_score,  # v48: Ridge校准分
                    'prob': round(face_probs[f_idx], 3) if f_idx < len(face_probs) else 1.0,
                })

                if f_idx == 0:
                    # 第一个结果立即显示
                    pass

            elapsed = round((time.time() - t0) * 1000)

            self.after(0, self._apply_results, display_rects, face_results, precheck, elapsed, face_probs)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.after(0, self._on_error, str(e))
        finally:
            self.is_analyzing = False
            self.after(0, lambda: self.btn_analyze.configure(state="normal"))
            self.after(0, lambda: self.btn_stop.configure(state="disabled"))

    def _on_cancel(self, t0: float) -> None:
        """分析被取消"""
        elapsed = round((time.time() - t0) * 1000)
        self._set_status(f"⏹ 分析已取消 ({elapsed}ms)", color=WARNING, status_text="空闲")
        self.is_analyzing = False
        self.btn_analyze.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _on_error(self, msg: str):
        """分析失败"""
        self._set_status(f"❌ 分析失败: {msg}", color=DANGER, status_text="错误")
        self.is_analyzing = False
        self.btn_analyze.configure(state="normal")
        self.btn_stop.configure(state="disabled")

    def _apply_results(self, face_rects: list[tuple[int, int, int, int]], face_results: list[dict[str, Any]], precheck: dict[str, Any], elapsed: float, _face_probs: list[float]) -> None:
        """主线程: 应用分析结果"""
        self.face_rects = face_rects
        self.face_results = face_results
        self.active_face_idx = 0

        # 更新纹理预检卡片
        self._update_precheck_card(precheck)

        # 更新 Canvas (绘制人脸框) — 必须用原图重建PIL, 确保与face_rects坐标系一致
        if self.current_image is not None:
            pil_img = Image.fromarray(self.current_image)
            self._update_canvas_preview(pil_img, face_rects)

        # 显示第一个结果
        if face_results:
            self._show_face_result(face_results[0])
            self.btn_makeup.configure(state="normal")
            self.btn_export.configure(state="normal")

        # 多人脸导航
        n = len(face_rects)
        if n > 1:
            self.face_nav_label.configure(text=f"人脸 1 / {n}")
            self.face_nav_frame.pack(fill="x", padx=_sp(5), pady=(0, _sp(5)))
        else:
            self.face_nav_frame.pack_forget()

        self._set_status(
            f"✅ 分析完成 — 检测到 {n} 张人脸 ({elapsed}ms)",
            color=SUCCESS if face_results else WARNING,
            status_text=f"{n}人脸" if face_results else "无脸",
        )

    def _update_precheck_card(self, precheck: dict[str, Any]) -> None:
        """更新纹理预检状态卡片"""
        quality = precheck.get('quality_label', 'unknown')
        q_text = precheck.get('quality_text', '')
        lap_var = precheck.get('laplacian_var', 0)
        noise = precheck.get('noise_level', 0)
        illum = precheck.get('illumination_uniformity', 1)
        res = precheck.get('resolution', '?')
        mp = precheck.get('megapixels', 0)

        # 颜色映射
        color_map = {
            'good': (SUCCESS, '✓'),
            'warn': (WARNING, '⚠'),
            'bad':  (DANGER, '✗'),
        }
        color, icon = color_map.get(quality, (TEXT_SECONDARY, ''))

        self.precheck_label.configure(
            text=f"纹理预检: {icon} {q_text}",
            text_color=color,
        )
        self.precheck_detail.configure(
            text=f"清晰度:{lap_var:.0f} | 噪点:{noise:.1f} | 光照:{illum:.0%} | {res} ({mp}MP)",
            text_color=TEXT_SECONDARY,
        )

    def _show_no_face(self, _img: np.ndarray, precheck: dict[str, Any], scene: SceneDescription, t0: float) -> None:
        """显示无脸结果"""
        elapsed = round((time.time() - t0) * 1000)
        self.face_rects = []
        self.face_results = []
        self.face_nav_frame.pack_forget()
        self.btn_makeup.configure(state="disabled")
        self.btn_export.configure(state="disabled")

        # 更新纹理预检卡片
        self._update_precheck_card(precheck)

        # 评分Tab
        self.score_text.configure(state="normal")
        self.score_text.delete("1.0", "end")
        lines = [
            "",
            "  😔 未检测到人脸",
            "",
            f"  🔬 纹理质量: {precheck.get('quality_label', '未知')}",
            f"  🏷 场景分类: {scene.category}",
            f"  📝 描述: {scene.natural_text}",
            f"  ⏱ 耗时: {elapsed}ms",
        ]
        self.score_text.insert("1.0", "\n".join(lines))
        self.score_text.configure(state="disabled")

        # 清空其他Tab
        for tw in [self.compare_text, self.advice_text]:
            tw.configure(state="normal")
            tw.delete("1.0", "end")
            tw.insert("1.0", "(非人脸图片 — 无法分析)")
            tw.configure(state="disabled")

        # 清空矩阵 Canvas
        if self._matrix_canvas is not None:
            self._draw_matrix_placeholder("(非人脸图片 — 无法分析)")

        # 清空 Canvas 柱状图
        if self._feat_bars is not None:
            self._feat_bars.delete("all")
            cw = self._feat_bars.winfo_width()
            ch = self._feat_bars.winfo_height()
            if cw < 50:
                cw, ch = _sp(420), _sp(280)
            self._feat_bars.create_text(
                cw // 2, ch // 2,
                text="(非人脸图片 — 无特征数据)",
                fill=TEXT_MUTED,
                font=("Microsoft YaHei", _fs(11)),
            )

        self._set_status(f"😔 分析完成 — 未检测到人脸 ({elapsed}ms)", color=WARNING, status_text="无脸")

    def _show_face_result(self, result: dict[str, Any]) -> None:
        """显示单张人脸的分析结果"""
        r = result
        total = r['total']
        grade = r['grade']

        # ── 评分Tab: 环形仪表 + 详细信息 ──
        self.score_text.configure(state="normal")
        self.score_text.delete("1.0", "end")

        color = self._score_color(total)

        # ASCII 环形仪表
        gauge_w = 28
        filled = max(1, int(total / 10 * gauge_w))
        gauge_lines = [
            "",
            "    ┌" + "─" * (gauge_w + 2) + "┐",
            "    │ " + "█" * filled + "░" * (gauge_w - filled) + " │",
            "    └" + "─" * (gauge_w + 2) + "┘",
            "",
        ]
        for gl in gauge_lines:
            self.score_text.insert("end", gl + "\n")

        # 大号评分
        score_idx = self.score_text.index("end-1c")
        self.score_text.insert("end", f"         {total:.1f}\n", "")
        self.score_text.tag_add("big_score", score_idx, "end-1c")
        # CTkTextbox 禁止 tag_config 传 font, 走底层 tkinter Text
        self.score_text._textbox.tag_config("big_score", foreground=color,
                                             font=("Consolas", _fs(24), "bold"))

        grade_idx = self.score_text.index("end-1c")
        self.score_text.insert("end", f"       {grade.get('label', '?')}\n", "")
        self.score_text.tag_add("grade_tag", grade_idx, "end-1c")
        self.score_text._textbox.tag_config("grade_tag", foreground=color,
                                             font=("Microsoft YaHei", _fs(14), "bold"))

        self.score_text.insert("end", "\n")

        # 详细信息
        gender_label = {'male': '男性', 'female': '女性', 'unknown': '未识别'}.get(r['style'].get('gender', 'unknown'), '未识别')
        gender_conf = getattr(r.get('feats', None), 'gender_confidence', 0) or 0
        detail_parts = [
            ("👤", f"性别: {gender_label}"),
            ("🎨", f"风格: {r['style']['primary_style']}"),
            ("",  f"纯度: {r['style']['purity']}"),
            ("🎭", f"肤色: {r['skin_tone']}"),
            ("🎯", f"性别置信: {gender_conf:.0%}" if gender_conf > 0 else f"置信: {r.get('prob', 1.0):.1%}"),
        ]
        for emoji, text in detail_parts:
            prefix = f"  {emoji} " if emoji else "     "
            self.score_text.insert("end", f"{prefix}{text}\n")

        self.score_text.insert("end", "\n  ── 📊 评分分解 ──\n")
        decomps = [
            ("det(A)矩阵秩", f"{r['det_val']:.4f}"),
            ("基础美学分", f"{r['beauty']:.2f}"),
            ("特征质量", f"{r['quality']}"),
            ("肤质加分", f"+{r['skin_bonus']:.2f}"),
            ("肤色加分", f"+{r['tone_bonus']:.2f}"),
            ("几何加分", f"+{r.get('geo_bonus', 0):.2f}"),          # v53.1
            ("瑕疵扣分", f"-{r['blemish_penalty']:.2f}"),
            ("最终总分", f"{r['total']:.2f} / 10"),
        ]
        for label, val in decomps:
            self.score_text.insert("end", f"    {label:<14s} {val}\n")

        # v48: Ridge校准分
        if r.get('calib_score') is not None:
            cs = r['calib_score']
            cs_color = self._score_color(cs)
            self.score_text.insert("end", f"\n  [ML校准] {cs:.1f} / 9", "")
            cs_idx = self.score_text.index("end-1c")
            self.score_text._textbox.tag_config("calib_tag", foreground=cs_color,
                                                font=("Microsoft YaHei", _fs(11), "bold"))
            self.score_text.tag_add("calib_tag", cs_idx, "end-1c")

        # v48: 缺陷判定 (score < 2)
        defects = r.get('defects', [])
        if defects:
            self.score_text.insert("end", "\n\n  ⚠ 缺陷诊断 (score<2):\n", "")
            dheader_idx = self.score_text.index("end-2c")
            self.score_text._textbox.tag_config("defect_header", foreground=DANGER,
                                                font=("Microsoft YaHei", _fs(11), "bold"))
            self.score_text.tag_add("defect_header", dheader_idx, "end-1c")
            for d in defects:
                self.score_text.insert("end",
                    f"    [{d['severity']}] {d['label']}\n"
                    f"      {d['detail']}\n"
                    f"      → {d['advice']}\n")

        # v48: 增值判定 (score > 9)
        bonuses = r.get('bonuses', [])
        if bonuses:
            self.score_text.insert("end", "\n\n  🌟 卓越特质 (score>9):\n", "")
            bheader_idx = self.score_text.index("end-2c")
            self.score_text._textbox.tag_config("bonus_header", foreground=SUCCESS,
                                                font=("Microsoft YaHei", _fs(11), "bold"))
            self.score_text.tag_add("bonus_header", bheader_idx, "end-1c")
            for b in bonuses:
                self.score_text.insert("end",
                    f"    ✦ {b['label']}\n"
                    f"      {b['detail']}\n")

        self.score_text.configure(state="disabled")

        # ── 维度Tab: Canvas 柱状图 ──
        try:
            if hasattr(r['feats'], 'as_dict'):
                self._draw_feature_bars(r['feats'])
            else:
                # fallback: 可能是dict或其他格式
                self._draw_feature_bars_fallback(r['feats'])
        except Exception as e:
            if self._feat_bars is not None:
                self._feat_bars.delete("all")
                cw = max(self._feat_bars.winfo_width(), _sp(420))
                ch = max(self._feat_bars.winfo_height(), _sp(300))
                self._feat_bars.create_text(
                    cw // 2, ch // 2,
                    text=f"维度图表渲染失败\n{type(r['feats']).__name__}",
                    fill=TEXT_MUTED, font=("Microsoft YaHei", _fs(11)),
                )

        # ── 矩阵Tab ──
        md = matrix_for_display(r['feats'], pref_raw=PREFERENCE_PRESETS.get(self.current_pref))
        self._draw_matrix_canvas(md)

        # 对比Tab
        self.compare_text.configure(state="normal")
        self.compare_text.delete("1.0", "end")

        # 获取分数列表计算相对比例
        all_s = r.get('all_scores', [])
        scores_list = [s['score'] for s in all_s]
        s_min, s_max = (min(scores_list), max(scores_list)) if scores_list else (0, 10)
        s_range = max(s_max - s_min, 0.3)  # 最小动态范围避免除零
        bar_w = 35  # 条形图总宽度

        # 颜色tag
        for tg, fg in [("cmp_hot", GOLD), ("cmp_norm", SUCCESS), ("cmp_cool", ACCENT2)]:
            self.compare_text._textbox.tag_config(tg, foreground=fg)

        self.compare_text.insert("end", "  📊 11种审美视角得分对比\n\n")
        self.compare_text.insert("end", f"  {'排名':<4s} {'审美偏好':<14s} {'得分':<8s} {'条形图'}".ljust(55) + "    差距\n")
        self.compare_text.insert("end", "  " + "─" * 65 + "\n")

        for i, s in enumerate(all_s, 1):
            marker = ' ◀' if s['pref_name'] == self.current_pref else ''
            # 相对比例: 最低分=1格，最高分=满格
            rel = (s['score'] - s_min) / s_range
            bar_len = max(1, int(rel * bar_w))
            bar = "█" * bar_len + "░" * (bar_w - bar_len)
            gap = s['score'] - s_min

            line = f"  {i:<4d} {s['pref_name']:<14s} {s['score']:<8.2f} {bar}{marker}"
            self.compare_text.insert("end", line)
            p0 = self.compare_text.index("end-1c")
            diff_str = f" +{gap:.2f}"
            self.compare_text.insert("end", diff_str)
            p1 = self.compare_text.index("end-1c")
            # 当前偏好暖色强调
            tag = "cmp_hot" if s['pref_name'] == self.current_pref else (
                "cmp_norm" if gap < 1.0 else "cmp_cool")
            self.compare_text.tag_add(tag, p0, p1)
            self.compare_text.insert("end", "\n")

        self.compare_text.insert("end", "  " + "─" * 65 + "\n")
        self.compare_text.insert("end", f"  📏 分值范围 [{s_min:.2f} ~ {s_max:.2f}]  条形图按相对比例缩放\n")
        self.compare_text.configure(state="disabled")

        # 建议Tab
        self.advice_text.configure(state="normal")
        self.advice_text.delete("1.0", "end")
        advice_lines = [f"  💡 「{self.current_pref}」审美提升建议\n", ""]
        for i, a in enumerate(r.get('advice', []), 1):
            advice_lines.append(f"  {i}. {a['text']}")
        if not r.get('advice'):
            advice_lines.append("  (当前各维度表现均衡，暂无明显短板)")

        # v49: 瑕疵检测反馈
        blemish_details = r.get('blemish_details', '')
        blemish_score = r.get('blemish_score', 0)
        blemish_penalty = r.get('blemish_penalty', 0)
        if blemish_details and blemish_score > 0:
            advice_lines.append("")
            advice_lines.append(f"  🔍 肤质检测: {blemish_details}")
            if blemish_penalty > 0:
                advice_lines.append(f"      瑕疵减分: -{blemish_penalty:.2f}")
            advice_lines.append(f"      瑕疵评分: {blemish_score:.1f}/10")

        self.advice_text.insert("1.0", "\n".join(advice_lines))
        self.advice_text.configure(state="disabled")

    # ═══════════════════════════════════════════
    #  偏好
    # ═══════════════════════════════════════════
    def _select_pref(self, name: str):
        """选择审美偏好"""
        self.current_pref = name
        for n, btn in self.pref_buttons.items():
            btn.configure(fg_color=ACCENT if n == name else ACCENT2)
        self._set_status(f"🎨 审美偏好: {name}")

        # 更新自定义滑块
        weights = PREFERENCE_PRESETS.get(name, [1, 1, 1, 1, 1])
        label_keys = list(self.weight_sliders.keys())
        for i, key in enumerate(label_keys):
            if i < len(weights):
                self.weight_sliders[key].set(weights[i])

    def _apply_custom_weights(self) -> None:
        """应用自定义滑块权重"""
        w_list = [self.weight_sliders[k].get() for k in self.weight_sliders]
        name = f"自定义"
        self.current_pref = name
        self._set_status(f"🎨 审美偏好: {name} [{', '.join(f'{w:.1f}' for w in w_list)}]")

    def _save_custom_pref(self):
        """保存当前自定义偏好"""
        w_list = [self.weight_sliders[k].get() for k in self.weight_sliders]
        name = f"自定义-{len(self.custom_pref_store) + 1}"
        self.custom_pref_store[name] = w_list
        self._write_custom_pref_store()
        self._refresh_saved_pref_display()
        self._set_status(f"💾 偏好已保存: {name}")

    def _restore_custom_pref(self):
        """恢复上一次保存的自定义偏好"""
        if not self.custom_pref_store:
            messagebox.showinfo("提示", "没有已保存的自定义偏好")
            return
        # 取最新的
        names = list(self.custom_pref_store.keys())
        latest = names[-1]
        weights = self.custom_pref_store[latest]
        label_keys = list(self.weight_sliders.keys())
        for i, key in enumerate(label_keys):
            if i < len(weights):
                self.weight_sliders[key].set(weights[i])
        self.current_pref = latest
        self._set_status(f"↩ 已恢复: {latest}")

    def _load_custom_pref_store(self):
        """从文件加载自定义偏好"""
        try:
            if os.path.exists(_CUSTOM_PREF_PATH):
                with open(_CUSTOM_PREF_PATH, 'r', encoding='utf-8') as f:
                    self.custom_pref_store = json.load(f)
        except Exception:
            self.custom_pref_store = {}

    def _write_custom_pref_store(self) -> None:
        """保存自定义偏好到文件"""
        try:
            with open(_CUSTOM_PREF_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.custom_pref_store, f, indent=2, ensure_ascii=False)
        except Exception:
            import traceback
            traceback.print_exc()

    def _refresh_saved_pref_display(self):
        """刷新已保存偏好列表"""
        for w in self.saved_pref_frame.winfo_children():
            w.destroy()

        if not self.custom_pref_store:
            ctk.CTkLabel(
                self.saved_pref_frame,
                text="(暂无保存的偏好)",
                font=self._font("Microsoft YaHei", 10),
                text_color=TEXT_SECONDARY,
            ).pack(pady=_sp(5))
            return

        for name, weights in self.custom_pref_store.items():
            row = ctk.CTkFrame(self.saved_pref_frame, fg_color=BG_DARK, corner_radius=4)
            row.pack(fill="x", pady=_sp(1))

            ctk.CTkLabel(
                row, text=name,
                font=self._font("Microsoft YaHei", 10),
                text_color=TEXT_PRIMARY,
            ).pack(side="left", padx=_sp(6), pady=_sp(2))

            ctk.CTkLabel(
                row,
                text=", ".join(f"{w:.1f}" for w in weights),
                font=self._font("Consolas", 9),
                text_color=TEXT_SECONDARY,
            ).pack(side="left", padx=_sp(6))

            ctk.CTkButton(
                row, text="加载",
                command=lambda n=name, w=weights: self._load_saved_pref(n, w),
                width=_sp(50),
                font=self._font("Microsoft YaHei", 9),
                fg_color=ACCENT2,
            ).pack(side="right", padx=_sp(4), pady=_sp(2))

    def _load_saved_pref(self, name: str, weights: list[float]) -> None:
        """加载已保存的自定义偏好"""
        label_keys = list(self.weight_sliders.keys())
        for i, key in enumerate(label_keys):
            if i < len(weights):
                self.weight_sliders[key].set(weights[i])
        self.current_pref = name
        self._set_status(f"📂 已加载偏好: {name}")

    # ═══════════════════════════════════════════
    #  化妆模拟
    # ═══════════════════════════════════════════
    def _simulate_makeup(self) -> None:
        if not self.face_results:
            return
        r = self.face_results[self.active_face_idx]

        popup = ctk.CTkToplevel(self)
        popup.title("化妆模拟")
        popup.geometry(f"{_sp(400)}x{_sp(300)}")
        popup.configure(fg_color=BG_CARD)

        ctk.CTkLabel(
            popup,
            text=f"当前评分: {r['total']} / 10",
            font=self._font("Microsoft YaHei", 14, "bold"),
            text_color=ACCENT,
        ).pack(pady=_sp(10))

        ctk.CTkLabel(
            popup,
            text="化妆强度 δ (0=素颜, 2=浓妆大片):",
            font=self._font("Microsoft YaHei", 11),
            text_color=TEXT_PRIMARY,
        ).pack()

        delta_var = ctk.DoubleVar(value=1.0)
        delta_slider = ctk.CTkSlider(
            popup, from_=0, to=2, variable=delta_var, number_of_steps=20,
        )
        delta_slider.pack(fill="x", padx=_sp(30), pady=_sp(10))

        result_label = ctk.CTkLabel(
            popup, text="", font=self._font("Microsoft YaHei", 13),
            text_color=GOLD,
        )
        result_label.pack(pady=_sp(5))

        def update_sim(val=None):
            d = delta_var.get()
            sim = simulate_makeup(r['total'], delta=d)
            result_label.configure(
                text=f"δ={d:.1f}  →  模拟分: {sim['simulated_score']:.2f}  ({sim['effect_label']})"
            )

        delta_slider.configure(command=update_sim)
        update_sim()

        ctk.CTkButton(
            popup, text="关闭", command=popup.destroy,
            font=self._font("Microsoft YaHei", 11),
        ).pack(pady=_sp(10))

    # ═══════════════════════════════════════════
    #  导出报告
    # ═══════════════════════════════════════════
    def _export_report(self) -> None:
        """导出当前分析报告"""
        if not self.face_results:
            return

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="导出报告",
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("JSON", "*.json")],
            initialfile=f"颜值分析报告_{ts}.txt",
        )
        if not path:
            return

        try:
            is_json = path.endswith('.json')
            if is_json:
                data: dict[str, Any] = {
                    'filename': os.path.basename(self.current_image_path or ''),
                    'preference': self.current_pref,
                    'face_count': len(self.face_results),
                    'faces': [],
                }
                for i, r in enumerate(self.face_results):
                    gender_export = r['style'].get('gender', 'unknown')
                    data['faces'].append({
                        'index': i + 1,
                        'score': r['total'],
                        'grade': r['grade']['label'],
                        'gender': gender_export,                        # v53
                        'style': r['style']['primary_style'],
                        'skin_tone': r['skin_tone'],
                        'det_val': r['det_val'],
                        'quality': r['quality'],
                        'skin_bonus': r['skin_bonus'],
                        'tone_bonus': r['tone_bonus'],
                        'calib_score': r.get('calib_score'),        # v48
                        'defects': r.get('defects', []),             # v48
                        'bonuses': r.get('bonuses', []),             # v48
                    })
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                lines = [
                    "颜值矩阵分析系统 v53.1 — 分析报告",
                    "=" * 40,
                    f"文件: {os.path.basename(self.current_image_path or '')}",
                    f"偏好: {self.current_pref}",
                    f"人脸数: {len(self.face_results)}",
                    "",
                ]
                for i, r in enumerate(self.face_results):
                    gender_label = {'male': '男性', 'female': '女性', 'unknown': '未识别'}.get(r['style'].get('gender', 'unknown'), '未识别')
                    lines.append(f"--- 人脸 #{i + 1} ---")
                    lines.append(f"  评分: {r['total']} / 10  ({r['grade']['label']})")
                    lines.append(f"  性别: {gender_label}")
                    if r.get('calib_score') is not None:
                        lines.append(f"  ML校准分: {r['calib_score']:.1f} / 9")
                    lines.append(f"  风格: {r['style']['primary_style']}")
                    lines.append(f"  肤色: {r['skin_tone']}")
                    lines.append(f"  基础: {r['beauty']}  + 肤质: {r['skin_bonus']}  + 肤色: {r['tone_bonus']}")
                    # v48: 缺陷/增值
                    for d in r.get('defects', []):
                        lines.append(f"  ⚠ [{d['severity']}] {d['label']}: {d['advice']}")
                    for b_ in r.get('bonuses', []):
                        lines.append(f"  🌟 {b_['label']}: {b_['detail']}")
                    lines.append("")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

            self._set_status(f"📥 报告已导出: {os.path.basename(path)}", color=SUCCESS, status_text="完成")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))

    # ═══════════════════════════════════════════
    #  问卷提交
    # ═══════════════════════════════════════════
    def _submit_questionnaire(self):
        """提交问卷结果"""
        answers = []
        for var in self.q_answers:
            val = var.get()
            if not val:
                messagebox.showwarning("提示", "请回答所有问题")
                return
            answers.append(ord(val) - ord('A'))

        user_vec = compute_user_vector(answers)
        match = match_best_preset(user_vec)

        self.q_result_label.configure(
            text=f"你的审美DNA: {match['matched_preset']}\n"
                 f"匹配度: {match['all_matches'][0]['similarity']:.1%}\n"
                 f"已自动应用此偏好！"
        )
        self._select_pref(match['matched_preset'])

        # 切换到分析标签
        self.tabview.set(" 分析 ")

    # ═══════════════════════════════════════════
    #  批量分析
    # ═══════════════════════════════════════════
    def _batch_select_folder(self):
        """选择批量分析文件夹"""
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if folder:
            self.batch_folder = folder
            self.batch_folder_label.configure(text=folder)
            self.btn_batch_start.configure(state="normal")
            # 预览文件数
            exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
            files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
            self.batch_result_text.configure(state="normal")
            self.batch_result_text.delete("1.0", "end")
            self.batch_result_text.insert("1.0", f'文件夹: {folder}\n找到 {len(files)} 个图片文件\n\n就绪，点击「开始批量分析」')
            self.batch_result_text.configure(state="disabled")
            self.batch_progress.set(0)
            self.batch_progress_label.configure(text=f"共 {len(files)} 个文件")

    def _batch_start_analysis(self) -> None:
        """启动批量分析"""
        if not self.batch_folder:
            return

        self.btn_batch_start.configure(state="disabled")
        self.btn_batch_stop.configure(state="normal")
        self.batch_cancel.clear()
        self.batch_results = []

        thread = threading.Thread(target=self._batch_run_analysis, daemon=True)
        thread.start()

    def _batch_run_analysis(self) -> None:
        """后台: 批量分析"""
        pref_name = self.current_pref
        folder = self.batch_folder
        if not folder:
            return  # safety: no folder selected

        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        files = sorted([
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ])
        total = len(files)

        for idx, fname in enumerate(files):
            if self.batch_cancel.is_set():
                break

            filepath = os.path.join(folder, fname)
            try:
                img = load_and_normalize_image(filepath, source_type='path')
                face_rects = detect_face(img)

                if face_rects:
                    face_crop_bgr = cv2.cvtColor(
                        img[face_rects[0][1]:face_rects[0][1] + face_rects[0][3],
                            face_rects[0][0]:face_rects[0][0] + face_rects[0][2]],
                        cv2.COLOR_RGB2BGR,
                    ) if img.shape[2] == 3 else img
                    skin_tone = classify_skin_tone(face_crop_bgr)
                    blemish_result = detect_skin_blemishes(face_crop_bgr)

                    feats = extract_face_roi_features(img, face_rects, _remove_bg=True)
                    feats.blemish_score = blemish_result['blemish_score']
                    pref_raw = PREFERENCE_PRESETS.get(pref_name, [1, 1, 1, 1, 1])
                    a_mat = features_to_matrix(feats, pref_raw=pref_raw)
                    det_val = float(np.linalg.det(a_mat))
                    quality = features_quality(feats, pref_raw=pref_raw)
                    beauty = raw_to_beauty(det_val, quality=quality)
                    skin_w = SKIN_CLARITY_WEIGHTS.get(pref_name, 1.0)
                    tone_w = SKIN_TONE_WEIGHTS.get(pref_name, 0.0)
                    # v53.5: 批量分析补充geo_bonus (MediaPipe 关键点解剖学面宽比)
                    from image_utils import extract_face_landmarks as _extract_lm2
                    landmarks_img = _extract_lm2(img, face_rects[0])
                    geo_dims = compute_geo_dimensions(img, face_rects, landmarks=landmarks_img)
                    geo_b = geo_clarity_bonus(geo_dims, pref_weight=skin_w)
                    blemish_p = blemish_penalty(feats.blemish_score)
                    total = round(min(beauty + skin_clarity_bonus(feats.skin_clarity, skin_w) +
                                     skin_tone_affinity_bonus(feats.skin_tone_label, tone_w) +
                                     geo_b - blemish_p, 10.0), 2)
                    grade = get_grade(total)
                    defects = diagnose_defects(feats, total, pref_name)
                    bonuses = diagnose_bonuses(feats, total, pref_name)

                    self.batch_results.append({
                        'filename': fname,
                        'has_face': True,
                        'score': total,
                        'grade': grade['label'],
                        'skin_tone': skin_tone,
                        'defects': defects,    # v48
                        'bonuses': bonuses,    # v48
                    })
                else:
                    self.batch_results.append({
                        'filename': fname,
                        'has_face': False,
                        'score': 0,
                        'grade': '无',
                        'skin_tone': '未知',
                    })

            except Exception:
                self.batch_results.append({
                    'filename': fname,
                    'has_face': False,
                    'score': -1,
                    'grade': '错误',
                    'skin_tone': '未知',
                })

            # 更新进度
            progress = (idx + 1) / total
            _total: int = int(total)  # capture as int for lambda
            self.after(0, lambda p=progress, i=idx, t=_total, f=fname: self._batch_update_progress(p, i, t, f))

        self.after(0, self._batch_finish)

    def _batch_update_progress(self, progress: float, idx: int, total: int, fname: str) -> None:
        """主线程: 更新批量进度"""
        self.batch_progress.set(progress)
        self.batch_progress_label.configure(text=f"{idx + 1} / {total}: {fname}")

    def _batch_finish(self) -> None:
        """批量分析完成"""
        self.btn_batch_start.configure(state="normal")
        self.btn_batch_stop.configure(state="disabled")
        self.btn_batch_export.configure(state="normal")
        self.batch_progress.set(1.0)

        # 统计
        faces = [r for r in self.batch_results if r['has_face']]
        scores = [r['score'] for r in faces]

        lines = [f"批量分析完成 — {len(self.batch_results)} 个文件\n"]
        lines.append(f"有人脸: {len(faces)} | 无人脸: {len(self.batch_results) - len(faces)}")
        if scores:
            lines.append(f"最高分: {max(scores):.2f} | 最低分: {min(scores):.2f} | 平均分: {sum(scores)/len(scores):.2f}")
        lines.append(f"\n{'文件名':<30s} {'得分':<8s} {'等级':<10s} {'肤色':<8s}")
        lines.append("-" * 60)

        for r in sorted(self.batch_results, key=lambda x: x['score'], reverse=True):
            lines.append(
                f"{r['filename'][:28]:<30s} {r['score']:<8.2f} {r['grade']:<10s} {r['skin_tone']:<8s}"
            )

        self.batch_result_text.configure(state="normal")
        self.batch_result_text.delete("1.0", "end")
        self.batch_result_text.insert("1.0", "\n".join(lines))
        self.batch_result_text.configure(state="disabled")

        self._set_status(f"✅ 批量分析完成 — {len(faces)} 张人脸")

    def _batch_stop(self):
        """停止批量分析"""
        self.batch_cancel.set()
        self.btn_batch_stop.configure(state="disabled")
        self._set_status("⏹ 批量分析已停止", color=WARNING, status_text="空闲")

    def _batch_export_report(self) -> None:
        """导出批量分析报告"""
        if not self.batch_results:
            return

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="导出批量报告",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("JSON", "*.json"), ("文本", "*.txt")],
            initialfile=f"批量分析报告_{ts}.csv",
        )
        if not path:
            return

        try:
            if path.endswith('.json'):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(self.batch_results, f, indent=2, ensure_ascii=False)
            elif path.endswith('.csv'):
                import csv
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['filename', 'has_face', 'score', 'grade', 'skin_tone'])
                    writer.writeheader()
                    writer.writerows(self.batch_results)
            else:
                lines = ["文件名,有人脸,得分,等级,肤色"]
                for r in self.batch_results:
                    lines.append(f"{r['filename']},{r['has_face']},{r['score']},{r['grade']},{r['skin_tone']}")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines))

            self._set_status(f"📥 批量报告已导出: {os.path.basename(path)}", color=SUCCESS, status_text="完成")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))


# ═══════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    try:
        app = BeautyGUI()
        app.mainloop()
    except Exception as e:
        import traceback, time
        tb = traceback.format_exc()
        # 写崩溃日志
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_path = os.path.join(log_dir, 'crash_log.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Crash time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(tb)
        # 弹窗显示错误
        try:
            from tkinter import messagebox
            messagebox.showerror("程序崩溃", f"错误详情已写入:\n{log_path}\n\n{tb[:500]}")
        except Exception:
            import traceback
            traceback.print_exc()
        # 打印到 stderr
        print(tb, file=sys.stderr)
        input("按任意键退出...")
        sys.exit(1)
