# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['beauty_gui_desktop.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('beauty_core.py', '.'),
        ('image_utils.py', '.'),
        ('gender_inference.py', '.'),
        ('preference_questionnaire.py', '.'),
        ('beauty_system/config.yaml', 'beauty_system'),
        ('stats_output/beauty_model_full.pkl', 'stats_output'),
        # v53: ML性别分类器模型
        ('stats_output/gender_model_v1.pkl', 'stats_output'),
        # v53.6: 肤色CNN模型
        ('stats_output/skin_tone_cnn_v3.onnx', 'stats_output'),
        # v52.1: 级联文件内置打包
        ('cascades', 'cascades'),
    ],
    hiddenimports=[
        'customtkinter',
        'numpy',
        'PIL',
        'cv2',
        'beauty_core',
        'image_utils',
        'gender_inference',
        'preference_questionnaire',
        'sklearn',
        'sklearn.linear_model',
        'sklearn.preprocessing',
        'pickle',
        'onnxruntime',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='颜值矩阵分析系统 v53.6',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
