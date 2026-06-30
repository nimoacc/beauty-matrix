"""
预处理: 对 SCUT-FBP5500 批量检测人脸 → 裁剪 → 保存到临时目录
这样训练时不需要每 epoch 重复人脸检测，大幅提速
"""
import os, sys, io, time
import numpy as np
import cv2

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'SCUT-FBP5500_v2', 'SCUT-FBP5500_v2', 'Images')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'stats_output', 'gender_cnn_cache')

CASCADE_PATH = os.path.join(PROJECT_ROOT, 'cascades', 'haarcascade_frontalface_default.xml')

cascade = None
for p in [CASCADE_PATH, cv2.data.haarcascades + 'haarcascade_frontalface_default.xml']:
    if os.path.exists(p):
        cascade = cv2.CascadeClassifier(p)
        break

os.makedirs(CACHE_DIR, exist_ok=True)

all_images = sorted([f for f in os.listdir(IMAGES_DIR)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f'共 {len(all_images)} 张图片')

failed = noface = 0
t0 = time.time()
for idx, fname in enumerate(all_images):
    src = os.path.join(IMAGES_DIR, fname)
    dst = os.path.join(CACHE_DIR, fname)

    if os.path.exists(dst):
        continue  # 已处理

    img = cv2.imread(src)
    if img is None:
        failed += 1
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) == 0:
        noface += 1
        # 无人脸 → 直接用原图
        cv2.imwrite(dst, img)
        continue

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    H, W = img.shape[:2]
    mx = int(w * 0.3)
    my = int(h * 0.3)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)

    face_roi = img[y1:y2, x1:x2]
    cv2.imwrite(dst, face_roi)

    if (idx + 1) % 1000 == 0:
        elapsed = time.time() - t0
        print(f'  {idx+1}/{len(all_images)} ({elapsed:.0f}s)')

elapsed = time.time() - t0
print(f'完成! 耗时 {elapsed:.0f}s, 失败 {failed}, 无人脸 {noface}')
print(f'缓存目录: {CACHE_DIR}')
