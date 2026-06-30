"""
校准数据下载脚本
─────────────────────────
1. FFHQ 缩略图 (128x128, ~2GB) → FFHQ/ 目录, 用作洁净脸基线
2. ISIC Archive 色斑/病变图像 → ISIC/ 目录, 用作瑕疵特征参考
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import zipfile
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
FFHQ_DIR = os.path.join(BASE, 'FFHQ')
ISIC_DIR = os.path.join(BASE, 'ISIC')
os.makedirs(FFHQ_DIR, exist_ok=True)
os.makedirs(ISIC_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════
# 1. FFHQ 下载 (Google Drive → gdown)
# ═══════════════════════════════════════════════════
def download_ffhq_subset(target_count: int = 1000):
    """下载FFHQ子集用于洁净脸基线校准
    
    FFHQ 数据来自 NVIDIA, 使用 gdown 从 Google Drive 下载.
    完整 thumbnails128x128.zip 约 2GB (70k张128x128缩略图)
    也可以用 --images 分片下载 (每片约 1GB)
    
    策略: 先下载 metadata JSON, 然后拉取前N张缩略图
    """
    print("=" * 60)
    print("  下载 FFHQ 数据集 (洁净人脸基线)")
    print("=" * 60)
    
    # 先安装 gdown
    print("\n[准备] 安装 gdown...")
    os.system(f"{sys.executable} -m pip install gdown -q")
    
    import gdown
    
    # FFHQ Google Drive 文件列表 (来自 NVlabs/ffhq-dataset)
    # 缩略图打包 zip
    thumbs_zip_id = "1tg-Ur7d4vkQxwW2qRz2JqE4M1Zs3NHkv"  # thumbnails128x128.zip
    thumbs_zip_path = os.path.join(FFHQ_DIR, 'thumbnails128x128.zip')
    
    # 元数据 JSON
    json_id = "1ibLjLvLi1rZHKl4DqW3qPvFNQ-lWprRi"  # ffhq-dataset-v1.json
    json_path = os.path.join(FFHQ_DIR, 'ffhq-dataset-v1.json')
    
    # ── 下载元数据 (254MB - 包含70k张图片的meta信息) ──
    if not os.path.exists(json_path):
        print(f"\n[下载] 元数据 JSON (254MB)...")
        gdown.download(f"https://drive.google.com/uc?id={json_id}", json_path, quiet=False)
        print("  [OK] 元数据下载完成")
    else:
        print(f"\n[跳过] 元数据已存在: {json_path}")
    
    # ── 下载缩略图 ──
    if not os.path.exists(thumbs_zip_path):
        print(f"\n[下载] 缩略图包 (约 2GB, 70k张128x128)...")
        gdown.download(f"https://drive.google.com/uc?id={thumbs_zip_id}", thumbs_zip_path, quiet=False)
        print("  [OK] 缩略图下载完成")
        
        # 解压
        print("[解压] 正在解压缩略图...")
        extract_dir = os.path.join(FFHQ_DIR, 'thumbnails128x128')
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(thumbs_zip_path, 'r') as zf:
            members = zf.namelist()
            total = len(members)
            for i, member in enumerate(members):
                zf.extract(member, extract_dir)
                if i % 5000 == 0:
                    print(f"\r  解压进度: {i}/{total} ({i*100//total}%)", end='', flush=True)
        print(f"\r  解压完成: {total} 个文件")
        
        # 清理zip
        os.remove(thumbs_zip_path)
        print("  [OK] 已清理zip包")
    else:
        print(f"\n[跳过] 缩略图zip已存在: {thumbs_zip_path}")
        extract_dir = os.path.join(FFHQ_DIR, 'thumbnails128x128')
    
    # ── 统计 ──
    if os.path.exists(extract_dir):
        files = [f for f in os.listdir(extract_dir) if f.endswith('.png')]
        print(f"\n[结果] FFHQ可用图片: {len(files)} 张")
        if len(files) < target_count:
            print(f"  [注意] 可用数量 ({len(files)}) < 目标数量 ({target_count})")
    
    return extract_dir if os.path.exists(extract_dir) else None


# ═══════════════════════════════════════════════════
# 2. ISIC Archive 下载 (REST API)
# ═══════════════════════════════════════════════════
def download_isic_subset(target_count: int = 500):
    """下载ISIC皮肤病变图像用于瑕疵特征校准
    
    使用 ISIC Archive REST API (无需认证):
    - GET https://api.isic-archive.com/api/v2/images/?limit=N
    - 优先筛选 diagnosis=nevus (痣), seborrheic keratosis (脂溢性角化, 类似色斑)
    """
    print("\n" + "=" * 60)
    print("  下载 ISIC 色斑/病变图像数据集")
    print("=" * 60)
    
    API_BASE = "https://api.isic-archive.com/api/v2"
    
    # ── 尝试多种查询获取图像列表 ──
    diagnoses = ['nevus', 'seborrheic keratosis', 'lentigo', 'solar lentigo']
    all_images = []
    
    for diagnosis in diagnoses:
        print(f"\n[查询] diagnosis={diagnosis} ...")
        offset = 0
        batch_size = 50
        
        while len(all_images) < target_count:
            url = f"{API_BASE}/images/?limit={batch_size}&offset={offset}"
            if diagnosis:
                url += f"&diagnosis={diagnosis}"
            
            req = urllib.request.Request(url, headers={
                'Accept': 'application/json',
                'User-Agent': 'BeautyAnalyzer/1.0',
            })
            
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    
                results = data.get('results', [])
                if not results:
                    break
                    
                all_images.extend(results)
                offset += batch_size
                print(f"\r  已获取: {len(all_images)} 张 (= {len(all_images) * 100 // target_count}%)", 
                      end='', flush=True)
                
                if len(all_images) >= target_count:
                    break
                
                time.sleep(0.5)  # 避免触发rate limit
                
            except urllib.error.HTTPError as e:
                print(f"\n  [错误] HTTP {e.code}: {url}")
                break
            except Exception as e:
                print(f"\n  [错误] {e}")
                break
        
        if len(all_images) >= target_count:
            break
    
    print(f"\n  [结果] 共获取 {len(all_images)} 条图像元数据")
    
    if not all_images:
        print("  [备用] 尝试不指定diagnosis获取任意图像...")
        url = f"{API_BASE}/images/?limit={target_count}&offset=0"
        try:
            req = urllib.request.Request(url, headers={
                'Accept': 'application/json',
                'User-Agent': 'BeautyAnalyzer/1.0',
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            all_images = data.get('results', [])
            print(f"  [结果] 获取 {len(all_images)} 条")
        except Exception as e:
            print(f"  [错误] {e}")
    
    # ── 下载图像 ──
    downloaded = 0
    max_download = min(target_count, len(all_images))
    
    print(f"\n[下载] 开始下载 {max_download} 张图像...")
    
    for i, img_meta in enumerate(all_images[:max_download]):
        isic_id = img_meta.get('isic_id', '')
        
        # 从 files.full.url 获取下载地址
        files = img_meta.get('files', {})
        img_url = files.get('full', {}).get('url', '')
        if not img_url:
            # fallback: 用 thumbnail
            img_url = files.get('thumbnail_256', {}).get('url', '')
        if not img_url:
            continue
        
        diagnosis = img_meta.get('metadata', {}).get('clinical', {}).get('diagnosis_3', 'unknown')
        safe_name = f"{isic_id}.jpg"
        save_path = os.path.join(ISIC_DIR, safe_name)
        
        if os.path.exists(save_path):
            downloaded += 1
            continue
        
        # 下载图像 (S3 URL)
        try:
            req = urllib.request.Request(img_url, headers={
                'User-Agent': 'BeautyAnalyzer/1.0',
            })
            
            with urllib.request.urlopen(req, timeout=30) as resp:
                img_data = resp.read()
            
            with open(save_path, 'wb') as f:
                f.write(img_data)
            
            downloaded += 1
            
            if downloaded % 50 == 0:
                print(f"\r  下载进度: {downloaded}/{max_download} "
                      f"({downloaded*100//max_download}%)  "
                      f"[最新: {diagnosis}]", end='', flush=True)
            
            time.sleep(0.2)
            
        except Exception as e:
            if downloaded <= 3:
                print(f"\n  [跳过] {isic_id}: {e}")
            continue
    
    print(f"\r  下载完成: {downloaded}/{max_download} 张图像")
    
    # ── 保存元数据 ──
    meta_path = os.path.join(ISIC_DIR, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_images[:max_download], f, indent=2, ensure_ascii=False)
    print(f"  [OK] 元数据保存: {meta_path}")
    
    return ISIC_DIR if downloaded > 0 else None


# ═══════════════════════════════════════════════════
# 3. 主入口
# ═══════════════════════════════════════════════════
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='下载校准数据集')
    parser.add_argument('--ffhq', type=int, default=1000, help='FFHQ下载数量 (0=跳过)')
    parser.add_argument('--isic', type=int, default=500, help='ISIC下载数量 (0=跳过)')
    parser.add_argument('--skip-ffhq', action='store_true', help='跳过FFHQ')
    parser.add_argument('--skip-isic', action='store_true', help='跳过ISIC')
    args = parser.parse_args()
    
    if not args.skip_ffhq and args.ffhq > 0:
        print(f"\n[FFHQ] 目标: {args.ffhq} 张洁净人脸\n")
        download_ffhq_subset(target_count=args.ffhq)
    
    if not args.skip_isic and args.isic > 0:
        print(f"\n[ISIC] 目标: {args.isic} 张色斑/病变图\n")
        download_isic_subset(target_count=args.isic)
    
    print("\n" + "=" * 60)
    print("  下载完成")
    print("=" * 60)
    print(f"\nFFHQ (洁净脸): {FFHQ_DIR}")
    print(f"  {len([f for f in os.listdir(FFHQ_DIR) if f.endswith('.png')]) if os.path.exists(FFHQ_DIR) else 0} 张" 
          if os.path.exists(FFHQ_DIR) else "  目录不存在")
    print(f"\nISIC (色斑): {ISIC_DIR}")
    isic_pics = [f for f in os.listdir(ISIC_DIR) if f.endswith(('.jpg','.png'))] if os.path.exists(ISIC_DIR) else []
    print(f"  {len(isic_pics)} 张")
