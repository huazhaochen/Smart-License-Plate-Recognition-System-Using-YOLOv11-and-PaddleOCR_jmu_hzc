import os
import shutil
import cv2
import random

# ================= 1. 终极版全局配置区 =================
# 蓝牌路径：蓝牌解压后的实际路径！
BLUE_DIR = r"D:\project\CCPD\CCPD2019\CCPD2019\ccpd_base"

# 绿牌路径
GREEN_DIR = r"D:\project\CCPD\CCPD2020\CCPD2020\ccpd_green\train"

# 目标数据集路径
TARGET_DIR = r"D:\project\design\datasets\dataset_yolo"

SAMPLE_SIZE = 3000  # 各抽 3000 张，1:1 绝对平衡
TRAIN_RATIO = 0.8


# =======================================================

def setup_env():
    print("🔧 [系统初始化] 正在重置 YOLO 数据集目录...")
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    dirs = [
        os.path.join(TARGET_DIR, "images", "train"),
        os.path.join(TARGET_DIR, "images", "val"),
        os.path.join(TARGET_DIR, "labels", "train"),
        os.path.join(TARGET_DIR, "labels", "val")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    yaml_path = os.path.join(TARGET_DIR, "data.yaml")
    target_dir_str = TARGET_DIR.replace('\\', '/')
    yaml_content = f"""path: {target_dir_str}
train: images/train
val: images/val

names:
  0: blue_plate
  1: green_plate
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print("✅ data.yaml 配置文件已更新为双类别 (蓝牌=0, 绿牌=1)！\n")


def process_dataset(source_dir, class_id, prefix):
    print(f"🚀 [数据流水线] 开始处理 {prefix} (类别ID: {class_id})...")
    if not os.path.exists(source_dir):
        print(f"❌ [致命错误] 找不到路径: {source_dir}，请仔细检查顶部配置区！")
        return

    files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    actual_sample = min(SAMPLE_SIZE, len(files))

    random.shuffle(files)
    selected_files = files[:actual_sample]

    train_size = int(len(selected_files) * TRAIN_RATIO)
    splits = {
        "train": selected_files[:train_size],
        "val": selected_files[train_size:]
    }

    success_count = 0
    for split_type, file_list in splits.items():
        img_dir = os.path.join(TARGET_DIR, "images", split_type)
        lbl_dir = os.path.join(TARGET_DIR, "labels", split_type)

        for filename in file_list:
            try:
                parts = filename.split('-')
                if len(parts) < 4: continue
                p1, p2 = parts[2].split('_')
                xmin, ymin = map(int, p1.split('&'))
                xmax, ymax = map(int, p2.split('&'))

                img_path = os.path.join(source_dir, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                h, w, _ = img.shape

                x_center = ((xmin + xmax) / 2.0) / w
                y_center = ((ymin + ymax) / 2.0) / h
                box_width = (xmax - xmin) / w
                box_height = (ymax - ymin) / h

                new_name = f"{prefix}_{success_count:05d}"
                with open(os.path.join(lbl_dir, new_name + ".txt"), 'w') as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                shutil.copy(img_path, os.path.join(img_dir, new_name + ".jpg"))
                success_count += 1

            except Exception as e:
                continue
    print(f"🎯 {prefix} 数据集处理完成！有效提取 {success_count} 张。\n")


if __name__ == '__main__':
    setup_env()
    process_dataset(BLUE_DIR, class_id=0, prefix="blue")
    process_dataset(GREEN_DIR, class_id=1, prefix="green")
    print("🎉 [全局通知] 蓝绿双拼数据集构建完毕！可以开始最终训练了！")