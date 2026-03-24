import os
import shutil
import cv2
import random

# ================= 配置区 =================
# 注意检查这里的路径对不对
GREEN_DIR = r"D:\project\design\CCPD\CCPD2020\CCPD2020\ccpd_green\train"

TARGET_DIR = r"D:\project\design\dataset_yolo"
SAMPLE_SIZE = 3000
TRAIN_RATIO = 0.8


# ==========================================

def setup_env():
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

    # 🌟 修复部分：把反斜杠的替换操作拿到了 f-string 外面
    target_dir_str = TARGET_DIR.replace('\\', '/')

    yaml_content = f"""path: {target_dir_str}
train: images/train
val: images/val

names:
  0: green_plate
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print("✅ data.yaml 配置文件已自动生成！")


def process_green_plates():
    print("🚀 开始处理绿牌数据...")
    files = [f for f in os.listdir(GREEN_DIR) if f.endswith('.jpg')]

    actual_sample_size = min(SAMPLE_SIZE, len(files))

    random.shuffle(files)
    selected_files = files[:actual_sample_size]

    train_size = int(len(selected_files) * TRAIN_RATIO)
    splits = {
        "train": selected_files[:train_size],
        "val": selected_files[train_size:]
    }

    success_count = 0
    for split_type, file_list in splits.items():
        print(f"-> 正在生成 {split_type} 集 ({len(file_list)} 张)...")
        img_dir = os.path.join(TARGET_DIR, "images", split_type)
        lbl_dir = os.path.join(TARGET_DIR, "labels", split_type)

        for filename in file_list:
            try:
                parts = filename.split('-')
                if len(parts) < 4: continue
                p1, p2 = parts[2].split('_')
                xmin, ymin = map(int, p1.split('&'))
                xmax, ymax = map(int, p2.split('&'))

                img_path = os.path.join(GREEN_DIR, filename)
                img = cv2.imread(img_path)
                if img is None: continue
                h, w, _ = img.shape

                x_center = ((xmin + xmax) / 2.0) / w
                y_center = ((ymin + ymax) / 2.0) / h
                box_width = (xmax - xmin) / w
                box_height = (ymax - ymin) / h

                new_name = f"green_{success_count:05d}"
                with open(os.path.join(lbl_dir, new_name + ".txt"), 'w') as f:
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                shutil.copy(img_path, os.path.join(img_dir, new_name + ".jpg"))
                success_count += 1

            except Exception as e:
                continue

    print(f"\n🎉 全部清洗完成！有效生成 {success_count} 张绿牌数据。")


if __name__ == '__main__':
    setup_env()
    process_green_plates()