import os
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ================= 1. 全局配置区 =================
# ⚠️ 请确保这里指向你今天刚跑完的 100 轮双拼模型的 best.pt
YOLO_MODEL_PATH = r"D:\project\design\runs\detect\train_mixed\weights\best.pt"
TEST_IMAGE_PATH = r"D:\project\design\test\testl.jpg"

# 我们今天训练的心血：类别字典
PLATE_TYPE = {
    0: "🟦 传统蓝牌 (燃油车)",
    1: "🟩 新能源绿牌"
}


# ===============================================

def main():
    print("🔧 [系统初始化] 正在加载双拼 AI 模型...")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        ocr_model = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        print("✅ [系统初始化] YOLO11 与 PaddleOCR 引擎双双加载完毕！\n")
    except Exception as e:
        print(f"❌ [致命错误] 模型加载失败，请检查路径: {e}")
        return

    img = cv2.imread(TEST_IMAGE_PATH)
    if img is None:
        print(f"❌ [致命错误] 找不到测试图片 '{TEST_IMAGE_PATH}'")
        return

    print(f"🚀 [流水线启动] 正在扫描图片: {TEST_IMAGE_PATH}")
    results = yolo_model(img, conf=0.5, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("⚠️ [检测结果] 画面中未发现车牌目标。")
        return

    print(f"🎯 [检测结果] 成功锁定 {len(boxes)} 个车牌目标，开始智能解析...\n")

    for i, box in enumerate(boxes):
        # 提取坐标和置信度
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = box.conf[0].item()

        # 🌟 提取类别 (0 还是 1)，判断是蓝牌还是绿牌
        class_id = int(box.cls[0].item())
        plate_color = PLATE_TYPE.get(class_id, "未知类型")

        # 内存级切片提取车牌
        cropped_plate = img[y1:y2, x1:x2]
        cv2.imwrite(f"debug_crop_{i}.jpg", cropped_plate)  # 保存下来看看刀法

        # 送入 OCR 识别文字
        ocr_result = ocr_model.ocr(cropped_plate, cls=True)

        # ====== 终极版华丽输出 ======
        print(f"--- 🚗 目标 [{i + 1}] 档案 ---")
        print(f"🎨 车牌类型: {plate_color} (YOLO 判定可信度: {confidence:.2f})")

        if ocr_result and ocr_result[0]:
            for line in ocr_result[0]:
                text = line[1][0]
                ocr_conf = line[1][1]
                print(f"🔤 车牌号码: {text} (OCR 识别可信度: {ocr_conf:.2f})")
        else:
            print("🔤 车牌号码: [模糊/未识别出有效文字]")
        print("-" * 30)


if __name__ == '__main__':
    main()