import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# 配置路径 (请确保这里是你真正的 best.pt 路径)
YOLO_MODEL_PATH = r"D:\project\design\runs\detect\train_mixed\weights\best.pt"

print("🔧 正在初始化 AI 引擎...")
yolo_model = YOLO(YOLO_MODEL_PATH)
ocr_model = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
print("✅ 准备就绪，正在开启摄像头...")


# 提取画中文的复用函数
def draw_chinese_box(cv2_img, box, text, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(cv2_img, (x1, y1), (x2, y2), color, 3)

    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("simhei.ttf", 30)
    except:
        font = ImageFont.load_default()

    text_y_pos = max(0, y1 - 40)
    text_bbox = draw.textbbox((x1, text_y_pos), text, font=font)
    draw.rectangle([text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5], fill=color)
    draw.text((x1, text_y_pos), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# 开启物理摄像头 (0 表示自带的第一个摄像头)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 实时侦测
    results = yolo_model(frame, conf=0.6, verbose=False)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        box_color = (255, 100, 50) if class_id == 0 else (50, 205, 50)

        # 实时抠图 OCR 认字
        cropped_plate = frame[y1:y2, x1:x2]
        # 注意：每一帧都跑 OCR 可能会让摄像头画面变卡，因为你的 4060 在满负荷运转！
        ocr_result = ocr_model.ocr(cropped_plate, cls=True)

        plate_text = ""
        if ocr_result and ocr_result[0]:
            plate_text = ocr_result[0][0][1][0]
        else:
            plate_text = "分析中..."

        frame = draw_chinese_box(frame, (x1, y1, x2, y2), plate_text, box_color)

    # 渲染监控画面
    cv2.imshow("Smart Security Camera (Press 'q' to exit)", frame)

    # 敲击键盘上的 'q' 键退出监控
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()