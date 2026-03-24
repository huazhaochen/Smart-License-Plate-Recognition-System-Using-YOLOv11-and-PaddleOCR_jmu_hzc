import os
import cv2
import csv
import time
from datetime import datetime
import numpy as np
import gradio as gr
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# ================= 1. 全局配置与模型加载 =================
YOLO_MODEL_PATH = r"D:\project\design\runs\detect\train_mixed\weights\best.pt"
HISTORY_DIR = r"D:\project\design\history"
IMG_SAVE_DIR = os.path.join(HISTORY_DIR, "images")
CSV_FILE = os.path.join(HISTORY_DIR, "records.csv")
VIP_FILE = os.path.join(HISTORY_DIR, "marked_vehicles.txt")

os.makedirs(IMG_SAVE_DIR, exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(["识别时间", "车牌类型", "车牌号码", "置信度", "分析标记", "抓拍文件"])

print("🔧 正在初始化 AI 引擎，请稍候...")
yolo_model = YOLO(YOLO_MODEL_PATH)
ocr_model = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
print("✅ 引擎加载完毕！")

PLATE_TYPE = {0: "🟦 传统蓝牌", 1: "🟩 新能源绿牌"}
MARKED_VEHICLES = set()
PLATE_COOLDOWN = {}  # 🌟 新增：用于实时监控的“节流缓存库”


# ================= 2. 微型数据库管理 =================
def load_vips():
    global MARKED_VEHICLES
    if os.path.exists(VIP_FILE):
        with open(VIP_FILE, 'r', encoding='utf-8') as f:
            MARKED_VEHICLES = set(line.strip() for line in f if line.strip())
    return list(MARKED_VEHICLES)


def save_vips():
    with open(VIP_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(MARKED_VEHICLES))


load_vips()


def add_vip(plate):
    if plate and plate not in MARKED_VEHICLES:
        MARKED_VEHICLES.add(plate)
        save_vips()
    return "\n".join(MARKED_VEHICLES), ""


def remove_vip(plate):
    if plate in MARKED_VEHICLES:
        MARKED_VEHICLES.remove(plate)
        save_vips()
    return "\n".join(MARKED_VEHICLES), ""


# ================= 3. 图像绘制工具 =================
def draw_chinese_box(cv2_img, box, text, rgb_color):
    x1, y1, x2, y2 = box
    img_h = cv2_img.shape[0]
    dynamic_font_size = max(30, int(img_h * 0.035))

    pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([x1, y1, x2, y2], outline=rgb_color, width=4)

    try:
        font = ImageFont.truetype("simhei.ttf", dynamic_font_size)
    except:
        font = ImageFont.load_default()

    text_y_pos = max(0, y1 - dynamic_font_size - 10)
    text_bbox = draw.textbbox((x1, text_y_pos), text, font=font)
    draw.rectangle([text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5], fill=rgb_color)
    draw.text((x1, text_y_pos), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ================= 4. 核心大脑 =================
def process_image(image, need_flip, is_stream=False):
    if image is None:
        return None, "⚠️ 请先提供画面。"

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if need_flip:
        img_bgr = cv2.flip(img_bgr, 1)

    draw_img = img_bgr.copy()
    results = yolo_model(img_bgr, conf=0.5, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return draw_img, "⏳ 监控中：未锁定车牌..." if is_stream else "⚠️ 未锁定车牌。"

    final_text = ""
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_timestamp = time.time()  # 获取当前物理时间

    saved_count = 0  # 记录本次存了几张

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        confidence = box.conf[0].item()
        plate_color_name = PLATE_TYPE.get(class_id, "未知类型")
        box_rgb = (0, 110, 255) if class_id == 0 else (30, 200, 80)

        cropped_plate = img_bgr[y1:y2, x1:x2]
        ocr_result = ocr_model.ocr(cropped_plate, cls=True)

        plate_text = "未识别"
        if ocr_result and ocr_result[0]:
            plate_text = "".join([line[1][0] for line in ocr_result[0]])

        display_text = f"{plate_text} ({int(confidence * 100)}%)"
        draw_img = draw_chinese_box(draw_img, (x1, y1, x2, y2), display_text, box_rgb)

        is_marked = plate_text in MARKED_VEHICLES
        analysis_tag = "🔴 [警报] 发现重点车辆" if is_marked else "🟢 常规通行车辆"

        # 🌟 核心逻辑：判断是否需要保存日志
        # 如果是静态抓拍 (is_stream=False) -> 必须保存
        # 如果是动态监控 (is_stream=True) -> 检查这辆车是不是在 5 秒内刚存过
        should_save = False
        if not is_stream:
            should_save = True
        else:
            if plate_text != "未识别":
                # 距离上次看到这辆车是否超过了 5 秒
                if plate_text not in PLATE_COOLDOWN or (current_timestamp - PLATE_COOLDOWN[plate_text] > 5):
                    should_save = True
                    PLATE_COOLDOWN[plate_text] = current_timestamp  # 更新时间戳

        if should_save:
            img_filename = f"capture_{file_time}_{i}.jpg"
            cv2.imwrite(os.path.join(IMG_SAVE_DIR, img_filename), draw_img)
            with open(CSV_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
                csv.writer(f).writerow(
                    [now_time, plate_color_name, plate_text, f"{confidence:.2f}", analysis_tag, img_filename])
            saved_count += 1

        final_text += f"🚗 【目标 {i + 1}】 {plate_color_name} | {plate_text}\n"
        final_text += f"🛂 {analysis_tag}\n" + "━" * 25 + "\n"

    if saved_count > 0:
        final_text += f"\n📁 数据已自动同步至本地日志库。"

    return cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB), final_text


def run_static(img, flip): return process_image(img, flip, is_stream=False)


def run_stream(img, flip): return process_image(img, flip, is_stream=True)


def open_history_folder():
    try:
        os.startfile(HISTORY_DIR)
        return "✅ 已在 Windows 中打开日志文件夹！"
    except Exception as e:
        return f"❌ 无法打开文件夹: {e}"


# ================= 5. 多层级企业 UI =================
with gr.Blocks(theme=gr.themes.Soft(), title="智慧治安卡口") as demo:
    gr.Markdown("# 👁️ 智慧卡口：全栖视觉处理终端")

    with gr.Tabs():
        # --- 标签页 1 ---
        with gr.Tab("📸 卡口抓拍研判"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="监控探头画面 (上传/拍照)", sources=["upload", "webcam", "clipboard"])
                    flip_chk = gr.Checkbox(label="🔄 开启镜像翻转", value=False)
                    submit_btn = gr.Button("🚀 立即分析", variant="primary")
                    clear_btn = gr.Button("🗑️ 清空")
                with gr.Column():
                    # 🌟 严格修正：去掉了废弃参数，最新版自带放大镜功能
                    img_output = gr.Image(label="🎯 智慧追踪锁定", interactive=False)
                    text_output = gr.Textbox(label="📊 研判报告", lines=8)
                    gr.Button("📂 打开本地日志库").click(fn=open_history_folder, outputs=gr.Markdown(""))

            submit_btn.click(fn=run_static, inputs=[img_input, flip_chk], outputs=[img_output, text_output])
            clear_btn.click(lambda: (None, False, ""), outputs=[img_input, flip_chk, text_output])

        # --- 标签页 2 ---
        with gr.Tab("🎥 实时动态布控"):
            gr.Markdown("⚠️ 注意：本模块直接接管物理摄像头，带5秒防抖机制（同车牌5秒内仅入库一次）。")
            with gr.Row():
                with gr.Column():
                    stream_input = gr.Image(sources=["webcam"], streaming=True, label="实况监控画面")
                    stream_flip = gr.Checkbox(label="🔄 开启镜像翻转", value=False)
                with gr.Column():
                    # 🌟 严格修正：同样去掉废弃参数
                    stream_output = gr.Image(label="🎯 智慧追踪锁定", interactive=False)
                    stream_text = gr.Textbox(label="📊 研判报告", lines=8)
                    gr.Button("📂 打开本地日志库").click(fn=open_history_folder, outputs=gr.Markdown(""))

            stream_input.stream(fn=run_stream, inputs=[stream_input, stream_flip], outputs=[stream_output, stream_text])

        # --- 标签页 3 ---
        with gr.Tab("📋 重点车辆名单库"):
            with gr.Row():
                with gr.Column():
                    new_plate = gr.Textbox(label="输入需布控的车牌 (如: 苏A·6NK10)")
                    with gr.Row():
                        add_btn = gr.Button("➕ 加入布控库", variant="primary")
                        del_btn = gr.Button("🗑️ 撤销布控", variant="stop")
                with gr.Column():
                    plate_list = gr.Textbox(label="当前布控名单", value="\n".join(load_vips()), lines=10,
                                            interactive=False)

            add_btn.click(fn=add_vip, inputs=new_plate, outputs=[plate_list, new_plate])
            del_btn.click(fn=remove_vip, inputs=new_plate, outputs=[plate_list, new_plate])

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)






