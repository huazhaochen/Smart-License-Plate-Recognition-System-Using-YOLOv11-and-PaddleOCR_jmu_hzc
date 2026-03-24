from ultralytics import YOLO

if __name__ == '__main__':
    print("🚀 正在加载正统 YOLO11 模型...")
    # 依然使用最强轻量级底座
    model = YOLO('yolo11n.pt')

    print("开始召唤 RTX 4060 进行【蓝绿双类别】终极炼丹！")
    model.train(
        data=r'D:\project\design\datasets\dataset_yolo\data.yaml',  # 数据集路径

        # ======== 进阶参数调优区 ========
        epochs=100,  # 总训练轮数提升至 100 轮
        patience=25,  # 早停机制：如果连续 25 轮成绩不提升，自动停止防过拟合
        batch=16,  # 显卡每次吞吐量，榨干 4060 的 8G 显存
        # ====================================

        imgsz=640,  # 图片输入尺寸
        device=0,  # 指定第一张独立显卡
        workers=0,  # Windows 防卡死必加
        name='train_mixed'  # 战绩保存的文件夹名字
    )

    print("双类别终极模型训练结束！")