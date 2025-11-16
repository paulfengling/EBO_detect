from ultralytics import YOLO
import cv2

def run_pose_video(input_path, output_path="output.mp4"):
    # 1. 加载 YOLO11-Pose 模型（例如：yolo11n-pose.pt 或 yolo11s-pose.pt）
    model = YOLO("yolo11n-pose.pt")

    # 2. 打开视频
    cap = cv2.VideoCapture(input_path)

    # 获取视频信息
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # 3. 输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. YOLO11-Pose 推理
        results = model(frame, verbose=False)

        # 5. 直接使用 results[0].plot() 在图像上绘制姿态点/骨架/框
        plotted = results[0].plot()

        # 写入输出视频
        out.write(plotted)

        # 可视化（可关闭）
        cv2.imshow("YOLO11 Pose", plotted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose_video("1.mp4", "pose_output.mp4")
