from ultralytics import YOLO
import cv2
import numpy as np

# 只保留上半身关键点编号（COCO 格式）
UPPER_BODY_KPTS = [0,1,2,3,4,5,6,7,8,9,10]

# 上半身骨架连线（使用 COCO 原始连接）
UPPER_BODY_PAIRS = [
    (5,7), (7,9),      # 左肩-左肘-左腕
    (6,8), (8,10),     # 右肩-右肘-右腕
    (5,6)              # 左肩-右肩
]

def draw_upper_body_keypoints(frame, keypoints, conf_threshold=0.3):
    """
    keypoints: shape = (17,3) -> (x,y,confidence)
    """
    for i in UPPER_BODY_KPTS:
        x, y, conf = keypoints[i]
        if conf > conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0,255,0), -1)

    # 画骨架线
    for (a, b) in UPPER_BODY_PAIRS:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        
        if ca > conf_threshold and cb > conf_threshold:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (0,255,0), 2)


def run_pose_video(input_path, output_path="upper_body_output.mp4"):
    model = YOLO("yolo11s-pose.pt")

    cap = cv2.VideoCapture(input_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            if hasattr(r, "keypoints") and r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy()   # [num_person,17,2]
                conf = r.keypoints.conf.cpu().numpy() # [num_person,17]

                for i in range(len(kpts)):
                    # 拼成 (17,3)
                    kp = np.concatenate([kpts[i], conf[i][:,None]], axis=1)
                    draw_upper_body_keypoints(frame, kp)

        out.write(frame)

        cv2.imshow("YOLO11 Upper Body Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose_video("1.mp4", "upper_body_pose.mp4")
