from ultralytics import YOLO
import cv2
import numpy as np

# 上半身关键点编号（COCO）
NOSE, LEYE, REYE, LEAR, REAR = 0,1,2,3,4
LSHO, RSHO = 5,6
LELB, RELB = 7,8
LWRI, RWRI = 9,10

# 只保留上半身关键点
UPPER_BODY_KPTS = [0,1,2,3,4,5,6,7,8,9,10]

def compute_angle(a, b, c):
    """
    计算点 b 为顶点的角度 (a-b-c)
    a,b,c: [x,y]
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def draw_kpts(frame, keypoints, conf_th=0.3):
    for i in UPPER_BODY_KPTS:
        x,y,conf = keypoints[i]
        if conf > conf_th:
            cv2.circle(frame, (int(x),int(y)), 5, (0,255,0), -1)

def run_pose_video(input_path, output_path="upper_body_angle_count.mp4"):
    model = YOLO("yolo11n-pose.pt")

    cap = cv2.VideoCapture(input_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    left_count = 0
    right_count = 0

    # 状态机：是否处于“低角度状态”
    left_low = False
    right_low = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        for r in results:
            if not hasattr(r, "keypoints") or r.keypoints is None:
                continue

            kpts = r.keypoints.xy.cpu().numpy()
            conf = r.keypoints.conf.cpu().numpy()

            for i in range(len(kpts)):
                kp = np.concatenate([kpts[i], conf[i][:,None]], axis=1)

                draw_kpts(frame, kp)

                # 获取关键点
                L_sho, L_elb, L_wri = kp[LSHO][:2], kp[LELB][:2], kp[LWRI][:2]
                R_sho, R_elb, R_wri = kp[RSHO][:2], kp[RELB][:2], kp[RWRI][:2]

                # 左手臂角度
                left_angle = compute_angle(L_sho, L_elb, L_wri)
                # 右手臂角度
                right_angle = compute_angle(R_sho, R_elb, R_wri)

                # ---- 左臂计数逻辑 ----
                if left_angle < 10:
                    left_low = True
                if left_low and left_angle > 160:
                    left_count += 1
                    left_low = False

                # ---- 右臂计数逻辑 ----
                if right_angle < 10:
                    right_low = True
                if right_low and right_angle > 160:
                    right_count += 1
                    right_low = False

                # 显示角度与计数
                cv2.putText(frame, f"L angle: {left_angle:.1f}",  (30, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"R angle: {right_angle:.1f}", (30, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                cv2.putText(frame, f"L count: {left_count}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                cv2.putText(frame, f"R count: {right_count}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        out.write(frame)

        cv2.imshow("Angle Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose_video("1.mp4", "upper_body_angle_count.mp4")
