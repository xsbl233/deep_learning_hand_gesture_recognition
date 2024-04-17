import cv2
import mediapipe as mp

def solve_gesture_with_mediapipe(frame, image, results_path=""):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils
    results = hands.process(image)  # 手势识别
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)  # 用于指定地标如何在图中连接。
            for point in hand_landmarks.landmark:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                z = round(point.z, 2)
                print(x, y, z)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 画出关键点
                font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
                font_scale = 1  # 字体大小
                font_color = (0, 0, 0)  # 字体颜色，这里是黑色
                thickness = 2  # 字体粗细
                cv2.putText(frame, str(z), (x, y), font, font_scale, font_color, thickness)

def get_gesture_with_capture():
    # coding:utf-8
    cap = cv2.VideoCapture(0)  # 打开摄像头
    while True:
        ret, frame = cap.read()  # 读取视频帧
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间

        # 处理识别结果
        solve_gesture_with_mediapipe(frame, image)
        cv2.imshow('Gesture Recognition', frame)  # 显示结果
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_gesture_with_photos(filename):
    frame = cv2.imread(filename)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间

    # 处理识别结果
    solve_gesture_with_mediapipe(frame, image)
    cv2.imshow('Gesture Recognition', frame)  # 显示结果
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


get_gesture_with_photos("./data/1.jpg")
