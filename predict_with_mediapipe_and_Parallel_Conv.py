import cv2
import mediapipe as mp
import torch
import numpy as np
from scipy import ndimage as ndimage
from train_model import HandGestureNet

def get_gesture_with_mediapipe(frame, image, result_path=""):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils
    results = hands.process(image)
    ret = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            tmp = []
            for point in hand_landmarks.landmark:
                x = point.x
                y = point.y
                z = point.z
                tmp.append(x)
                tmp.append(y)
                tmp.append(z)
            ret.append(np.array(tmp))
    print(ret)
    ret_arr = np.array(ret)
    print(ret_arr)
    return ret_arr


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

def test(filename):
    frame = cv2.imread(filename)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间

    # 处理识别结果
    return get_gesture_with_mediapipe(frame, image)

def resize_gestures(input_gestures, final_length=100):
    """
    Resize the time series by interpolating them to the same length

    Input:
        - input_gestures: list of numpy.ndarray tensors.
              Each tensor represents a single gesture.
              Gestures can have variable durations.
              Each tensor has a shape: (duration, channels)
              where duration is the duration of the individual gesture
                    channels = 44 = 2 * 22 if recorded in 2D and
                    channels = 66 = 3 * 22 if recorded in 3D
    Output:
        - output_gestures: one numpy.ndarray tensor.
              The output tensor has a shape: (records, final_length, channels)
              where records = len(input_gestures)
                   final_length is the common duration of all gestures
                   channels is the same as above
    """
    # please use python3. if you still use python2, important note: redefine the classic division operator / by importing it from the __future__ module
    output_gestures = np.array([np.array(
        [ndimage.zoom(x_i.T[j], final_length / len(x_i), mode='reflect') for j in range(np.size(x_i, 1))]).T for x_i
                                   in input_gestures])
    return output_gestures

demo_gesture_batch=[]
demo_gesture_batch.append(test("./data/1.jpg"))
demo_gesture_batch.append(test("./data/2.jpg"))
print(demo_gesture_batch)
demo_gesture_batch = resize_gestures(demo_gesture_batch)
print(demo_gesture_batch)
demo_gesture_batch = torch.tensor(demo_gesture_batch, dtype=torch.float32)

n_classes = 14
duration = 100
n_channels = 63
learning_rate = 1e-3

model = HandGestureNet(n_channels=n_channels, n_classes=n_classes)

model.load_state_dict(torch.load('.\\saves\\gesture_pretrained_model.pt'))
model.eval()
test_batch = torch.randn(32, duration, n_channels)
print(demo_gesture_batch)
with torch.no_grad():
    predictions = model(demo_gesture_batch)
    _, predictions = predictions.max(dim=1)
    print("Predicted gesture classes: {}".format(predictions.tolist()))