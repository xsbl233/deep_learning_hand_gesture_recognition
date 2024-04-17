import numpy as np

test=np.array(test_L)

def normalize_2d_list(data):
    normalized_data = []
    for col in range(len(data[0])):
        column = [row[col] for row in data]
        max_val = max(column)
        min_val = min(column)
        normalized_column = [(val - min_val) / (max_val - min_val) for val in column]
        normalized_data.append(normalized_column)
    return list(zip(*normalized_data))


normalized_data = normalize_2d_list(test)
print(normalized_data)

import cv2
import mediapipe as mp

def solve_gesture_with_mediapipe(frame, image, results_path=""):
    z = 0
    for point in normalized_data:
        x = int(point[0] * (frame.shape[1] - 200) + 100)
        y = int( (1-point[1]) * (frame.shape[0]-200) + 100)
        print(x,y,z)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # 画出关键点
        font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型
        font_scale = 1  # 字体大小
        font_color = (0, 0, 0)  # 字体颜色，这里是黑色
        thickness = 2  # 字体粗细
        cv2.putText(frame, str(z), (x, y), font, font_scale, font_color, thickness)
        z = z + 1

def get_frame_with_photos(filename):
    frame = cv2.imread(filename)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换颜色空间

    # 处理识别结果
    solve_gesture_with_mediapipe(frame, image)
    cv2.imshow('Gesture Recognition', frame)  # 显示结果
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

get_frame_with_photos("./data/1.jpg")
list.append([(list[0][0]+list[5][0]+list[9][0]+list[13][0]+list[17][0])/5,
            (list[0][1]+list[5][1]+list[9][1]+list[13][1]+list[17][1])/5,
            (list[0][2]+list[5][2]+list[9][2]+list[13][2]+list[17][2])/5])
