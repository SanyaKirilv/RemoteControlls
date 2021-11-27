import tkinter as tk
import cv2
import mediapipe as mp
import mouse
from pynput.mouse import Button, Controller
import numpy as np
import win32api
from win32con import *

def frame_pos2screen_pos(frame_size=(1080, 1920), screen_size=(1080, 1920), frame_pos=None):
    x, y = screen_size[1] / frame_size[0], screen_size[0] / frame_size[1]

    screen_pos = [frame_pos[0] * x, frame_pos[1] * y]

    return screen_pos


def euclidean(pt1, pt2):
    d = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    return d


mouseScroll = Controller()

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

euclidean((0, 3), (0, 0))
ssize = (screen_height, screen_width)

cam = cv2.VideoCapture(0)

fsize = (screen_height, screen_width)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

left, top, right, bottom = (np.int(screen_width * 0.15), np.int(screen_height * 0.15), np.int(screen_width * 0.85), np.int(screen_height * 0.85))

events = ["sclick", "dclick", "rclick", "drag", "release", "scroll up", "scroll down"]

check_every = 15
check_cnt = 0
last_event = None

#out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'XVID'), 144, (fsize[1], fsize[0]))

with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=1,
                    min_detection_confidence=0.4) as hands:
    while cam.isOpened():
        ret, frame = cam.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (fsize[1], fsize[0]))

        h, w, _ = frame.shape

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand_landmarks in res.multi_hand_landmarks:

                thumb_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                    w, h)

                index_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                    w, h)

                index_mcp = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                    w, h)

                index_pip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    w, h)

                thumb_mcp = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
                    w, h)

                middle_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                    w, h)

                middle_mcp = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                    w, h)

                ring_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                    w, h)

                ring_mcp = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
                    w, h)

                pinky_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                    w, h)

                pinky_mcp = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
                    w, h)

                thumb_pip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
                    w, h)
                #INDEX FINGER DOWN POINT
                index_pip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
                    w, h)
                #MIDDLE FINGER DOWN POINT
                middle_pip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
                    w, h)

                index_tipm = [_, _]
                if middle_mcp is not None:
                    index_tipm[0] = np.clip(middle_mcp[0], left, right)
                    index_tipm[1] = np.clip(middle_mcp[1], top, bottom)

                    index_tipm[0] = (middle_mcp[0] - left) * fsize[0] / (right - left)
                    index_tipm[1] = (middle_mcp[1] - top) * fsize[1] / (bottom - top)
                if middle_mcp is not None and pinky_mcp is not None:
                    t = euclidean(middle_mcp, pinky_mcp)

                if check_cnt == check_every:
                    # LEFT CLICK
                    if index_pip is not None and index_tip is not None:
                        if euclidean(index_pip, index_tip) < t / 2:
                            last_event = "left click"
                        else:
                            if last_event == "left click":
                                last_event = "release"

                    # RIGHT CLICK
                    if middle_pip is not None and middle_tip is not None:
                        if euclidean(middle_pip, middle_tip) < t / 2:
                            last_event = "right click"
                        else:
                            if last_event == "right click":
                                last_event = "release"

                    # DRAG
                    if ring_tip is not None and thumb_tip is not None:
                        if euclidean(ring_tip, thumb_tip) < t / 2:
                            if last_event == "press" or last_event == "after press":
                                last_event = "after press"
                            else:
                                last_event = "press"
                        else:
                            if last_event == "press":
                                last_event = "release"

                    if thumb_pip is not None and thumb_tip is not None:
                        if euclidean(thumb_pip, thumb_tip) < t / 2:
                            last_event = "double click"
                        else:
                            if last_event == "double click":
                                last_event = "release"

                    check_tic = 0

                    if thumb_tip is not None and middle_tip is not None and index_tip is not None and ring_tip is not None and pinky_tip is not None\
                            and middle_mcp is not None and index_mcp is not None and ring_mcp is not None and pinky_mcp is not None:
                        if (middle_tip is None or euclidean(middle_tip, middle_mcp) < t / 3 * 2) and (middle_tip is None or euclidean(index_tip, index_mcp) < t / 3 * 2) and (middle_tip is None or euclidean(ring_tip, ring_mcp) < t / 3 * 2) and (middle_tip is None or euclidean(pinky_tip, pinky_mcp) < t / 3 * 2):
                            if index_tip[1] - thumb_tip[1] > t/2:
                                last_event = "scroll up"
                            else:
                                if last_event == "scroll up":
                                    last_event = "release"
                        else:
                            if last_event == "scroll up":
                                last_event = "release"

                    if thumb_tip is not None\
                            and middle_mcp is not None and index_mcp is not None and ring_mcp is not None and pinky_mcp is not None:
                        if (middle_tip is None or euclidean(middle_tip, middle_mcp) < t / 3 * 2) and (index_tip is None or euclidean(index_tip, index_mcp) < t / 3 * 2) and (ring_tip is None or euclidean(ring_tip, ring_mcp) < t / 3 * 2) and (pinky_tip is None or euclidean(pinky_tip, pinky_mcp) < t / 3 * 2):
                            if thumb_tip[1] - index_tip[1] > t/2:
                                last_event = "scroll down"
                            else:
                                if last_event == "scroll down":
                                    last_event = "release"
                        else:
                            if last_event == "scroll down":
                                last_event = "release"

                    check_cnt = 0

                #if check_cnt > 1:
                #    last_event = None

                screen_pos = frame_pos2screen_pos(fsize, ssize, index_tipm)

                # print(screen_pos)

                mouse.move(screen_pos[0], screen_pos[1])

                if check_cnt == 0:
                    if last_event == "left click":
                        mouse.click()
                    elif last_event == "right click":
                        mouse.right_click()
                    elif last_event == "after press":
                        _
                    elif last_event == "press":
                        mouse.press()
                    elif last_event == "scroll up":
                        mouseScroll.scroll(0, 2)
                    elif last_event == "scroll down":
                        mouseScroll.scroll(0, -2)
                    elif last_event == "double click":
                        mouse.double_click()
                    else:
                        mouse.release()

                    print(last_event)
                    #cv2.putText(frame, last_event, (32, 32), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
                check_cnt += 1
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        #cv2.imshow("RemoteControl", frame)
        #out.write(frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
cam.release()
#out.release()
#cv2.destroyAllWindows()