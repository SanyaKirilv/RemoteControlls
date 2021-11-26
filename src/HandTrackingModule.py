import tkinter as tk
import cv2
import mediapipe as mp
import mouse
import numpy as np


def frame_pos2screen_pos(frame_size=(1080 , 1920), screen_size=(1080, 1920), frame_pos=None):
    
    x,y = screen_size[1]/frame_size[0], screen_size[0]/frame_size[1]
    
    screen_pos = [frame_pos[0]*x, frame_pos[1]*y]
    
    return screen_pos

def Distance(pt1, pt2):
    d = np.sqrt((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)
    return d

#SCREEN PARAMS
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
ssize = (screen_height, screen_width)
cam = cv2.VideoCapture(0)
fsize = ssize
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#RED BOX PARAMS
left, top, right, bottom = (5, 5, 1915, 1075)

#POSSIBLE EVENTS
events = ["left click", "right click", "drag", "release", None]

#VARIABLES
check_every_tic = 10
check_tic = 0
previous_event = None
isRelease = 1

#SCREEN OUPUT
out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*'XVID'), 60, (fsize[1], fsize[0]))

with mp_hands.Hands(static_image_mode=True, max_num_hands = 1,min_detection_confidence=0.4) as hands:
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
                #BIG FINGER UP POINT
                thumb_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y, 
                    w, h)
                #INDEX FINGER UP POINT
                index_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
                    w, h)
                #MIDDLE FINGER UP POINT
                middle_tip = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y, 
                    w, h)

                #BIG FINGER DOWN POINT
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

                #CENTRAL POINT
                middle = mp_drawing._normalized_to_pixel_coordinates(
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x, 
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y, 
                    w, h)

            
                #POINTER POSITION
                index_tipm = list(middle)
                index_tipm[0] = np.clip(middle[0], left, right)
                index_tipm[1] = np.clip(middle[1], top, bottom)
                
                index_tipm[0] = (middle[0]-left) * fsize[0]/(right-left)
                index_tipm[1] = (middle[1]-top) * fsize[1]/(bottom-top)
                
                
                if check_tic == check_every_tic:

                    #LEFT CLICK
                    if index_pip is not None and index_tip is not None:
                        if Distance(index_pip, index_tip) < 10:
                            previous_event = events[0]
                            isRelease = 0
                        else:
                            if previous_event == events[0]:
                                previous_event = events[4]
                                isRelease = 1

                    #RIGHT CLICK
                    if middle_pip is not None and middle_tip is not None:
                        if Distance(middle_pip, middle_tip) < 10:
                            previous_event = events[1]
                            isRelease = 0
                        else:
                            if previous_event == events[1]:
                                previous_event = events[4]
                                isRelease = 1

                    #DRAG
                    if thumb_pip is not None and thumb_tip is not None:
                        if Distance(thumb_pip, thumb_tip) < 10:
                            previous_event = events[2]
                            isRelease = 0
                        else:
                            if previous_event == events[2]:
                                previous_event = events[3]
                                isRelease = 1

                    check_tic = 0

                
                if check_tic > 3:
                    previous_event = events[3]
                
                
                screen_pos = frame_pos2screen_pos(fsize, ssize, index_tipm)
                
                #MOVE MOUSE
                mouse.move(screen_pos[0], screen_pos[1])
                
                #CLICK CONTROL
                if check_tic == 0:
                    if previous_event == events[0] and isRelease == 1:
                        mouse.click()
                    elif previous_event == events[1] and isRelease == 1:
                        mouse.right_click()
                    elif previous_event == events[3] and isRelease == 1:
                        mouse.press()
                    elif isRelease == 0:
                        mouse.release()
                    print(previous_event) 
                check_tic += 1
                #TEXT OUTPUT
                cv2.putText(frame, previous_event, (800, 1000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        cv2.imshow("RemoteControl", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cam.release()
out.release()
cv2.destroyAllWindows()
