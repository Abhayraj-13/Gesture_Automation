# import mediapipe as mp
# import cv2
# import numpy as np

# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

# cap = cv2.VideoCapture(0)

# with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = cv2.flip(image, 1)
#         image.flags.writeable = False
#         results = hands.process(image)
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
#                                           mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
#                                           mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
#                                           )

#         cv2.imshow("Hand Tracking", image)
#         if cv2.waitKey(10) & 0xFF == ord("q"):
#             break

# cap.release()
# cv2.destroyAllWindows()
import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Function to move the cursor to specified coordinates
def move_cursor(x, y):
    screen_size = pyautogui.size()
    target_x = int(x * screen_size.width)
    target_y = int(y * screen_size.height)
    pyautogui.moveTo(target_x, target_y)

# Function to perform a click event
def perform_click():
    pyautogui.click()

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the thumb tip (point 4)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_tip_x = thumb_tip.x
                thumb_tip_y = thumb_tip.y

                # Get the coordinates of the index finger tip (point 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_x = index_finger_tip.x
                index_finger_tip_y = index_finger_tip.y

                # Connect the index finger tip with the cursor
                move_cursor(index_finger_tip_x, index_finger_tip_y)

                # Calculate the distance between thumb tip and index finger tip
                distance = math.sqrt((thumb_tip_x - index_finger_tip_x)**2 + (thumb_tip_y - index_finger_tip_y)**2)

                # Check if thumb tip and index finger tip are close to each other for clicking
                if distance < 0.1:  # Adjust this threshold as needed
                    perform_click()

                # Draw landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
                                          )

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
