import mediapipe as mp
import cv2
import numpy as np
import pyautogui
import math

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Function to move the cursor to specified coordinates
def move_cursor(x, y):
    screen_size = pyautogui.size()
    target_x = int(x * screen_size.width)
    target_y = int(y * screen_size.height)
    pyautogui.moveTo(target_x, target_y)

# Function to perform a click event
def perform_click():
    pyautogui.click()

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

cap = cv2.VideoCapture(0)
prev_left_eye_x, prev_left_eye_y = None, None
is_navigation_active = False

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract the landmarks for the left eye (labeled 133)
                left_eye_x = face_landmarks.landmark[133].x
                left_eye_y = face_landmarks.landmark[133].y

                # Calculate movement direction based on change in position of left eye
                if prev_left_eye_x is not None and prev_left_eye_y is not None:
                    dx = left_eye_x - prev_left_eye_x
                    dy = left_eye_y - prev_left_eye_y

                    # Check if the left eye blinks (disappears)
                    if left_eye_x == 0.0 and left_eye_y == 0.0:
                        is_navigation_active = False
                    else:
                        is_navigation_active = True
                        # Calculate the distance between consecutive frames
                        distance = calculate_distance(prev_left_eye_x, prev_left_eye_y, left_eye_x, left_eye_y)
                        # Normalize the direction vector
                        dx /= distance
                        dy /= distance
                        # Move the cursor based on the direction of eye movement
                        move_cursor(dx, dy)

                # Save current eye position for the next frame
                prev_left_eye_x, prev_left_eye_y = left_eye_x, left_eye_y

                # Draw landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2)
                )

        # If navigation is not active, reset previous eye position
        if not is_navigation_active:
            prev_left_eye_x, prev_left_eye_y = None, None

        cv2.imshow("Face Mesh", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
