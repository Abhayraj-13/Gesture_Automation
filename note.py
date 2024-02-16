import mediapipe as mp
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Constants for text and drawing
FONT_SIZE = 20
TEXT_COLOR = (255, 255, 255)
BG_COLOR = (0, 0, 0)
DEFAULT_PEN_WIDTH = 2
DEFAULT_PEN_COLOR = (255, 255, 255)  # White

# Create a blank canvas (white background)
canvas = Image.new("RGB", (640, 480), BG_COLOR)
draw = ImageDraw.Draw(canvas)
font = ImageFont.truetype("arial.ttf", FONT_SIZE)

# Initialize variables for drawing
prev_x, prev_y = None, None
erasing = False
pen_width = DEFAULT_PEN_WIDTH
pen_color = DEFAULT_PEN_COLOR

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
                # Get the coordinates of the index finger tip (point 8)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_x = int(index_finger_tip.x * image.shape[1])
                index_finger_tip_y = int(index_finger_tip.y * image.shape[0])

                # Get the coordinates of the thumb tip (point 4)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_tip_x = int(thumb_tip.x * image.shape[1])
                thumb_tip_y = int(thumb_tip.y * image.shape[0])

                # Erase if thumb tip is detected
                if thumb_tip_x and thumb_tip_y:
                    erasing = True
                    canvas.paste((0, 0, 0), (thumb_tip_x - 10, thumb_tip_y - 10, thumb_tip_x + 10, thumb_tip_y + 10))

                # Change pen width if middle finger tip is detected
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_finger_tip_y = int(middle_finger_tip.y * image.shape[0])
                if middle_finger_tip_y:
                    pen_width = max(1, min(10, int(10 * (1 - middle_finger_tip_y))))

                # Draw on canvas if previous point exists and not erasing
                if prev_x is not None and prev_y is not None and not erasing:
                    draw.line([(prev_x, prev_y), (index_finger_tip_x, index_finger_tip_y)],
                              fill=pen_color, width=pen_width)

                # Update previous point
                prev_x, prev_y = index_finger_tip_x, index_finger_tip_y

        # Convert canvas to OpenCV format and display
        canvas_cv = np.array(canvas)
        image = cv2.addWeighted(image, 0.5, canvas_cv, 0.5, 0)

        cv2.imshow("Finger Writing", image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
