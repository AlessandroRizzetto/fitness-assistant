import cv2
import numpy as np
import mediapipe as mp

# Function to calculate the angle between three points


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)

# Variables for elbow angle counting
left_counter = 0
right_counter = 0
left_stage = None  # None = not curling, 1 = curling
right_stage = None  # None = not curling, 1 = curling

# Setup MediaPipe Holistic instance
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setup drawing utility
mp_drawing = mp.solutions.drawing_utils

with holistic as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe Holistic detection
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract elbow landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            right_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
            left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
            right_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
            left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Calculate angles
            left_elbow_angle = calculate_angle(
                left_shoulder, left_elbow, left_wrist)
            right_elbow_angle = calculate_angle(
                right_shoulder, right_elbow, right_wrist)

            # Visualize angles
            cv2.putText(image, str(left_elbow_angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, str(right_elbow_angle), tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            # Curl counter logic for left elbow
            if left_elbow_angle > 160:
                left_stage = "down"

            if left_elbow_angle < 30 and left_stage == 'down':
                left_stage = "up"
                left_counter += 1
                print("Left Elbow Counter:", left_counter)

            # Curl counter logic for right elbow
            if right_elbow_angle > 160:
                right_stage = "down"

            if right_elbow_angle < 30 and right_stage == 'down':
                right_stage = "up"
                right_counter += 1
                print("Right Elbow Counter:", right_counter)

        except:
            pass

        # Render counters
        cv2.rectangle(image, (10, 400), (210, 480), (245, 117, 16), -1)

        cv2.putText(image, 'LEFT ARM', (20, 430),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(left_counter), (50, 455),
                    cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Stage: ' + str(left_stage), (20, 470),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, 'RIGHT ARM', (120, 430),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(right_counter), (150, 455),
                    cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Stage: ' + str(right_stage), (120, 470),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Render landmarks
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )

        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('MediaPipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
