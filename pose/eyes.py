import time
import cv2
import mediapipe as mp


def detect_blink_duration():
    cap = cv2.VideoCapture(0)

    # Variables for eye detection
    eye_closed = False
    eye_closed_start_time = None
    eye_closed_duration = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Detect eye landmarks
            if results.face_landmarks is not None:
                left_eye_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark[mp_holistic.FaceLandmark.LEFT_EYE]]
                right_eye_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in results.face_landmarks.landmark[mp_holistic.FaceLandmark.RIGHT_EYE]]

                # Calculate horizontal distance between left eye landmarks
                left_eye_openness = left_eye_landmarks[3][0] - \
                    left_eye_landmarks[0][0]

                # Calculate horizontal distance between right eye landmarks
                right_eye_openness = right_eye_landmarks[0][0] - \
                    right_eye_landmarks[3][0]

                # Check if eyes are closed
                if left_eye_openness < 0.2 and right_eye_openness < 0.2:
                    if not eye_closed:
                        eye_closed_start_time = time.time()
                    eye_closed = True
                else:
                    if eye_closed:
                        eye_closed_duration = time.time() - eye_closed_start_time
                        print("Eye Closed Duration:", eye_closed_duration)
                    eye_closed = False

            # Visualize face landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

            cv2.imshow('Holistic', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return eye_closed, eye_closed_duration
