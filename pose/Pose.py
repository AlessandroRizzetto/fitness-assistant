import mediapipe as mp
import cv2 
import numpy as np

#variables for mediapipe to visualize and estimate pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a) #First
    b = np.array(b) #Mid
    c = np.array(c) #End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    
    return angle

cap = cv2.VideoCapture(0)

#Curl counter variables
left_counter = 0
right_counter = 0
left_stage = None  # None = not curling, 1 = curling
right_stage = None  # None = not curling, 1 = curling


# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        #Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        #Make detection
        results = pose.process(image)
        
        #Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            #Get coordinates
            # 11 - 13 - 15 = Angle of the left arm
            leftShoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            leftElbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            leftWrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # 12 - 14 - 16 = Angle of the right arm
            rightShoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            rightElbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            rightWrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            
            
            #Calculate Angles
            leftElbowAngle = calculate_angle(leftShoulder, leftElbow, leftWrist)
            rightElbowAngle = calculate_angle(rightShoulder, rightElbow, rightWrist)
            
            # Visualize leftElbowAngle
            cv2.putText(image, str(leftElbowAngle),
                        tuple(np.multiply(leftElbow, [640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )
            # Visualize rightElbowAngle
            cv2.putText(image, str(rightElbowAngle),
                        tuple(np.multiply(rightElbow, [640,480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )
            #print(landmarks)
            
            # Curl counter logic for left arm
            if leftElbowAngle > 160:
                left_stage = "down"
            if leftElbowAngle < 30 and left_stage == 'down':
                left_stage = "up"
                left_counter += 1
                print("Left Arm Counter:", left_counter)

            # Curl counter logic for right arm
            if rightElbowAngle > 160:
                right_stage = "down"
            if rightElbowAngle < 30 and right_stage == 'down':
                right_stage = "up"
                right_counter += 1
                print("Right Arm Counter:", right_counter)
            
        except:
            pass
        
        # #Render curl counter
        # #Setup status box
        # cv2.rectangle(image, (0,0), (270,73), (245,117,16), -1)
        
        # # Rep data for left arm
        # cv2.putText(image, 'LEFT ARM', (15, 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, 'Reps: ' + str(left_counter),
        #             (10, 70),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        # cv2.putText(image, 'Stage: ' + str(left_stage),
        #             (10, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # # Setup right arm status box
        # cv2.rectangle(image, (350,0), (620,100), (245,117,16), -1)
        # # Rep data for right arm
        # cv2.putText(image, 'RIGHT ARM', (365, 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        # cv2.putText(image, 'Reps: ' + str(right_counter),
        #             (360, 70),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
        # cv2.putText(image, 'Stage: ' + str(right_stage),
        #             (360, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
        # Render curl counters
        # Setup status box
        cv2.rectangle(image, (10, 400), (210, 480), (245,117,16), -1)

        # Left arm data
        cv2.putText(image, 'LEFT ARM', (20, 430),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(left_counter),
                    (50, 455),
                    cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Stage: ' + str(left_stage),
                    (20, 470),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Right arm data
        cv2.putText(image, 'RIGHT ARM', (120, 430),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(right_counter),
                    (150, 455),
                    cv2.FONT_HERSHEY_PLAIN, 1.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Stage: ' + str(right_stage),
                    (120, 470),
                    cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
        
        
        cv2.imshow('Mediapipe Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()