import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
model = load_model("model.h5")
labels = np.load("labels.npy")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    features = []

    if results.face_landmarks:
        for i in results.face_landmarks.landmark:
            features.append(i.x - results.face_landmarks.landmark[1].x)
            features.append(i.y - results.face_landmarks.landmark[1].y)
        
        if results.left_hand_landmarks:
            for i in results.left_hand_landmarks.landmark:
                features.append(i.x - results.left_hand_landmarks.landmark[8].x)
                features.append(i.y - results.left_hand_landmarks.landmark[8].y)
        else:
            features.extend([0.0] * 42)

        if results.right_hand_landmarks:
            for i in results.right_hand_landmarks.landmark:
                features.append(i.x - results.right_hand_landmarks.landmark[8].x)
                features.append(i.y - results.right_hand_landmarks.landmark[8].y)
        else:
            features.extend([0.0] * 42)

        features = np.array(features).reshape(1, -1)

        prediction = labels[np.argmax(model.predict(features))]
        cv2.putText(frame, f"Emotion: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Recommend songs
        print(f"Detected emotion: {prediction}")
        lang = input("Enter language: ")
        singer = input("Enter singer: ")
        query = f"https://www.youtube.com/results?search_query={lang}+{prediction}+song+{singer}"
        webbrowser.open(query)
        break

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
