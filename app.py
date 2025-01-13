import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from keras.models import load_model
from PIL import Image

# Load your model and face cascade
model = load_model("model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Streamlit UI
st.title("Real-time Emotion and Face Recognition")
st.write("Detect emotions and identify faces from live camera feed")

# Option to start/stop camera
run = st.checkbox("Start Camera")

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Function to detect face and analyze emotion
def analyze_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        emotion_result = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)

        # Prepare face for model prediction
        face_img_resized = cv2.resize(face_img, (150, 150))
        img_array = np.expand_dims(face_img_resized, axis=0)
        pred = model.predict(img_array)

        # Determine name based on prediction
        name = "Hitesh" if pred[0][0] == 1 else "Aditya" if pred[0][0] == 0 else "None Match"

        return emotion_result[0]["dominant_emotion"], name

# Camera loop
if run:
    frame_placeholder = st.empty()  # Placeholder for video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image.")
            break

        emotion, name = analyze_face(frame)
        
        if emotion and name:
            # Display text on the frame
            cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Name: {name}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_placeholder.image(frame_image)

    cap.release()
