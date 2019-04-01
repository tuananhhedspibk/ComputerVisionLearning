#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("./haarcascade_eye.xml")

# Define the function that will do the detections
def detect(gray, frame):
    # frame is original image
    # get coordinate, width, height of rectangle that detect face
    # Cascade work on gray image
    # 1.3: reduce image 1.3 times
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # roi: region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)