from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os

cascades_path = 'lib/haar_cascades'

detectorPaths = {
    "face": "haarcascade_frontalface_default.xml",
    "eyes": "haarcascade_eye.xml",
    # "smile": "haarcascade_upperbody.xml",
}
# initialize a dictionary to store our haar cascade detectors
print("[INFO] loading haar cascades...")

detectors = {}
# loop over our detector paths
for (name, path) in detectorPaths.items():
    # load the haar cascade from disk and store it in the detectors
    # dictionary
    path = os.path.sep.join([cascades_path, path])
    detectors[name] = cv2.CascadeClassifier(path)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)

# loop over the frames from the video stream
while cap.isOpened():
    # grab the frame from the video stream, resize it, and convert it
    # to grayscale
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    frame = imutils.resize(frame, width=500)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # perform face detection using the appropriate haar cascade
    faceRects = detectors["face"].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # loop over the face bounding boxes
    for (fX, fY, fW, fH) in faceRects:
        # extract the face ROI
        faceROI = gray[fY:fY + fH, fX:fX + fW]
        # apply eyes detection to the face ROI
        eyeRects = detectors["eyes"].detectMultiScale(faceROI, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

        # loop over the eye bounding boxes
        for (eX, eY, eW, eH) in eyeRects:
            # draw the eye bounding box
            ptA = (fX + eX, fY + eY)
            ptB = (fX + eX + eW, fY + eY + eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)

        # draw the face bounding box on the frame
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()