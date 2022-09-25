from tkinter import N
import cv2
# Ai libraries
from imageai.Detection import ObjectDetection
import os
import threading
import pyttsx3

execution_path = os.getcwd()


def speaks(audio) -> None:
    engine = pyttsx3.init()
    engine.say(audio)
    engine.runAndWait()
    engine.stop()


class VideoCamera(object):
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "yolo-tiny.h5"))
    detector.loadModel(detection_speed="flash")

    # buffer
    All_faces = [0, 0, 0, 0, 0, 0]

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # queue
    def second(self, name):
        self.All_faces[self.All_faces.index(name)] = 0

    # text-to-speech
    def speak(self, detections):
        for eachObject in detections:
            name = eachObject["name"]
            if name not in self.All_faces:
                if (self.All_faces[0] != 0):
                    for i in range(5, 0, -1):
                        self.All_faces[i] = self.All_faces[i-1]
                self.All_faces[0] = name

                timer = threading.Timer(20, self.second, args=[name])
                timer.start()

                speaks('I see '+name)

    # detection
    def get_frame(self, flag):

        if flag == 1:
            speaks('starting object detetction live stream.')
            self.video = cv2.VideoCapture(0)

        ret, frame = self.video.read()

        detected_image_array, detections = self.detector.detectObjectsFromImage(
            input_image=frame, input_type="array", output_type="array", minimum_percentage_probability=30)

        self.speak(detections)

        ret, jpeg = cv2.imencode('.jpg', detected_image_array)
        return jpeg.tobytes()

    def switching(self):
        if (self.video.isOpened()):
            speaks('switching to face recognition')
            self.video.release()

    # closing

    def close(self):
        speaks('stopping live stream.')
        self.video.release()
