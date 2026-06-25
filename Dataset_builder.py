import os
import cv2
import numpy as np

from frame_extraction import extract_frames
from face_detection import FaceDetector

class DatasetBuilder:

    def __init__(self,
                 img_size=(299,299)):

        self.img_size = img_size
        self.detector = FaceDetector()

    def build(self,
              real_dir,
              fake_dir):

        X = []
        y = []

        for label, folder in enumerate(
            [real_dir, fake_dir]
        ):

            for file in os.listdir(folder):

                path = os.path.join(folder, file)

                frames = extract_frames(path)

                for frame in frames:

                    faces = self.detector.detect(frame)

                    if len(faces) == 0:
                        continue

                    x,y1,w,h = faces[0]

                    face = frame[y1:y1+h,
                                 x:x+w]

                    face = cv2.resize(
                        face,
                        self.img_size
                    )

                    face = face / 255.0

                    X.append(face)
                    y.append(label)

        return np.array(X), np.array(y)