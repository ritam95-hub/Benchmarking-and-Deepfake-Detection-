{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e41e5fe7",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "\n",
    "class FaceDetector:\n",
    "\n",
    "    def __init__(self):\n",
    "        self.detector = cv2.CascadeClassifier(\n",
    "            cv2.data.haarcascades +\n",
    "            \"haarcascade_frontalface_default.xml\"\n",
    "        )\n",
    "\n",
    "    def detect(self, frame):\n",
    "\n",
    "        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
    "\n",
    "        faces = self.detector.detectMultiScale(\n",
    "            gray,\n",
    "            scaleFactor=1.1,\n",
    "            minNeighbors=5\n",
    "        )\n",
    "\n",
    "        return faces"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
