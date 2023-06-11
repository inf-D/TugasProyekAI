import cv2
import numpy as np
from PIL import Image
from pathlib import Path

Path(f"trainer").mkdir(exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")


def getImagesAndLabels():
    faceSamples = []
    ids = []
    for filepath in Path("dataset").glob("*/*"):
        PIL_img = Image.open(filepath).convert("L")  # convert it to grayscale
        img_numpy = np.array(PIL_img, "uint8")

        id = int(str(filepath).split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for x, y, w, h in faces:
            faceSamples.append(img_numpy[y : y + h, x : x + w])
            ids.append(id)

    return faceSamples, ids


print("\nTraining dimulai\n")
faces, ids = getImagesAndLabels()
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write("trainer/trainer.yml")

# Print the numer of faces trained and end program
print("{0} wajah ditrain".format(len(np.unique(ids))))
