import cv2
import pickle
from pathlib import Path

SAMPLE = 30
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

with open("output/names.pkl", "rb") as f:
    names = pickle.load(f)

name = input("Masukkan nama: ")
Path(f"dataset/{name}").mkdir(exist_ok=True)
names.append(name)
id = names.index(name)

count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        cv2.imwrite(
            f"dataset/{name}/" + name + "." + str(id) + "." + str(count) + ".jpg",
            gray[y : y + h, x : x + w],
        )
        cv2.imshow("image", img)

    k = cv2.waitKey(100) & 0xFF  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= SAMPLE:
        break

with open("output/names.pkl", "wb") as f:
    pickle.dump(names, f)

cam.release()
cv2.destroyAllWindows()
