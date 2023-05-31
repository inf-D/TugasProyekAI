from pathlib import Path
import cv2
import pickle
import face_recognition
import numpy
from collections import Counter

encodings_location = Path("output/encodings.pkl")
with encodings_location.open(mode="rb") as f:
    loaded_encodings = pickle.load(f)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("unable to open the Camera ")
    exit()


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
    )
    if votes:
        return votes.most_common(1)[0][0]


while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = numpy.ascontiguousarray(small_frame[:, :, ::-1])
    input_image = rgb_small_frame

    input_face_locations = face_recognition.face_locations(input_image, model="hog")
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"

        top, right, bottom, left = bounding_box
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(
            frame, (left, bottom + 35), (right, bottom), (0, 0, 255), cv2.FILLED
        )
        cv2.putText(
            frame,
            name,
            (left + 6, bottom + 19),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
