import face_recognition
import pickle
from pathlib import Path

names = []
encodings = []
model = "hog"
encodings_location = Path("output/encodings.pkl")

for filepath in Path("training").glob("*/*"):
    name = filepath.parent.name
    image = face_recognition.load_image_file(filepath)

    face_locations = face_recognition.face_locations(image, model=model)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for encoding in face_encodings:
        names.append(name)
        encodings.append(encoding)

name_encodings = {"names": names, "encodings": encodings}
with encodings_location.open(mode="wb") as f:
    pickle.dump(name_encodings, f)
