import pickle
from pathlib import Path

Path("output").mkdir(exist_ok=True)

# membuat file pickle untuk menampung nama
names = []
filename = "output/names.pkl"
f = open(filename, "wb")
pickle.dump(names, f)
f.close()
