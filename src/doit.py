from pathlib import Path

import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("TkAgg")

img_path = Path("../data/sample_data/IMG_0119.PNG")
pil = Image.open(img_path)
im = np.asarray(pil)
plt.imshow(im)

pts = plt.ginput(4)  # number of clicks
with open("points.pkl", "wb") as f:
    pickle.dump(pts, f, pickle.HIGHEST_PROTOCOL)
print(pts)
