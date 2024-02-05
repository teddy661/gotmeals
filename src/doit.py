import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
matplotlib.use('TkAgg')

pil = Image.open("IMG_0118.PNG")
im = np.asarray(pil)
plt.imshow(im)

pts = plt.ginput(10) #number of clicks
print(pts)
