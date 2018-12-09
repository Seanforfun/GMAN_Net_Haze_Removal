from PIL import  Image
import os
import numpy as np

for image in os.listdir("./ClearImages/TrainingImages"):
    shape = np.shape(Image.open(os.path.join("./ClearImages/TrainingImages", image)))
    if shape[0] < 224 or shape[1] < 224:
        print(image)

