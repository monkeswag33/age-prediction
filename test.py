from model import model, checkpoint_path
from train import preprocessing
import sys, matplotlib.pyplot as plt

model.load_weights(checkpoint_path)
image_path = sys.argv[1]
img = preprocessing([image_path])
plt.imshow(img[0], cmap="gray", interpolation="nearest")
plt.show()
img = img / 255.0
pred = model.predict(img)
print(pred)