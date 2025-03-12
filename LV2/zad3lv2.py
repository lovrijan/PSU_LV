import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("C:\\Users\\student\\Downloads\\tiger.png")
img = img[:,:,0].copy()

print(img.shape)
print(img.dtype)
plt.figure()
plt.imshow(img, cmap="gray", )
plt.imshow(np.rot90(img), cmap="gray", )
plt.imshow(np.fliplr(img), cmap="gray", )
plt.show()
