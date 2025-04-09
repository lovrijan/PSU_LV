import matplotlib.image as mpimg
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

# Učitaj sliku
face = mpimg.imread('C:\\Users\\student\\Downloads\\example_grayscale.png')

if face.max() <= 1.0:
    face = (face * 255).astype(np.uint8)

X = face.reshape((-1, 1))

k_means = cluster.KMeans(n_clusters=25, n_init=1)
k_means.fit(X)

values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_

face_compressed = np.choose(labels, values.astype(np.uint8))
face_compressed.shape = face.shape

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(face, cmap='gray')
plt.title("Originalna slika")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(face_compressed, cmap='gray')
plt.title("Kvantizirana slika")
plt.axis('off')

plt.tight_layout()
plt.show()
