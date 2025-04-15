import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns

# Učitavanje MNIST skupa podataka
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 1) Prikaz nekoliko slika iz MNIST skupa podataka
plt.figure(figsize=(10, 4))  # Postavljanje veličine figure
for i in range(10):
    plt.subplot(2, 5, i + 1) 
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Broj: {y_train[i]}")
    plt.axis('off') 
plt.tight_layout()
plt.show()

# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float64") / 255
x_test_s = x_test.astype("float64") / 255

# Pretvaranje slika 28x28 u vektore duljine 784
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiranje labela (0-9) u one-hot encoding
y_train_s = keras.utils.to_categorical(y_train, num_classes= 10)
y_test_s = keras.utils.to_categorical(y_test, num_classes= 10)


# 2) Izgradnja potpuno povezane neuronske mreže
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),  # Prvi skriveni sloj
    Dense(64, activation='relu'),                      # Drugi skriveni sloj
    Dense(10, activation='softmax')                    # Izlazni sloj
])
model.summary()

# Definiranje procesa učenja
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treniranje mreže
model.fit(x_train_s, y_train_s, epochs=10, batch_size=32)

# 3) Evaluacija mreže
train_loss, train_accuracy = model.evaluate(x_train_s, y_train_s, verbose=0)
test_loss, test_accuracy = model.evaluate(x_test_s, y_test_s, verbose=0)
print(f"Točnost na skupu podataka za učenje: {train_accuracy * 100:.2f}%")
print(f"Točnost na skupu podataka za testiranje: {test_accuracy * 100:.2f}%")

# 4) Prikaz matrice zabune
# Predikcije za testni skup
y_pred_test = np.argmax(model.predict(x_test_s), axis=1)
y_true_test = np.argmax(y_test_s, axis=1)

# Matrica zabune za testni skup
conf_matrix_test = confusion_matrix(y_true_test, y_pred_test)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.title("Matrica zabune - Testni skup")
plt.xlabel("Predviđene klase")
plt.ylabel("Stvarne klase")
plt.show()

# Predikcije za skup podataka za učenje
y_pred_train = np.argmax(model.predict(x_train_s), axis=1)
y_true_train = np.argmax(y_train_s, axis=1)

# Matrica zabune za skup podataka za učenje
conf_matrix_train = confusion_matrix(y_true_train, y_pred_train)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Greens')
plt.title("Matrica zabune - Skup za učenje")
plt.xlabel("Predviđene klase")
plt.ylabel("Stvarne klase")
plt.show()
