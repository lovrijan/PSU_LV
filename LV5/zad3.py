from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
X = np.random.rand(100, 2)
y = np.random.randint(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Velicina skupa za ucenje:", X_train.shape)
print("Velicina skupa za testiranje:", X_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.matshow(cm, cmap='Blues')
plt.colorbar()
plt.title('Matrica zabune')
plt.xlabel('Predviđene klase')
plt.ylabel('Stvarne klase')

plt.xticks([0, 1], ['Klasa 0', 'Klasa 1'])
plt.yticks([0, 1], ['Klasa 0', 'Klasa 1'])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.show()

accuracy = accuracy_score(y_test, y_pred)
print("Točnost:", accuracy)

precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)

print("Preciznost po klasama:", precision)
print("Odziv po klasama:", recall)

plt.figure(figsize=(20, 15))
plot_tree(clf, filled=True)
plt.title('Vizualizacija stabla odlučivanja')
plt.show()