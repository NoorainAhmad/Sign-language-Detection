import pickle
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./project.pickle', 'rb'))

data = np.asarray(data_dict['project'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

accuracy = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(accuracy * 100))


print('{}% of samples were classified correctly !'.format(score * 100))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Plot Accuracy
ax[0].bar(['Accuracy'], [accuracy], color='skyblue')
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')

# Plot Test Set Distribution
ax[1].hist(y_test, bins=len(np.unique(labels)), color='salmon', alpha=0.7)
ax[1].set_title('Test Set Distribution')
ax[1].set_xlabel('Classes')
ax[1].set_ylabel('Count')

# Plot Train Set Distribution
ax[2].hist(y_train, bins=len(np.unique(labels)), color='lightgreen', alpha=0.7)
ax[2].set_title('Train Set Distribution')
ax[2].set_xlabel('Classes')
ax[2].set_ylabel('Count')

plt.tight_layout()
plt.show()

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()