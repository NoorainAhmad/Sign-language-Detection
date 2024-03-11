import matplotlib.pyplot as plt

DATA_DIR = './project'
number_of_classes = 26
dataset_size = 30
samples_per_class=[]
class_labels = list(range(number_of_classes))
samples_per_class.append(dataset_size)

plt.figure(figsize=(10, 6))
plt.bar(class_labels, samples_per_class, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Bar Graph of Number of Samples Collected for Each Class A-Z')
plt.show()