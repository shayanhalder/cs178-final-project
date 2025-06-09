import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import utils.mnist_reader as mnist_reader

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train_reshaped = X_train.reshape(-1, 28, 28)
X_test_reshaped = X_test.reshape(-1, 28, 28)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Dataset Information:")
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Number of classes: {len(class_names)}")
print(f"Image dimensions: 28x28 pixels")
print(f"Total number of training samples: {len(X_train)}")
print(f"Total number of test samples: {len(X_test)}")


train_class_dist = Counter(y_train)
test_class_dist = Counter(y_test)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(class_names, [train_class_dist[i] for i in range(10)])
plt.title('Training Set Class Distribution')
plt.xticks(rotation=45)
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.bar(class_names, [test_class_dist[i] for i in range(10)])
plt.title('Test Set Class Distribution')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()


plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train_reshaped[i], cmap='gray')
    plt.title(f'Class: {class_names[y_train[i]]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_images.png')
plt.close()


plt.figure(figsize=(10, 6))
plt.hist(X_train.flatten(), bins=50, alpha=0.7)
plt.title('Distribution of Pixel Values')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.savefig('pixel_distribution.png')
plt.close()


print("\nSummary Statistics:")
print(f"Minimum pixel value: {X_train.min()}")
print(f"Maximum pixel value: {X_train.max()}")
print(f"Mean pixel value: {X_train.mean():.2f}")
print(f"Standard deviation: {X_train.std():.2f}")


print("\nClass-wise Statistics:")
for i in range(10):
    class_images = X_train[y_train == i]
    print(f"\n{class_names[i]}:")
    print(f"Number of samples: {len(class_images)}")
    print(f"Mean pixel value: {class_images.mean():.2f}")
    print(f"Standard deviation: {class_images.std():.2f}")


mean_images = np.array([X_train[y_train == i].mean(axis=0) for i in range(10)])
correlation_matrix = np.corrcoef(mean_images)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Correlation between Classes')
plt.tight_layout()
plt.savefig('class_correlation.png')
plt.close()

plt.figure(figsize=(15, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    mean_image = X_train_reshaped[y_train == i].mean(axis=0)
    plt.imshow(mean_image, cmap='gray')
    plt.title(f'Mean {class_names[i]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('mean_images.png')
plt.close() 