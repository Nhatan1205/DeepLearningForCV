import numpy as np
from matplotlib import pyplot as plt

file_path = "D:/VietnguyenAI/Deep Learning/PythonProject/Exercises/full_numpy_bitmap_butterfly.npy"
images = np.load(file_path).astype(np.float32)
# print(images.shape)
# train_images = images[:-10]
# test_images = images[-10:]

# avg_image = np.mean(images, 0)
# reshaped_avg_image = avg_image.reshape(28, 28)

# plt.imshow(reshaped_avg_image)
# plt.show()

# index = 4
# test_images = test_images[index]

# # score = np.dot(avg_image, test_images)
# score = avg_image @ test_images
# print(score)

test_image = images[100]

categories = ['butterfly', 'apple', 'book', 'crab', 'eye']
# categories = ['butterfly']
scores = []
weight = []
for category in categories:
    file_path = f"D:/VietnguyenAI/Deep Learning/PythonProject/Exercises/full_numpy_bitmap_{category}.npy"
    images = np.load(file_path).astype(np.float32)
    avg_image = np.mean(images, axis=0)
    weight.append(avg_image)
    scores.append(avg_image @ test_image)

print(scores)

print(f'the test_image is most likely {categories[np.argmax(scores)]}')

plt.figure(figsize=(10, 4))
for i in range(len(weight)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(weight[i].reshape(28, 28))
    plt.axis('off')
    plt.title(categories[i])
plt.show()