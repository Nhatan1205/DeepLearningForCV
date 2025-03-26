# Deep Learning for Computer Vision
## Outline:
1. Deep Learning
2. Neural Network
3. Convolutional NN
4. Image Classification
5. Object detection
6. Other topics (Segmentation)
----
## Deep learning: Tổng quan về deep learning
![image](https://github.com/user-attachments/assets/53a93692-5eb8-437f-a6d7-ad444250883f)

- Trong deeplearning, input là ảnh thì sẽ đưa cả ảnh vào model, không như ML phải có bước feature extraction.

![image](https://github.com/user-attachments/assets/590a926e-173a-48af-9214-2bc526f003dd)

- Sự khác biệt giữa mắt người và thị giác máy tính.
<details>
<summary>## 1. Điểm qua 1 vài tasks trong DL4CV </summary>
![image](https://github.com/user-attachments/assets/c4832c63-07e6-44ce-9950-924490244c1c)

+ Classification: cả 1 bước ảnh chỉ có 1 output (single object).
+ Detection: chỉ trong 1 bước ảnh có những đối tượng gì và ở đâu, thường sử dụng bounding box.
+ Segmentation: như detection nhưng không dùng bounding box mà phải viền đúng, khít nhất với từng object, hay nói cách khác là mô hình phải dự đoán được từng pixel của bức ảnh đó thuộc về label nào.
+ Others: text to image, ....
</details>
## 2. Định nghĩa về Neural Network và DL
![image](https://github.com/user-attachments/assets/23398554-7dc0-4732-a5e6-19e2b0095165)

+ NN: có 3 tầng input, middle và output layer. NN chỉ có 1 middle layer/ hidden layer
