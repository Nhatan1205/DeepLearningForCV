# Deep Learning for Computer Vision
## Outline:
- [x] 1. Deep Learning
- [ ] 2. Neural Network
- [ ] 3. Convolutional NN
- [ ] 4. Image Classification
- [ ] 5. Object detection
- [ ] 6. Other topics (Segmentation)
----
## Deep learning: Tổng quan về deep learning
![image](https://github.com/user-attachments/assets/53a93692-5eb8-437f-a6d7-ad444250883f)

- Trong deeplearning, input là ảnh thì sẽ đưa cả ảnh vào model, không như ML phải có bước feature extraction.

![image](https://github.com/user-attachments/assets/590a926e-173a-48af-9214-2bc526f003dd)

_(Sự khác biệt giữa mắt người và thị giác máy tính.)_

-----
<details>
  <summary><b> 1. Điểm qua 1 vài tasks trong DL4CV</b></summary>
  
  ![image](https://github.com/user-attachments/assets/a39291ff-f8ae-471f-856d-e1b73f48d9fb)


  + **Classification**: cả 1 bước ảnh chỉ có 1 output (single object).
  + **Detection**: chỉ trong 1 bước ảnh có những đối tượng gì và ở đâu, thường sử dụng bounding box.
  + **Segmentation**: như detection nhưng không dùng bounding box mà phải viền đúng, khít nhất với từng object, hay nói cách khác là mô hình phải dự đoán được từng pixel của bức ảnh đó thuộc về label nào.
  + **Others**: text to image, ....
</details>

----

<details>
  <summary><b>2. Định nghĩa về Neural Network và DL</b></summary>

  ![image](https://github.com/user-attachments/assets/23398554-7dc0-4732-a5e6-19e2b0095165)

  + **NN**: có 3 tầng input, middle và output layer. NN chỉ có 1 middle layer/ hidden layer. Hay chỉ về architecture.
  + **DL**: giống NN nhưng có nhiều hidden layers. Hay chỉ về 1 area, lĩnh vực. 
</details>

----

<details>
  <summary><b>3. Điểm qua 1 chút về Image Classification</b></summary>

  ![image](https://github.com/user-attachments/assets/3bf53e6f-6173-4034-b64b-2c0398fe7a0f)

  Ảnh được tạo ra từ pixel (có giá trị từ `0`(đen) -> `255`(trắng))
  + Ảnh **đen trắng** (lưu trữ bằng mảng `2` chiều bằng kích thước bước ảnh) vs **ảnh màu** (dùng mảng `3` chiều: số hàng, số cột, số lượng kênh màu`(R, G, B)`)
  + Khi nói về kích thước 1 bức ảnh, phải nói về chiều dọc trước rồi chiều ngang sau. (số hàng trước số cột).
    
  **❓: Có ảnh nào có 4 kênh màu không?** : ảnh transparent (png), kênh cuối là kênh transparent. 
  
</details>

-----

<details>
  <summary><b>4. Điểm qua 1 chút về Dataset</b></summary>

  ![image](https://github.com/user-attachments/assets/613a07e1-8ecf-46a1-ab9c-c076d56857de)

  **❓: Bạn có biết trong thực tế, người ta dùng công cụ nào để đánh nhãn ảnh?** : `cvat`.

</details>

----

<details>
  <summary><b>5. Linear Classifier - score function</b></summary>

  ![image](https://github.com/user-attachments/assets/a2ba7b10-6966-4556-b20b-cd22d1c8fa70)

  Như phần trước, nhiệm vụ của chúng ta là, khi có 1 bức ảnh với 1 đống pixels có giá trị từ 0 đến 255, ta phải làm sao từ 1 cái mảng 2 chiều => output có `n` phần tử, với `n` là số lượng class. Việc mapping từ input là 1 mảng 2 chiều thành output là  vector 1 chiều như này => gọi là **score function**.

  + Nó dùng phép  `element-wise multiplication` để tính toán với hàm `f` trong hình: `$f(x_i, W, b) = W \times x_i + b$` , $x_i$ là pixel
  + Cat score: $56 \times 0.2 + 231 \times (-0.5) + 24 \times 0.1 + 2 \times 2.0 + b = -96.8$

</details>

----

<details>
  <summary><b>6. Linear Classifier explanation</b></summary>

  ![image](https://github.com/user-attachments/assets/dca02f2b-6dbe-4789-b191-5de857373dbc)

 **`MNIST`**- model chữ số viết tay
  + Input = `728` pixels (28x28)
  + Output: `10` classes (từ `0` => `9`)
  + Weight visualization: class `0`.

    ![image](https://github.com/user-attachments/assets/d2e1f39e-70ad-4c66-ac3e-403acd3587f9)

    Ta muốn pixel nào mà có số `0` đi qua thì `w` tương ứng của nó phải lớn. Vì giá trị pixel được đi qua (gần `255`) nhân với `w` thì kết quả sẽ lớn. Các vị trí khác `w`càng bé càng tốt. 
    
    ![image](https://github.com/user-attachments/assets/978ea44a-247c-4dc0-9c26-d8ff1fe7a6ab)

    Trọng số của class nào thì sẽ biểu diễn gần giống giá trị class đó, có thể coi là ảnh trung bình của ảnh số `0`. 


    **`CIFAR`**

    ![image](https://github.com/user-attachments/assets/ad75e67a-ddb2-4b98-a60f-0873530466fd)

    Dữ liệu khó hơn do tư thế của object trong ảnh khác nhau, nhưng vẫn có 1 vài điểm đặc trưng như `ship` thì hay có background `biển`

</details>

----
