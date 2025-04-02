# Deep Learning for Computer Vision
## Outline:
- [x] 1. Deep Learning
- [x] 2. Neural Network
- [ ] 3. Convolutional NN
- [ ] 4. Image Classification
- [ ] 5. Object detection
- [ ] 6. Other topics (Segmentation)
----
## Neural Network
<details>
  <summary><b> 1. Biological Neuron vs Artificial Neuron</b></summary>
  
  ![image](https://github.com/user-attachments/assets/d1bbba13-2a36-4a74-abc4-963b076b0823)

  + Ý nghĩa của NN là mô phỏng lại não bộ để máy móc có thể xử lý con người.
  + Tác dụng của hàm `activation function` là quyết định xem có bao lượng dữ liệu sẽ được đi qua. (sự khác biệt của neuron nhân tạo). 
  + **`Neuron = linear classifier + activation function`**
  + `w` và `b` là 2 tham số có thể chỉnh sửa sau quá trình huấn luyện.
  + Activation function: sẽ giới thiệu thêm tính chất phi tuyến tính vào mô hình. Ép miền dữ liệu nhỏ hơn/ khác đi.
  From linear classifier to neuron: 

  ![image](https://github.com/user-attachments/assets/19b1bdfe-8869-471f-9e26-d2b03e1f0d43)

</details>

----

<details>
  <summary><b> 2. Activation function</b></summary>

  ![image](https://github.com/user-attachments/assets/7669db47-f2a5-4063-b7e6-165bc45fe5e3)

  + **Sigmoid**: từ input đưa ra output có giá trị từ `0 => 1`. 10 năm trở lại đây, hàm này không còn sử dụng nữa vì gây ra `vanishing gradient`. Bởi vì khi chúng ta lấy đạo hàm của hàm này. Giá trị chỉ nằm trong phạm vi `0 => 0.25`.
  + **Tanh**: giá trị nằm từ `-1` đến `1`.
  + **ReLU**: input < `0` thì output = `0`, input = `z `thì output = `z`.  `RELU` là hàm không khả vi tại điểm `x = 0`. Nhưng người ta vẫn sử dụng vì tỉ lệ output = `0` rất khó.
  + **Leaky ReLU** : input < `0` thì gán output = `ε`*`z`
</details>

-----

<details>
  <summary><b> 3. Neural Network Architecture</b></summary>

  ![image](https://github.com/user-attachments/assets/26a8b0b4-6668-4557-a9f3-6799ffcb3e6b)

  + Trong NN, output của 1 neuron thì có thể trở thành input cho neuron khác. NN chỉ có 1 chiều, không có vòng lặp.
  + **Fully connected layer**: các `neuron` trong cùng 1 layer không kết nối lẫn nhau thì kết nói từng đôi 1 một với `neuron` của layer khác.
  + NN có 3 layers thì bao gồm 2 `hidden layers` và 1 `output layer`, không đề cập đến `input`.

  **❓: Vì sao NN lại phải có Activation function ?** : bởi vì, ta biết trong một NN có nhiều layers `y = ax + b` , `z = cy + d`, `t = ez + f`, .. nếu không có hàm `activation` thì người ta có thể gộp các hàm tuyến tính thành 1 hàm chung, như vậy việc phân tầng không còn ý nghĩa gì. Vậy `activation function` có 2 ý nghĩa, một là gúp việc phân tầng trở nên có giá trị, hai là giúp cho mô hình được học tính chất phi tuyến tính, từ đó học được nhiều tính chất hơn từ dữ liệu.

  + Quá trình mà một mô hình NN gửi input từ input layer qua các hidden layers và cuối cùng đến output layer gọi là `Forward propagation`.

  ![image](https://github.com/user-attachments/assets/8263b7ca-55f9-438a-b125-c335f414e647)

  https://ml4a.github.io/demos/simple_forward_pass/

  + Complete example of Neural Network

  ![image](https://github.com/user-attachments/assets/54a463a6-e553-4727-90a5-ad0f7be9c90f)

  https://ml4a.github.io/demos/forward_pass_mnist/

  ![image](https://github.com/user-attachments/assets/8d00327a-f4bb-424e-9db8-516a7db5daeb)

  **❓: Vì sao NN lại phải có Hidden layers ?**: 
</details>

-----

<details>
  <summary><b> 4. Loss function</b></summary>

   ![image](https://github.com/user-attachments/assets/cff0afd3-91d0-4aef-a5a7-bc5327c00a3e)

   

</details>
