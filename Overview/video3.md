# Deep Learning for Computer Vision
## Outline:
- [x] 1. Deep Learning
- [x] 2. Neural Network
- [ ] 3. Convolutional NN
- [ ] 4. Image Classification
- [ ] 5. Object detection
- [ ] 6. Other topics (Segmentation)
----
## Neural Network/ Gradient
<details>
  <summary><b> 1. Gradient</b></summary>

  ![image](https://github.com/user-attachments/assets/bbd434b4-704c-469d-9052-b958208a23fc)

  `Gradient` dịch ra tiếng việt là độ dốc. 
  + Ôn lại bài, cực trị, cực tiểu, cực đại. Sự khác biệt của cực tiểu/ đại so với gtln/ gtnn là cực tiểu/ đại là giá trị local thôi còn gtln/gtnn là toàn cục.
  + Ý tưởng của gradient là tìm hướng nào mà loss giảm nhanh nhất.

  ![image](https://github.com/user-attachments/assets/5b2333c1-6100-4091-9f1f-4d8efa8bb42d)

  + Vấn đề thực tế:

    ![image](https://github.com/user-attachments/assets/f209caec-ada9-4ef5-a1dd-5bd1cdc33c97)


    ## 🔹 Hình bên trái: Gradient Descent trong Linear Classifier

    Đây là một ví dụ của **hàm cost đơn giản**, có hình dạng **bát parabol**.
    
    - Trục hoành là các tham số `m` và `b`, còn trục tung là **giá trị hàm mất mát (cost)**.
    - Đường mũi tên mô tả quá trình **Gradient Descent** – một thuật toán tối ưu dốc nhất để tìm điểm thấp nhất của hàm (minimum).
    - Do hàm này **lõm (convex)**, nên Gradient Descent **luôn hội tụ** về điểm tối ưu toàn cục (_global minimum_), bất kể điểm bắt đầu.
    
    👉 **Linear classifier** thường dẫn đến những hàm cost dạng này nên việc tối ưu **đơn giản và hiệu quả**.
    
    ---
    
    ## 🔹 Hình bên phải: Gradient Descent trong Neural Network
    
    Đây là một **hàm mất mát phức tạp**, với nhiều **đỉnh (_maxima_)** và **đáy (_minima_)** – gọi là **non-convex function**.
    
    - Bề mặt gồ ghề cho thấy có **nhiều local minima** và **saddle points**.
    - Gradient Descent trong trường hợp này có thể:
      - Bị mắc kẹt trong một **local minimum** (tối ưu cục bộ),
      - Hoặc chậm do rơi vào vùng **saddle point**.
    - Về mặt lý thuyết ta phải tìm được `global minimum` nhưng thực tế thì chỉ cần tìm 1 cái `local minimum` mà đủ tốt thì cũng ok rồi.

</details>

----

<details>
  <summary><b> 2. Gradient Descent versions</b></summary>

  ![image](https://github.com/user-attachments/assets/c1ad95a9-711c-4a4d-b690-ae79813db8bd)

  ## Gradient Descent: Different Versions

  ### 🟢 Batch Gradient Descent
  > **Tất cả datapoints được đưa vào mô hình cùng 1 lúc để tính gradient**
  
  - Tính gradient dựa trên toàn bộ tập dữ liệu.
  - Ổn định và hội tụ mượt mà.
  - Tuy nhiên, chậm và tốn tài nguyên khi dữ liệu lớn.
  
  **Hình ảnh minh họa:** Đường đi mượt, thẳng đến điểm tối ưu.
  
  ---
  
  ### 🟠 Stochastic Gradient Descent (SGD)
  > **Từng datapoint một sẽ được đưa vào mô hình để tính gradient**
  
  - Cập nhật trọng số sau mỗi datapoint.
  - Nhanh hơn, nhưng nhiễu và không ổn định.
  - Có thể vượt qua local minima.
  
  **Hình ảnh minh họa:** Đường đi dao động mạnh quanh điểm tối ưu.
  
  ---
  
  ### 🔴 (Stochastic) Mini-batch Gradient Descent
  > **N datapoint sẽ được đưa vào mô hình cùng lúc để tính gradient**
  
  - Cân bằng giữa Batch và SGD.
  - Vừa nhanh, vừa ổn định.
  - Rất phổ biến trong huấn luyện mô hình deep learning.
  
  **Hình ảnh minh họa:** Đường đi hơi lượn sóng nhưng vẫn hội tụ nhanh.
  
  ---
  
  ### 📌 So sánh trực quan:
  | Loại Gradient Descent | Mức độ ổn định | Tốc độ cập nhật | Yêu cầu bộ nhớ |
  |------------------------|----------------|------------------|----------------|
  | Batch                 | Cao           | Chậm             | Cao            |
  | Stochastic            | Thấp          | Rất nhanh        | Thấp           |
  | Mini-batch            | Trung bình    | Nhanh            | Vừa phải       |

  > **Note: Về mặt lý thuyết thì `SGD` khác `Mini-batch` nhưng trên thực tế, người ta thường dùng `SGD` để ám chỉ cho `mini-batch`**

  

</details>
