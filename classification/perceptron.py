import numpy as np
import matplotlib.pyplot as plt
class PLA:
    def loss_fuction(self,x,w,y):
        return -y*np.dot(x,w)
    def fit(self,X,Y):
        # X is a matrix, Y is label
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X, axis=0)

        # Scale X
        X = (X - self.mean) / self.std
        self.X_scaled = X
        # Thêm bias
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        self.w= np.zeros(shape=(X.shape[1],1))
        k=0
        while True :
            complete=True
            k=0
            for i in range(len(X)):
                x=X[i].reshape(1,-1)
                y=Y[i]
                if self.loss_fuction(x,self.w,y) >= 0:
                    k+=1
                    self.w+=y*x.reshape(-1,1)
                    complete=False
            print(k)
            if complete :
                break
    def predict(self,X):
        X = (X - self.mean) / self.std
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return [np.sign(np.dot(x.reshape(1,-1),self.w))for x in X]

    def score(self, X, Y):
        y_pred = np.array(self.predict(X)).reshape(-1, 1)
        return np.mean(y_pred == Y)
# Tạo 10 điểm cho lớp +1 (nằm ở góc trên bên phải)
X1 = np.random.randn(100, 2) + np.array([2, 2])
y1 = np.ones((100, 1))

# Tạo 10 điểm cho lớp -1 (nằm ở góc dưới bên trái)
X2 = np.random.randn(100, 2) + np.array([-2, 0])
y2 = -np.ones((100, 1))

# Gộp lại thành bộ dữ liệu X (20, 2) và y (20, 1)
X = np.vstack((X1, X2))
y = np.vstack((y1, y2))

# Trộn ngẫu nhiên dữ liệu
rd_idx = np.random.permutation(X.shape[0])
X = X[rd_idx]
y = y[rd_idx]
pla = PLA()
pla.fit(X,y)
plt.figure(figsize=(10, 6))

# Lấy dữ liệu đã scale từ trong mô hình ra để vẽ
X_plot = pla.X_scaled
y_plot = y

# Vẽ các điểm dữ liệu lớp +1 (màu xanh lam) và -1 (màu đỏ)
plt.scatter(X_plot[y_plot.flatten() == 1, 0], X_plot[y_plot.flatten() == 1, 1], color='blue', label='Lớp +1', edgecolors='k', s=100)
plt.scatter(X_plot[y_plot.flatten() == -1, 0], X_plot[y_plot.flatten() == -1, 1], color='red', label='Lớp -1', edgecolors='k', s=100)

# Vẽ đường phân chia (Decision Boundary)
# Công thức: w0 + w1*x1 + w2*x2 = 0  => x2 = -w1/w2 * x1 - w0/w2
w0 = pla.w[0, 0]
w1 = pla.w[1, 0]
w2 = pla.w[2, 0]

# Tạo các điểm x1 để vẽ đường thẳng
x1_boundary = np.linspace(np.min(X_plot[:, 0]) - 0.5, np.max(X_plot[:, 0]) + 0.5, 100)

# Tính x2 tương ứng (kiểm tra w2 != 0 để tránh chia cho 0)
if w2 != 0:
    x2_boundary = -(w1 / w2) * x1_boundary - (w0 / w2)
    plt.plot(x1_boundary, x2_boundary, color='black', linewidth=2, label='Đường phân chia')
else:
    # Trường hợp đặc biệt w2=0, đường thẳng đứng x1 = -w0/w1
    plt.axvline(x=-w0/w1, color='black', fontweight='bold', label='Đường phân chia (đứng)')

plt.title(f'Trực quan hóa PLA (Dữ liệu đã Scale) - Accuracy: %')
plt.xlabel('Đặc trưng 1 (Scaled $x_1$)')
plt.ylabel('Đặc trưng 2 (Scaled $x_2$)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()