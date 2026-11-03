import numpy as np
import matplotlib.pyplot as plt

def calculate_prob(X, w):
    # X=(k,d), w=(d,c) → (k,c)
    product = np.dot(X, w)

    # tránh overflow (trick rất quan trọng)
    max_row = np.max(product, axis=1, keepdims=True)
    product = product - max_row

    # exp
    exp_product = np.exp(product)

    # chuẩn hóa
    sum_row = np.sum(exp_product, axis=1, keepdims=True)
    prob = exp_product / sum_row

    return prob

class SoftMax():
    def fit(self,X,Y,lr=0.1,lam=0.01, nepoches = 2000,batch_num=10):
        # X=(k,d), w=(d,c) → (k,c)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X = (X - self.mean) / (self.std + 0.001)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        i=0
        batch_X =[]
        batch_Y=[]
        self.N=len(X)
        self.d=X.shape[1]
        self.c=Y.shape[1]
        self.w = np.random.randn(self.d, self.c) * 0.01
        w_old = self.w
        ep=0
        while i<self.N:
            if self.N-i<10:
                batch_X.append(X[i::])
                batch_Y.append(Y[i::])
                break
            batch_X.append(X[i:i+10:])
            batch_Y.append(Y[i:i+10:])
            i+=10
        total_batches = len(batch_X)
        while ep<nepoches:
            ep+=1
            for i in range(total_batches):
                x = batch_X[i]
                y= batch_Y[i]
                a = calculate_prob(x,self.w)
                self.w = self.w -lr/batch_num*np.dot(np.transpose(x),(a-y))
            if np.mean((self.w - w_old) ** 2) < 1e-12:
                break
            w_old =self.w
    def predict(self,X):
        X = (X - self.mean) / (self.std + 0.001)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        prob = calculate_prob(X,self.w)
        idx_max = np.argmax(prob,axis=1)
        return idx_max
    def accuracy(self,X,Y):
        list_res = []
        for y in Y:
            for i in range(self.c):
                if y[i]==1:
                    list_res.append(i)
                    break
        np_res = np.array(list_res)
        res_pred = self.predict(X)
        return np.mean(np_res==res_pred)
import numpy as np

def generate_data(n=300, d=2, c=3):
    np.random.seed(42)

    X = []
    Y = []

    samples_per_class = n // c

    for i in range(c):
        # tạo tâm cho mỗi class
        center = np.random.randn(d) * 5

        # tạo điểm xung quanh tâm
        x_class = center + np.random.randn(samples_per_class, d)

        # label one-hot
        y_class = np.zeros((samples_per_class, c))
        y_class[:, i] = 1

        X.append(x_class)
        Y.append(y_class)

    X = np.vstack(X)
    Y = np.vstack(Y)

    return X, Y
def generate_overlap_data(n=500, d=2, c=3):
    np.random.seed(42)

    X = np.random.randn(n, d) * 2  # tăng noise
    Y = np.zeros((n, c))

    for i in range(n):
        if X[i,0] + X[i,1] > 1:
            Y[i,0] = 1
        elif X[i,0] + X[i,1] > -1:
            Y[i,1] = 1
        else:
            Y[i,2] = 1

    return X, Y
def train_test_split(X, Y, test_size=0.2, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    split = int(n * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]

    return X_train, X_test, Y_train, Y_test
X, Y = generate_overlap_data(n=300, d=2, c=3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = SoftMax()
model.fit(X_train, Y_train, lr=0.1, nepoches=1000, batch_num=20)

train_acc = model.accuracy(X_train, Y_train)
test_acc = model.accuracy(X_test, Y_test)

print("Train accuracy:", train_acc)
print("Test accuracy:", test_acc)