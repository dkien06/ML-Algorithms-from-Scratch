import math
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x:int):
    return 1/(1+np.exp(-x))
def prob(w,x):
    return sigmoid(np.dot(w,x))
class LogisticRegression:
    def loss(self, X, Y, lam):
        z = sigmoid(X @ self.w)
        z = np.clip(z, 1e-15, 1 - 1e-15)

        loss = -np.mean(Y * np.log(z) + (1 - Y) * np.log(1 - z))
        reg = 0.5 * lam / X.shape[0] * np.sum(self.w ** 2)

        return loss + reg
    def fit(self,X,Y,lr=0.1,lam=0.01, nepoches = 2000):
        # chuan hoa du lieu
        self.mean = np.mean(X,axis =0)
        self.std = np.std(X,axis=0)
        X = (X-self.mean)/(self.std+0.001)
        X= np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        self.w = np.zeros(X.shape[1])
        N,d= X.shape[0] , X.shape[1]
        ep=0
        w_old = self.w
        hist_loss = []
        while ep < nepoches:
            ep+=1
            mix_idx = np.random.permutation(N)
            for i in mix_idx :
                x = X[i]
                y=Y[i]
                self.w = self.w - lr*((sigmoid(np.dot(self.w,x))-y)*x+lam*self.w)
            if np.linalg.norm(self.w-w_old)/d < 1e-6 :
                print("Check")
                break
            w_old = self.w
            print(self.loss(X,Y,lam))
        return hist_loss
    def predict(self,X,division=0.5):
        X = (X - self.mean) / (self.std + 0.001)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return [int(prob(self.w,x) >= division) for x in X]
    def accuracy(self,X,Y,division=0.5):
        Y_pred = self.predict(X,division)
        return np.mean(Y_pred==Y)*100

def plot_decision_boundary(model, X, Y):
    # Chuẩn hóa giống lúc train
    X_norm = (X - model.mean) / (model.std + 0.001)

    # Plot data points
    for i in range(len(Y)):
        if Y[i] == 0:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color='blue')
        else:
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color='red')

    # Vẽ decision boundary
    w = model.w
    x1 = np.linspace(X_norm[:,0].min()-1, X_norm[:,0].max()+1, 100)
    x2 = -(w[0] + w[1]*x1) / w[2]

    plt.plot(x1, x2)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.show()

