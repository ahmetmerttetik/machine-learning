import numpy as np
from tqdm import tqdm

class LinearRegression:

    def __init__(self, lr , iter):
        self.lr = lr
        self.iter = iter

    def fit(self, x , y):

        n , m = x.shape

        self.w = np.zeros(m)

        self.b = 0

        self.loss_list = []
        
        for iter in tqdm(range(self.iter)):

            y_hata = np.dot(x,self.w) + self.b

            error = y - y_hata

            loss = np.mean(error ** 2)

            self.loss_list.append(loss)

            dw = (-2/n) * np.dot(x.T,error)

            db = (-2/n) * np.sum(error)

            self.w = self.w - (self.lr*dw)
            self.b = self.b - (self.lr*db)

            if iter % 20 == 0:
                print(f"loss = {loss} , w = {self.w} , b = {self.b}")

    def predict(self,x):
        return np.dot(x , self.w) + self.b

