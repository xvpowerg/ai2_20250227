import numpy as np

import matplotlib.pyplot as plt


# 目標函數(損失函數):y=x^2
def func(x): return x ** 2

# 目標函數的一階導數:dy/dx=2*x
def dfunc(x): return 2 * x

def GD(x_start, df, epochs, lr):    
    """  梯度下降法。給定起始點與目標函數的一階導函數，求在epochs次反覆運算中x的更新值
        :param x_start: x的起始點    
        :param df: 目標函數的一階導函數    
        :param epochs: 反覆運算週期    
        :param lr: 學習率    
        :return: x在每次反覆運算後的位置（包括起始點），長度為epochs+1    
     """    
    xs = np.zeros(epochs+1)    
    w = x_start    
    xs[0] = w    
    for i in range(epochs):         
        dx = df(w)        
        # 權重的更新W_new
        # W_new = W — learning_rate * gradient        
        w += - dx * lr         
        xs[i+1] = w    
    return xs

# Main
# 起始權重
x_start = -5    
# 執行週期數
epochs = 5
# 學習率   
lr = 0.3  
# 梯度下降法 
# *** Function 可以直接當參數傳遞 ***
w = GD(x_start, dfunc, epochs, lr=lr) 
print (np.around(w, 2))
# 輸出：[-5.     -2.     -0.8    -0.32   -0.128  -0.0512]

color = 'r'    
from numpy import arange
t = arange(-6.0, 6.0, 0.01)
plt.plot(t, func(t), c='b')
plt.plot(w, func(w), c=color, label='lr={}'.format(lr))    
plt.scatter(w, func(w), c=color, ) 

from matplotlib.font_manager import FontProperties

plt.title('Gradient Descent')
plt.xlabel('w', fontsize=20)
plt.ylabel('Loss', fontsize=20)

plt.show()
