import numpy as np
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
m = 5.12499  # -2 to start, change me please
b = 14.4299 # 40 to start, change me please
# Sample data
x = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9])  #10個點
y = np.array([10, 20, 25, 30, 40, 45, 40, 50, 60, 55])  #10個點
#設定損失函數
def loss(m,b):
    return np.sum((y - (x*m+b) )**2)/len(x)
#利用for 迴圈 指定 跑的次數
for i in range(1,100000):
    dm=10**-6  #每跑一次將dm乘以一個很小的數字
    db=10**-7  #每跑一次將db乘以一個很小的數字
    if loss(m+dm,b) > loss(m-dm,b):
        m = m-dm*10
    else:
        m = m+dm*10
    if loss(m,b+db) > loss(m,b-db):
        b = b-db*10**5
    else:
        b = b+db*10**5
    if i%1000 ==0:
        print(loss(m,b))
print(m,b)
y_hat = x * m + b
plt.plot(x, y, '.')
plt.plot(x, y_hat, '-')
plt.show() 
print("Loss:", np.sum((y - y_hat)**2)/len(x))

