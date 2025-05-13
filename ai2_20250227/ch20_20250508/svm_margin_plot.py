# 導入必要的模組
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 定義一個函數來繪製 SVM 的決策邊界
def plot_svm_boundary(model, X, y):
    
    # 將輸入的 DataFrame 轉換為 NumPy 陣列
    X = X.values
    y = y.values
    
    # 繪製散點圖
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='seismic')
    
    # 獲取當前的軸對象
    ax = plt.gca()
    xlim = ax.get_xlim()  # 獲取 x 軸範圍
    ylim = ax.get_ylim()  # 獲取 y 軸範圍

    # 創建一個網格來評估模型
    xx = np.linspace(xlim[0], xlim[1], 30)  # 在 x 軸範圍內生成30個等距點
    yy = np.linspace(ylim[0], ylim[1], 30)  # 在 y 軸範圍內生成30個等距點
    YY, XX = np.meshgrid(yy, xx)  # 生成網格
    xy = np.vstack([XX.ravel(), YY.ravel()]).T  # 將網格展平並組合成二維點座標
    Z = model.decision_function(xy).reshape(XX.shape)  # 計算決策函數值並重塑為網格形狀

    # 繪製決策邊界和間隔
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # 繪製支持向量
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    
    # 顯示圖形
    plt.show()
