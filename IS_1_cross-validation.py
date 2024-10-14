# 知能システム論 第1回 課題1 カーネルモデルの正則化最小2乗回帰のバンド幅hと正則化パラメータλを交差検証により決定
# 藤井智哉(理学部生物情報科学科, 学籍番号: 05235509)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
np.random.seed(0) # set the random seed for reproducibility

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(xmin, xmax, sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise

def calc_design_matrix(x, c, h): # カーネル行列
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

# create sample
sample_size = 50
xmin, xmax = -3, 3
x, y = generate_sample(xmin, xmax, sample_size)

# k-fold cross-validation
best_l, best_h = None, None
L = [0.01, 0.1, 1, 10]
H = [0.1, 0.5, 1.0, 1.5]
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True)
lowest_mse = float('inf')

for l in L:
    for h in H:
        mse_list = []
        for train_idx, val_idx in kf.split(x):
            # 訓練データと検証データに分割
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # 学習
            k_train = calc_design_matrix(x_train, x_train, h)
            theta = np.linalg.solve(k_train.T.dot(k_train) + l * np.identity(len(k_train)), k_train.T.dot(y_train[:, None]))
            # 検証
            k_val = calc_design_matrix(x_train, x_val, h)
            y_pred = k_val.dot(theta)
            mse = mean_squared_error(y_val, y_pred)
            mse_list.append(mse)
        # MSEの平均を計算
        mse_mean = np.mean(mse_list)
        if mse_mean < lowest_mse:
            lowest_mse = mse_mean
            best_l, best_h = l, h

print('Best (l, h): ({}, {})'.format(best_l, best_h))
# 最適なパラメータで学習
k = calc_design_matrix(x, x, best_h)
theta = np.linalg.solve(k.T.dot(k) + best_l * np.identity(len(k)), k.T.dot(y[:, None]))

# create data to visualize the prediction
X = np.linspace(start=xmin, stop=xmax, num=5000)
K = calc_design_matrix(x, X, best_h)
prediction = K.dot(theta)

# visualization
plt.clf
plt.scatter(x, y, c='green', marker='o') # 訓練標本
plt.plot(X, prediction, c='blue') # 学習結果
plt.savefig('IS_1_homework1.png')
plt.show()
