# 知能システム論 第4回 課題 多次元データをPCAで次元削減し，k-meansでクラスタリングする
# 藤井智哉(理学部生物情報科学科, 学籍番号: 05235509)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans

def kmeans():
    data = pd.read_csv('pokemonGO.csv', na_filter=False) # NaNが生じないデータの読み込み
    df = data[['Type 1', 'Type 2', 'Max CP', 'Max HP']] # 必要な列のみ選択
    df.loc[:, 'Type 1'] = df['Type 1'].astype('category').cat.codes # Type 1(文字列)を離散値に変換
    df.loc[:, 'Type 2'] = df['Type 2'].astype('category').cat.codes # Type 1(文字列)を離散値に変換
    ids = data['Pokemon No.']
    
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df)
    kmeans_model = KMeans(n_clusters=2, random_state=1).fit(df_pca) # クラス数は射影した次元数に合わせる
    labels = kmeans_model.labels_

    return df_pca, labels, ids

def kernel_kmeans():
    data = pd.read_csv('pokemonGO.csv', na_filter=False) # NaNが生じないデータの読み込み
    df = data[['Type 1', 'Type 2', 'Max CP', 'Max HP']] # 必要な列のみ選択
    df.loc[:, 'Type 1'] = df['Type 1'].astype('category').cat.codes # Type 1(文字列)を離散値に変換
    df.loc[:, 'Type 2'] = df['Type 2'].astype('category').cat.codes # Type 1(文字列)を離散値に変換
    ids = data['Pokemon No.']
    
    k_pca = KernelPCA(n_components=2, kernel='poly') # 多項式カーネルを使用
    df_k_pca = k_pca.fit_transform(df)
    kmeans_model = KMeans(n_clusters=2, random_state=1).fit(df_k_pca) # クラス数は射影した次元数に合わせる
    labels = kmeans_model.labels_

    return df_k_pca, labels, ids

def visualize(df_pca, labels, ids, idx, title):
    plt.figure(figsize=(6, 6))
    color_codes = {0: 'red', 1: 'green'}
    colors = [color_codes[label] for label in labels]
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=colors)
    for i, (x, y) in enumerate(df_pca):
        plt.text(x + 0.02, y + 0.02, ids[i], fontsize=9)
    plt.title(title)
    plt.savefig('IS_4_homework{}.png'.format(idx))
    plt.show()

df_pca, labels, ids = kmeans()
visualize(df_pca, labels, ids, idx=1, title='PCA and k-means result for Gen1 Pokemon data')
df_k_pca, k_labels, k_ids = kernel_kmeans()
visualize(df_k_pca, k_labels, k_ids, idx=2, title='Kernel PCA and k-means result for Gen1 Pokemon data')

