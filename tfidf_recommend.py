#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 02:32:08 2026

@author: xxc
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# =====================
# 读取数据
# =====================
data_df = pd.read_excel("data.xlsx")
lib_df = pd.read_excel("library.xlsx")

# keywords列
user_texts = data_df["keywords"].fillna("").tolist()
lib_texts = lib_df["keywords"].fillna("").tolist()

# =====================
# 构建 TF-IDF
# =====================
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")

# ⚠️ 必须合并语料库（关键）
all_texts = user_texts + lib_texts

tfidf_matrix = vectorizer.fit_transform(all_texts)

# 切分
user_tfidf = tfidf_matrix[:len(user_texts)]
lib_tfidf = tfidf_matrix[len(user_texts):]

# =====================
# 用户权重
# =====================
# 指数权重（最新最大）
# =====================
n = len(user_texts)

alpha = 0.8  # ⭐ 衰减系数（0.7~0.9之间调）

weights = np.array([alpha**i for i in range(n)])


# 归一化（非常重要）
weights = weights / weights.sum()

# 转列向量
weights = weights.reshape(-1, 1)



# =====================
# 计算用户向量（加权）
# =====================
user_vector = np.sum(user_tfidf.toarray() * weights, axis=0)

# 归一化（很重要）
norm = np.linalg.norm(user_vector)
if norm != 0:
    user_vector = user_vector / norm
# =====================
# 计算相似度
# =====================
lib_vectors = lib_tfidf.toarray()

# 归一化
lib_vectors = normalize(lib_vectors)

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

scores = cosine_similarity(lib_vectors, user_vector.reshape(1, -1)).flatten()

# =====================
# 排序
# =====================
lib_df["score"] = scores

result = lib_df.sort_values(by="score", ascending=False)

# =====================
# 输出Top10
# =====================
lib_df.columns = lib_df.columns.str.strip()
print(result[["No.", "Title", "score"]].head(10).to_string(index=False))
# 保存
result.to_excel("recommend_result.xlsx", index=False)