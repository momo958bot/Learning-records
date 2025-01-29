import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# 自定义缩放器
class CustomScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        # 标准化
        X_standardized = (X - self.mean_) / self.std_
        # 缩放到 0-2 的范围，并设置均值为 1
        X_scaled = X_standardized * 1 + 50  # 目标均值为 1
        # 将所有值限制在 0 到 2 之间
        X_scaled = np.clip(X_scaled, 0, 100)
        return X_scaled

# 加载数据
data = pd.read_csv('clean_3.csv')  # 将 'TrainOnMe.csv' 替换为您的 CSV 文件路径

# 特征和标签
X = data.drop(['y'], axis=1)
y = data['y']

# 特征工程
categorical_features = ['x7']
numerical_features = [col for col in X.columns if col not in categorical_features]

# 构建特征转换器
transformer = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_features),
        ('custom_scaler', CustomScaler(), numerical_features)  # 使用自定义缩放器
    ])

# 机器学习模型
model = RandomForestClassifier()

# K-Fold 交叉验证
kf = KFold(n_splits=8, shuffle=True, random_state=42)
accuracies = []
predicted_labels = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    pipeline = Pipeline([
        ('transformer', transformer),
        ('model', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    predicted_labels.extend(y_pred)

    print(f"Fold Accuracy: {accuracy}")

# 计算平均准确度
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average Accuracy: {average_accuracy}")

# 将预测的标签写入文本文件
with open('predicted_labels.txt', 'w') as f:
    for label in predicted_labels:
        f.write(label + '\n')
