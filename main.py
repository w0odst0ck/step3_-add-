import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 读取数据集
train_data = pd.read_csv('train_dataset.csv')
test_data = pd.read_csv('test_dataset.csv')
val_data = pd.read_csv('val_dataset.csv')

# 提取特征和目标变量
X_train = train_data.iloc[:, :-1]  # 训练集特征向量
y_train = train_data.iloc[:, -1]  # 训练集目标变量
X_test = test_data.iloc[:, :-1]  # 测试集特征向量
y_test = test_data.iloc[:, -1]  # 测试集目标变量
X_val = val_data.iloc[:, :-1]  # 验证集特征向量
y_val = val_data.iloc[:, -1]  # 验证集目标变量

# 创建PCA对象和分类器对象
pca = PCA()
classifier = RandomForestClassifier()

# 创建Pipeline，包括特征缩放、降维和分类器
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', pca),
    ('classifier', classifier)
])

# 定义超参数空间
param_grid = {
    'pca__n_components': [5, 6, 7, 8, 9, 10],  # 不同的降维维度
    'classifier__n_estimators': [50, 75, 100, 150, 200, 250],  # 随机森林的树的数量
    'classifier__max_depth': [None, 1, 2, 5, 10, 20],  # 随机森林的最大树深度
}

# 使用GridSearchCV进行模型选择和超参数调优
grid_search = GridSearchCV(pipeline, param_grid, cv=5)

# 初始化模型
grid_search.fit(X_train, y_train)

# 输出最佳模型和超参数
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Model:", best_model)
print("Best Parameters:", best_params)

# 在验证集上进行预测和评估
y_val_pred = best_model.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred, average='weighted')
recall = recall_score(y_val, y_val_pred, average='weighted')
f1 = f1_score(y_val, y_val_pred, average='weighted')
print("Validation Accuracy:", accuracy)
print("Validation Precision:", precision)
print("Validation Recall:", recall)
print("Validation F1-score:", f1)

# 在测试集上进行预测和评估
y_test_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')
print("Test Accuracy:", accuracy)
print("Test Precision:", precision)
print("Test Recall:", recall)
print("Test F1-score:", f1)
