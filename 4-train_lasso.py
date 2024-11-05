import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score,recall_score,precision_score, confusion_matrix, roc_curve,ConfusionMatrixDisplay ,auc
from tqdm import tqdm
import matplotlib.pyplot as plt
from pylab import mpl
import copy
from sklearn.ensemble import RandomForestClassifier
### KNN Classifier    
from sklearn.neighbors import KNeighborsClassifier
 
### Logistic Regression Classifier    
from sklearn.linear_model import LogisticRegression
 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import  AdaBoostClassifier


from sklearn.svm import SVC
 

from sklearn import tree
# 在十折交叉验证的每一折中使用Lasso进行特征选择涉及到在每一折中独立地应用Lasso，然后用筛选出的特征来训练模型。这样可以确保特征选择过程不会泄露验证集的信息。

mpl.rcParams["font.sans-serif"] = ['SimHei']

from sklearn.linear_model import LassoCV

from sklearn.preprocessing import StandardScaler

import argparse

from sklearn.preprocessing import Normalizer

from sklearn.feature_selection import SelectFromModel

import pandas as pd

def get_model(model_name):
    if model_name == 'KNN':
        clf = KNeighborsClassifier()
    elif model_name == 'RandomForest':
        clf = RandomForestClassifier(random_state=2022)
    elif model_name == 'LogisticRegressor':
        clf = LogisticRegression(solver="sag",penalty='l2',random_state=2022)
    elif  model_name == 'DecisionTree':
        clf = tree.DecisionTreeClassifier(random_state=2022)
    elif  model_name == 'AdaBoost':
        clf = AdaBoostClassifier(random_state=2022)
    elif  model_name == 'SVM':
        clf = SVC(random_state=2022,kernel='rbf', probability=True)

    return clf


def lasso_feature_selection(X_train, y_train, X_test,y_test):
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    alphas = np.logspace(-10, 1, 100, base = 10)
    selector_lasso = LassoCV(alphas=alphas, cv = 10, max_iter = 1000000,random_state=2022)
    
    selector_lasso.fit(X_train_scaled,y_train)
    selected_features = selector_lasso.coef_ != 0
    
    print (X_train_scaled[:,selected_features].shape)
    return X_train_scaled[:,selected_features],  X_test_scaled[:,selected_features]




label_dict = {'HCM':0,'HHD':1,'NOR':2}
label_name_dict = {v: k for k, v in label_dict.items()}
# Main function
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname_list', default = ['KNN','RandomForest', 'LogisticRegressor','DecisionTree','AdaBoost','SVM'],nargs='+')
    args = parser.parse_args()
    modelname_list = args.modelname_list
    
    for model_name in modelname_list:

        print (model_name)
        # Load data
        X, y = np.load('feature.npy'),np.load('label.npy')


        # 创建十折交叉验证对象
        kfold = KFold(n_splits=10,shuffle=True,random_state=2022)
        clf = get_model(model_name)

        y_pred = []
        true_labels = []
        
        for train_index, test_index in tqdm(kfold.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # 使用LASSO进行特征选择
            X_train_lasso, X_test_lasso = lasso_feature_selection(X_train, y_train, X_test,y_test)
            
            # 训练并预测
            clf.fit(X_train_lasso, y_train)
            y_pred.extend(copy.deepcopy(clf.predict_proba(X_test_lasso)))
            true_labels.extend(copy.deepcopy(y_test))
        
        y_pred = np.array(y_pred)
        # y_pred = cross_val_predict(clf, X, y, cv=kfold, method='predict_proba')
        y_class_pred = np.argmax(y_pred, axis=1)

        # 计算指标
        accuracy = accuracy_score(y, y_class_pred)
        f1 = f1_score(y, y_class_pred, average='weighted')
        auc_score = roc_auc_score(y, y_pred, multi_class='ovr')
        weighted_recall = recall_score(y,  y_class_pred, average='weighted')
        weighted_precision = precision_score(y,  y_class_pred, average='weighted')
        confusion = confusion_matrix(y, y_class_pred)
        
        # 初始化TN和FP数组
        TN = np.zeros(len(label_dict))
        FP = np.zeros(len(label_dict))

        # 对于每个类别，计算TN和FP
        for i in range(len(label_dict)):
            # 对于当前类别i，所有不是类别i的索引
            non_i_indices = [j for j in range(len(label_dict)) if j != i]
            
            # 真阴性是混淆矩阵中不是当前类别i，且被正确预测为不是类别i的总和
            TN[i] = np.sum(confusion[non_i_indices, :][:, non_i_indices])
            
            # 假阳性是混淆矩阵中不是类别i，但被错误预测为类别i的总和
            FP[i] = np.sum(confusion[non_i_indices, i])

        # 计算每个类别的特异性
        TNR = TN / (TN + FP)

        # 计算加权特异性
        weighted_specificity = np.sum(TNR * (TN + FP)) / np.sum(TN + FP)
        
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"sensitivity: {weighted_recall}")
        print(f"specificity: {weighted_specificity}")
        print(f"AUC: {auc_score}")
        # print("Confusion Matrix:")
        # print(confusion)

        # 将数字类别标签转换为文字标签
        class_labels = [label for label, index in sorted(label_dict.items(), key=lambda item: item[1])]


        # 绘制并保存混淆矩阵
        ConfusionMatrixDisplay.from_predictions(y,  np.argmax(y_pred, axis=1),cmap='Blues',display_labels=class_labels)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.savefig(f'lasso/confusion_matrix_lasso_{model_name}.png')
        plt.clf()

        # 绘制并保存ROC曲线

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(np.unique(y))):
            fpr[i], tpr[i], _ = roc_curve(y == i, y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 绘制所有ROC曲线
        plt.figure()
        for i in range(len(np.unique(y))):
            plt.plot(fpr[i], tpr[i], lw=2, label='{0}类别的ROC曲线 (AUC = {1:0.2f})'.format(label_name_dict[i], roc_auc[i]))
        
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC 曲线')
        plt.legend(loc="lower right")
        plt.savefig(f'lasso/multi_class_roc_curve_lasso_{model_name}.png')
        plt.clf()

if __name__ == "__main__":
    main()
