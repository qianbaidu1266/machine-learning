# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 22:15:35 2018

@author: Administrator
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy import stats
#%matplotlib inline

#step1：读取数据
train_df=pd.read_csv('./data/train.csv',index_col=0)   #第零列
test_df=pd.read_csv('./data/test.csv',index_col=0)


#检视源数据
train_df.head()   #通过查看表格头部5行元素来观察数据的大致组成。
print(train_df.head())

#step2:合并数据 ：特征工程的工作
#label本身并不平滑，为了分类器的学习更加准确，首先把label平滑（正态化）
#log1p就是log(x+1),避免了复值的问题，这里把数据平滑化，最后算结果要把预测的平滑数据变回去
prices=pd.DataFrame({"price":train_df["SalePrice"],"log(price+1)":np.log1p(train_df["SalePrice"])})
prices.hist()

y_train=np.log1p(train_df.pop('SalePrice'))
#然后把剩下的数据合并
all_df=pd.concat((train_df,test_df),axis=0) #把训练集和测试集合在一起
all_df.shape
#print(all_df.shape)
y_train.head() #y_train是SalePrice那一列
#print(y_train.head())




#step3:变量转换，特征工程和数据清洗的工作！！！
"""
正确化变量属性：MSSubClass 的值其实应该是一个category（等级的划分），虽然是数字，
但是代表多类别，Pandas是不会懂这些。使用DF的时候，这类数字符号会被默认记成数字。
这种东西就很有误导性，我们需要把它变回成string
"""
print (all_df['MSSubClass'].dtypes )  #dtype('int64')
all_df['MSSubClass'] = all_df['MSSubClass'].astype(str)  #转为string，便于查看他的分布情况
print( all_df['MSSubClass'].dtypes)
print( all_df['MSSubClass'].value_counts())
"""
把category的变量转变成numerical表达形式：当我们用numerical来表达categorical的时候，
要注意，数字本身有大小的含义，所以乱用数字会给之后的模型学习带来麻烦。
于是我们可以用One-Hot的方法来表达category。pandas自带的get_dummies方法，可以帮你一键做到One-Hot。
"""
print( pd.get_dummies(all_df['MSSubClass'],prefix = 'MSSubClass'))#处理离散型变量的方法get_dummies,即就是one-hot).head()
all_dummy_df = pd.get_dummies(all_df)  #pandas自动选择那些事离散型变量，省去了我们做选择
print (all_dummy_df.head())

# 处理好numerical变量
print (all_dummy_df.isnull().sum().sort_values(ascending = False).head(11))
# 我们这里用mean填充
mean_cols = all_dummy_df.mean()
print (mean_cols.head(10))
all_dummy_df = all_dummy_df.fillna(mean_cols)
print( all_dummy_df.isnull().sum().sum())
 
# 标准化numerical数据
numeric_cols = all_df.columns[all_df.dtypes != 'object']
print (numeric_cols)
#查看缺失情况，按照缺失情况排序
# 注意：处理缺失情况时要看数据描述，确实值得处理方式工具意义和缺失情况有很大不同，有时确实本身就有意义，我们要把他当
#做一个类型，其他时候要将其补上或者删除这个特征
numeric_col_means = all_dummy_df.loc[:,numeric_cols].mean()
numeric_col_std = all_dummy_df.loc[:,numeric_cols].std()
all_dummy_df.loc[:,numeric_cols] = (all_dummy_df.loc[:,numeric_cols] - numeric_col_means) / numeric_col_std
 
# step4 建立模型
# 把数据处理之后，送回训练集和测试集
dummy_train_df = all_dummy_df.loc[train_df.index]
dummy_test_df = all_dummy_df.loc[test_df.index]
print (dummy_train_df.shape,dummy_test_df.shape)
 
# 将DF数据转换成Numpy Array的形式，更好地配合sklearn
 
X_train = dummy_train_df.values
X_test = dummy_test_df.values
 
# Ridge Regression
alphas = np.logspace(-3,2,50)
test_scores = []
for alpha in alphas:
    
    clf = Ridge(alpha)
    test_score = np.sqrt(-cross_val_score(clf,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))
    plt.plot(alphas,test_scores)
    plt.title('Alpha vs CV Error')
    plt.show()
 




