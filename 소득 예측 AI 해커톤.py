#!/usr/bin/env python
# coding: utf-8

# ## 데이터 불러오기 및 확인

# In[2]:


import pandas as pd

train = pd.read_csv('C:/Users/LSJ/Desktop/open/open/train.csv')
test = pd.read_csv('C:/Users/LSJ/Desktop/open/open/test.csv')

display(train.head(3))
display(test.head(3))


# ## 데이터 전처리 1: 학습 및 추론 데이터 설정

# In[6]:


train_x = train.drop(columns=['ID', 'Income'])
train_y = train['Income']

test_x = test.drop(columns=['ID'])


# ## 데이터 전처리 2: 범주형 변수 수치화

# In[14]:


from sklearn.preprocessing import LabelEncoder
import numpy as np

encoding_target = list(train_x.dtypes[train_x.dtypes == "object"].index)

for i in encoding_target:
    le = LabelEncoder()
    
# train과 test 데이터셋에서 해당 열의 모든 값을 문자열로 변환
train_x[i] = train_x[i].astype(str)
test_x[i] = test_x[i].astype(str)
    
le.fit(train_x[i])
train_x[i] = le.transform(train_x[i])
    
# test 데이터의 새로운 카테고리에 대해 le.classes_ 배열에 추가
for case in np.unique(test_x[i]):
    if case not in le.classes_: 
        le.classes_ = np.append(le.classes_, case)
    
test_x[i] = le.transform(test_x[i])


# ## 모델 선정 및 학습

# In[15]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor() 
model.fit(train_x, train_y) 


# ## 예측 수행

# In[16]:


preds = model.predict(test_x)


# ## 제출양식에 예측결과 입력

# In[17]:


submission = pd.read_csv("C:/Users/LSJ/Desktop/open/open/sample_submission.csv")
submission['Income'] = preds
submission


# ## 예측결과 저장

# In[18]:


submission.to_csv("C:/Users/LSJ/Desktop/open/open/baseline_submission.csv", index=False)

