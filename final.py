import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from io import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


data = open("flag.data","r")
names = open("flag.names","r")

df=pd.read_csv("flag.data")

#Filter Feature Selection + Classification
#ANOVA for numerical values
fX = df.drop(['name','language','landmass','zone','religion','red','green','blue','gold','white','black','orange','mainhue','crescent','triangle','icon','animate','text','topleft','botright'], 1)
fy=df['religion']
anova_feat, p_val = f_classif(fX, fy)
arr=[]
for i in range(0,p_val.size):
    if p_val[i]<0.05:
        arr.append(fX.columns[i])
print("Selected Features using ANOVA:")
print(arr)

fX2 = df.drop(['name','area','population','religion','bars','stripes','colours','mainhue','circles','crosses','saltires','quarters','sunstars','topleft','botright'], 1)

#Chi-squared for categorical values
chi_feat, p_val2 = chi2(fX2,fy)
arr2=[]
for i in range(0,p_val2.size):
    if p_val2[i]<0.05:
        arr2.append(fX2.columns[i])
print("Selected Features using Chi-squared:")
print(arr2)

arr=arr+arr2
print("Selected Features using ANOVA+Chi-squared(Filter Feature Selection):")
print(arr)
temp=df.drop(arr,1)
Filter_X=df.drop(temp.columns,1)
correlation = Filter_X.corr()
plt.figure(figsize=(30,8))
sns.heatmap(correlation,vmax=1,vmin=-1,center=0,linewidth=.5,square=True,annot=True,annot_kws={'size':8},fmt='.1f',cmap='BrBG_r')
plt.title('Correlation Map')
plt.savefig("correlation.png")



#Random Forest Classification
#X_train, X_test, y_train, y_test = train_test_split(Filter_X, fy, test_size=0.20,shuffle=False)
#model=RandomForestClassifier()
#model.fit(X_train,y_train)
#pred_fy=model.predict(X_test)
#print("Predicted test values and accuracy score using Filter Feature Selection + Random Forest Classifier:")
#print(pred_fy)
#sr=accuracy_score(y_test,pred_fy)
#print(sr)

#Decision Tree Classification
#model=DecisionTreeClassifier()
#model.fit(X_train,y_train)
#dot_data = StringIO()
#export_graphviz(model, out_file=dot_data,
#                filled=True, rounded=True,
#                special_characters=True,
#                feature_names=X_train.columns,
#                class_names=["0","1", "2", "3", "4", "5", "6", "7"])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(graph.create_png())
#graph.write_png("filter_selection_decision_tree.png")
#print("Predicted test values and accuracy score using Filter Feature Selection + Decision Tree Classifier:")
#pred_fy=model.predict(X_test)
#print(pred_fy)
#sr=accuracy_score(y_test,pred_fy)
#print(sr)


#Wrapper Feature Selection + Classification


#X = df.drop(['name','religion','mainhue','topleft','botright'], 1)
#y=df['religion']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

#Recursive Filter Elimination + Decision Tree Classification
X_train, X_test, y_train, y_test = train_test_split(Filter_X, fy, test_size=0.20,shuffle=True)
rfe2=RFE(estimator=DecisionTreeClassifier(criterion="entropy"), n_features_to_select=10)
rfe2.fit(X_train,y_train)
test=rfe2.transform(X_train)
dot_data = StringIO()
arr=rfe2.get_support()
arr4=[]
for i in range(0,arr.size):
    if arr[i]:
        arr4.append(X_train.columns[i])
print("Selected Features using RFE+Decision Tree:")
print(arr4)
export_graphviz(rfe2.estimator_, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=arr4,
                class_names=["0","1", "2", "3", "4", "5", "6", "7"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("rfe_decision_tree.png")
print("Predicted test values using Recursive Feature Elimination + Decision Tree Classifier:")
pred_y=rfe2.predict(X_test)
print(pred_y)
print("The real test values:")
print(y_test.to_numpy())
sr=accuracy_score(y_test,pred_y)
print("The accuracy:")
print(sr)
print(classification_report(y_test,pred_y,zero_division=1))