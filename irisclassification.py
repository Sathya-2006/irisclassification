import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# 1. Load Data
iris=load_iris()
iris
X,y=iris.data,iris.target


# 2. Preprocess (Scaling)
scaler=StandardScaler()
X=scaler.fit_transform(X)


# 3. Train-Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier(random_state=42)
models=[("DecisionTreeClassifier",dt)]
for name,model in models:
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)
  print(f'\n{name}')
  print(f'accuracy_score:{accuracy_score(y_test,y_pred)}')
  print(f'classification_report:{classification_report(y_test,y_pred)}')
  print(f'confusion_matrix:{confusion_matrix(y_test,y_pred)}')

from sklearn.svm import SVC
svm=SVC(probability=True,random_state=42)                       #support vector machine 
models=[("SVM",svm)]
for name,model in models:
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)
  print(f'\n{name}')
  print(f'accuracy_score:{accuracy_score(y_test,y_pred)}')
  print(f'classification_report:{classification_report(y_test,y_pred)}')
  print(f'confusion_matrix:{confusion_matrix(y_test,y_pred)}')


from sklearn.ensemble import RandomForestClassifier, VotingClassifier     # randomforestclassifciation
rf=RandomForestClassifier(random_state=42)
models=[("RandomForestClassifier",rf)]
models=[("RandomForestClassifier",rf)]
for name,model in models:
  model.fit(X_train,y_train)
  y_pred=model.predict(X_test)
  print(f'\n{name}')
  print(f'accuracy:{accuracy_score(y_pred,y_test)}')
  print(f'classification_report:{classification_report(y_pred,y_test)}')
  print(f'confusion_matrix:{confusion_matrix(y_pred,y_test)}')


# 5. Ensemble (Voting Classifier)
ensemble=VotingClassifier(estimators=models,voting='soft')
ensemble.fit(X_train,y_train)
y_pred=ensemble.predict(X_test)
print('\nEnsemble model')
print(f'Ensemble Accuracy:{accuracy_score(y_test,y_pred)}')
print(f'Ensemble Classification Report:{classification_report(y_test,y_pred)}')
from sklearn.model_selection import  cross_val_score, GridSearchCV


# 6. Cross-validation
scores=cross_val_score(rf,X,y,cv=5)
print("\n random forest cross validation accuracy:",np.mean(scores))
scores=cross_val_score(svm,X,y,cv=5)
print("\n SVM cross validation accuracy:",np.mean(scores))
scores=cross_val_score(dt,X,y,cv=5)
print("\n Decision Tree cross validation accuracy:",np.mean(scores))
scores=cross_val_score(ensemble,X,y,cv=5)
print("\n Ensemble cross validation accuracy:",np.mean(scores))


# 7. Hyperparameter Tuning
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}            #GridSearchCV--->
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("\nBest SVM Parameters:", grid.best_params_)
print("Best SVM Score:", grid.best_score_)
param_grid = {
    'n_estimators': [50, 100, 200],     # number of trees
    'max_depth': [None, 5, 10],         # tree depth
    'min_samples_split': [2, 5, 10]     # minimum samples to split
}


grid=GridSearchCV(RandomForestClassifier(),param_grid,cv=5)
grid.fit(X_train,y_train)
print("\n Best rf classifier:",grid.best_params_)
print("Best randomforest Score:",grid.best_score_)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}

grid=GridSearchCV(DecisionTreeClassifier(),param_grid,cv=5)
grid.fit(X_train,y_train)
print("\n Best rf classifier:",grid.best_params_)
print("Best decisiontree Score:",grid.best_score_)

