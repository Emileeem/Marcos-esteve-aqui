import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier #para o desicion tree
from joblib import dump
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, recall_score #Matriz de confusão e testes
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, HistGradientBoostingClassifier
from sklearn.datasets import make_classification, load_iris
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
df = pd.read_csv('phising.csv')
df.drop(['id'], axis=1, inplace=True)
df = df.dropna()
df.drop_duplicates(inplace = True)

# Histograma dos caracteres

# plt.hist(df['UrlLength'])
# plt.xlabel('Quantidade de Caracteres na Url')
# plt.ylabel('Quantidade de Url desse tamanho')
# plt.show()

#Boxplot de Numero de pontos contidos na url

# stud_bplt = df.boxplot(column=['NumDots'])
# stud_bplt.plot()
# plt.show()

#Scatter Plot

# df.plot.scatter(x = 'UrlLength', y = 'NumDash', s = 3)
# df.plot.scatter(x = 'NumDots', y = 'NumDash', s = 3)
# plt.show()

##Usando o Decision Tree

Y = df['Phising']
X = df.drop('Phising', axis=1)

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)

# dump(model, 'filename.joblib')

# Y_real = Y_train
# Y_pred = model.predict(X_train)
# train_error = mean_absolute_error(Y_real, Y_pred)
# Y_real = Y_test
# Y_pred = model.predict(X_test)
# test_error = mean_absolute_error(Y_real, Y_pred)
# print(Y_pred)

# print(train_error, test_error)

# model.predict([[2,41,0,0,0,0,0,0,3,41,0]])

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 42)

# model = GridSearchCV(
#     RandomForestClassifier(n_estimators=220),
#     {
#         'min_samples_split': [0, 1, 2, 4, 5, 7, 10 ],
#         'max_depth': [48, 50, 53, 57, 60, 64, 67], 
        
#     },
#     scoring='recall',
#     n_jobs=-1 
# )

# model.fit(X_train, Y_train)

# dump(model, 'model.pkl')

# y_predict = model.predict(X_test)

# print("Recall score:", model.best_score_)
# print("Melhores Parametros:", model.best_params_)

# mat = confusion_matrix(Y_test, y_predict)
# display = ConfusionMatrixDisplay(confusion_matrix=mat)
# display.yticks_rotation = 'vertical'
# display.plot()

# plt.show()


# Votação de melhores algoritmos 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 42)

svc = SVC(probability=True, gamma="auto")

bagging = GridSearchCV(
    SelfTrainingClassifier(svc),
    {
        'max_iter': [1, 5, 10, 15, 30],
        'k_best': [15, 30, 6, 9, 11]
    },
    scoring='recall',
    n_jobs=-1
)

rdForest =  GridSearchCV(
    RandomForestClassifier(n_estimators=220),
    {
        'min_samples_split': [0, 1, 2, 4, 5, 7, 10 ],
        'max_depth': [48, 50, 53, 57, 60, 64, 67], 
        
    },
    scoring='recall',
    n_jobs=-1 
)

histGrad = GridSearchCV(
    HistGradientBoostingClassifier(),
    {
       'max_iter': [100, 200, 50, 25],
       'max_depth': [10, 20, 30, 40, 15]
    },
    scoring='recall',
    n_jobs=-1 
)


# votate = VotingClassifier(
#     estimators=[('bagging', bagging), ('Random Forest', rdForest), ('Hist GradientBoosting Classifier', histGrad)],
#     voting='hard'
# )
 # bagging
bagging.fit(X_train, Y_train)

dump(bagging, 'modelVotacaoBagg.pkl')

# Regression
y_predict = bagging.predict(X_test)

mat = confusion_matrix(Y_test, y_predict)
display = ConfusionMatrixDisplay(confusion_matrix=mat)
display.yticks_rotation = 'vertical'
display.plot()
plt.show()

# rdForest

rdForest.fit(X_train, Y_train)

dump(rdForest, 'modelVotacaoRdForest.pkl')

y_predict = rdForest.predict(X_test)

mat = confusion_matrix(Y_test, y_predict)
display = ConfusionMatrixDisplay(confusion_matrix=mat)
display.yticks_rotation = 'vertical'
display.plot()
plt.show()

# histGrad

histGrad.fit(X_train, Y_train)

dump(histGrad, 'modelVotacaohistGrad.pkl')

y_predict = histGrad.predict(X_test)

mat = confusion_matrix(Y_test, y_predict)
display = ConfusionMatrixDisplay(confusion_matrix=mat)
display.yticks_rotation = 'vertical'
display.plot()
plt.show()