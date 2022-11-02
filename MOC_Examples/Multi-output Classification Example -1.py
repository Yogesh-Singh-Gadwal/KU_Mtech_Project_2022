# https://www.datatechnotes.com/2020/03/multi-output-classification-with-multioutputclassifier.html

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier

x, y = make_multilabel_classification(n_samples=1050, n_features=1,
                                      n_classes=3, random_state=0)

for i in range(10): 
 print(x[i]," => ", y[i])

xtrain, xtest, ytrain, ytest=train_test_split(x, y, train_size=0.95, random_state=0)
print(len(xtest))

svc = SVC(gamma="scale")
model = MultiOutputClassifier(estimator=svc)
print(model)

model.fit(xtrain, ytrain)
print(model.score(xtrain, ytrain))

yhat = model.predict(xtest)
auc_y1 = roc_auc_score(ytest[:,0],yhat[:,0])
auc_y2 = roc_auc_score(ytest[:,1],yhat[:,1])
 
print("ROC AUC y1: %.4f, y2: %.4f" % (auc_y1, auc_y2))

cm_y1 = confusion_matrix(ytest[:,0],yhat[:,0])
cm_y2 = confusion_matrix(ytest[:,1],yhat[:,1])
 
print(cm_y1)
print(cm_y2)

cr_y1 = classification_report(ytest[:,0],yhat[:,0])
cr_y2 = classification_report(ytest[:,1],yhat[:,1])

print(cr_y1)
print(cr_y2)
