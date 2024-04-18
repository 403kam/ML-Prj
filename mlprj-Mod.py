#Konnor Mascara CIS662 Project
#Most code modified from given guide on sclearn website (scikit-learn.org)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier #BEST model
#from sklearn.ensemble import RandomForestClassifier #old model
#from sklearn.neighbors import KNeighborsClassifier #New model
#from sklearn.linear_model import LogisticRegression #New model
#from sklearn.ensemble import GradientBoostingClassifier #New model
#from sklearn.naive_bayes import GaussianNB #new model

#Step 1: Load file (Dataset 1)
#Step 2: X is data, Y is answer
#Step 3: Get encoder to change all strings to int
df1 = pd.read_csv(".//csv_files//1.csv", header=None)
X1 = df1.iloc[:, :-1]
y1 = df1.iloc[:, -1]
label_encoders1 = {}
for col in X1.select_dtypes(include=['object']).columns:
    label_encoders1[col] = LabelEncoder()
    X1[col] = label_encoders1[col].fit_transform(X1[col])

#Dataset 2
df2 = pd.read_csv(".//csv_files//2.csv", header=None)
X2 = df2.iloc[:, :-1]
y2 = df2.iloc[:, -1]
label_encoders2 = {}
for col in X2.select_dtypes(include=['object']).columns:
    label_encoders2[col] = LabelEncoder()
    X2[col] = label_encoders2[col].fit_transform(X2[col])

#Dataset 10
df10 = pd.read_csv(".//csv_files//10.csv", header=None)
X10 = df10.iloc[:, :-1]
y10 = df10.iloc[:, -1]
label_encoders10 = {}
for col in X10.select_dtypes(include=['object']).columns:
    label_encoders10[col] = LabelEncoder()
    X10[col] = label_encoders10[col].fit_transform(X10[col])

#non malicous dataset (A)
dfA = pd.read_csv(".//csv_files//A.csv", header=None)
XA = dfA.iloc[:, :-1]
yA = dfA.iloc[:, -1]
label_encodersA = {}
for col in XA.select_dtypes(include=['object']).columns:
    label_encodersA[col] = LabelEncoder()
    XA[col] = label_encodersA[col].fit_transform(XA[col])

#Split data train and test (Some are higher test due to size of file, takes a minute)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.99, random_state=42)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.8, random_state=42)
X_train10, X_test10, y_train10, y_test10 = train_test_split(X10, y10, test_size=0.8, random_state=42)
X_trainA, X_testA, y_trainA, y_testA = train_test_split(XA, yA, test_size=0.99, random_state=42)

#Put data together for training (Maybe testing later)
X_trainD = pd.concat([X_train1, X_train2, X_train10], axis=0, ignore_index=True)
y_trainD = pd.concat([y_train1, y_train2, y_train10], axis=0, ignore_index=True)

#Get model and train
#model = RandomForestClassifier() #old model
#model = KNeighborsClassifier() #new model
#model = LogisticRegression() #new model
#model = GradientBoostingClassifier() #new model
model = AdaBoostClassifier() #BEST
#model = GaussianNB() #new model
model.fit(X_trainD, y_trainD)

#Test with 1st data (Lower due to train set being small)
y1_pred1 = model.predict(X_test1)
accuracy1 = accuracy_score(y_test1, y1_pred1)
print(f"Accuracy: {accuracy1}")

#Test with 2nd data
y_pred2 = model.predict(X_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)
print(f"Accuracy: {accuracy2}")

#Test with 10th data
y_pred10 = model.predict(X_test10)
accuracy10 = accuracy_score(y_test10, y_pred10)
print(f"Accuracy: {accuracy10}")

#Test with A's data
y_predA = model.predict(X_testA)
accuracyA = accuracy_score(y_testA, y_predA)
print(f"Accuracy: {accuracyA}")

#Added display for 1st data
#Only checking one data set for comparision
#More graphs could be used for displaying accuracy
cm = confusion_matrix(y_test1, y1_pred1)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()