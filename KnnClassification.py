import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pickle
data = pd.read_csv("car.data")
print(data.head())

#precprocesssing to convert non numeric data numeric retuns numpy array
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
print(type(door))
predict = "class"

X = list(zip(buying, maint, lug_boot, door, persons, safety, door))
y = list(cls)

X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
best = 0

for k in range(1, len(X_train)):
    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test,y_test)
    if acc > best:
        best = acc
        print("Accuracy : ", acc)
        with open("Car-Model.pickle", "wb") as f:
            pickle.dump(knn, f)


pickle_in = open("Car-Model.pickle", "rb")
knn = pickle.load(pickle_in)