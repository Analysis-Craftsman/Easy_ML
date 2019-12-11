# Taste the world's easiest machine learning - SVM
# Excludes training / test code
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier as rfc
import matplotlib.pyplot as plt

dataset = [ 
    # Restaurant size, Restaurant Distance from bus station = 1 month sales
    [10, 80, 469],
    [8, 0, 366],
    [8, 200, 371],
    [5, 200, 208],
    [7, 300, 246],
    [8, 230, 297],
    [7, 40, 363],
    [9, 0, 436],
    [6, 330, 198],
    [9, 180, 364],
 ]

train_data_x = []
train_data_y = []

for input_train_data in range( len(dataset) ) :
    train_data_x.append( dataset[input_train_data][:-1] )
    train_data_y.append( dataset[input_train_data][-1] )

print(train_data_x)
print(train_data_y)

learning = svm.SVC()
learning.fit(train_data_x, train_data_y)

predict = learning.predict(train_data_x)
print("Original sales : ", train_data_y)
print("Forecast sales : ", predict)

# accuracy
ac_scores = metrics.accuracy_score(train_data_y, predict)
print("accuracy :", int(ac_scores*100), "%")

# reports
reports = metrics.classification_report(train_data_y, predict)
print("Report :")
print(reports)

# visual
plt.title("Original sales(RED) VS Forecast sales(BLUE)")
plt.xlabel("count")
plt.ylabel("sales")
plt.plot(train_data_y, "r-")
plt.plot(predict, "b*")

plt.show()

# Taste the world's easiest machine learning - Random Forest
from sklearn import svm, metrics
import matplotlib.pyplot as plt

dataset = [ 
    # Restaurant size, Restaurant Distance from bus station = 1 month sales
    [10, 80, 469],
    [8, 0, 366],
    [8, 200, 371],
    [5, 200, 208],
    [7, 300, 246],
    [8, 230, 297],
    [7, 40, 363],
    [9, 0, 436],
    [6, 330, 198],
    [9, 180, 364],
 ]

train_data_x = []
train_data_y = []

for input_train_data in range( len(dataset) ) :
    train_data_x.append( dataset[input_train_data][:-1] )
    train_data_y.append( dataset[input_train_data][-1] )

print(train_data_x)
print(train_data_y)

learning = rfc()
learning.fit(train_data_x, train_data_y)

predict = learning.predict(train_data_x)
print("Original sales : ", train_data_y)
print("Forecast sales : ", predict)

# accuracy
ac_scores = metrics.accuracy_score(train_data_y, predict)
print("accuracy :", int(ac_scores*100), "%")

# reports
reports = metrics.classification_report(train_data_y, predict)
print("Report :")
print(reports)

# visual
plt.title("Original sales(RED) VS Forecast sales(BLUE)")
plt.xlabel("count")
plt.ylabel("sales")
plt.plot(train_data_y, "r-")
plt.plot(predict, "b*")

plt.show()
