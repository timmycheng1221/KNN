import numpy as np
import math
import matplotlib.pyplot as plt

#Knn classifier module
class Knn:
    def __init__(self):
        self.traindata = False
        self.vectorsize = False
        self.count = {}
        self.model = False
        self.testresult = False
    #tag is the list contains all class name in traindata
    def train(self, traindata, tag, k):
        self.model = []
        self.traindata = traindata
        self.vectorsize = len(traindata[0])
        for item1 in traindata:
            distlist = []
            for i in tag:
                self.count.update({i: 0})
            for item2 in traindata:
                distance = 0
                for j in range(0, self.vectorsize - 1):
                    distance += (float(item1[j]) - float(item2[j])) ** 2
                distance = math.sqrt(distance)
                distlist.append([distance, item2[self.vectorsize - 1]])
            distlist.sort()
            klist = distlist[:k]
            for dist in klist:
                self.count[dist[1]] += 1 / (1 + dist[0])
            self.model.append(max(self.count, key=self.count.get))
        return self.model

    def test(self, testdata, tag, k):
        self.testresult = []
        for item1 in testdata:
            distlist = []
            for i in tag:
                self.count.update({i: 0})
            for j in range(0, len(self.traindata)):
                distance = 0
                for l in range(0, self.vectorsize - 1):
                    distance += (float(item1[l]) - float(self.traindata[j][l])) ** 2
                distance = math.sqrt(distance)
                distlist.append([distance, self.model[j]])
            distlist.sort()
            klist = distlist[:k]
            for dist in klist:
                self.count[dist[1]] += 1 / (1 + dist[0])
            self.testresult.append(max(self.count, key=self.count.get))
        return self.testresult

#read data file and spilt every line by ','
file = open('iris.data', 'r')
list = []
for line in file:
    line = line.replace('\n', '')
    list.append(line.split(','))

#Separate the data into training (50%) and test (50%) datasets
train = list[:25] + list[50:75] + list[100:125]
test = list[25:50] + list[75:100] + list[125:150]

#taglist is the list contains all class name in traindata
taglist = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

#train_errlist and test_errlist store error for every k based on knn
train_errlist = []
test_errlist = []

#train_acclist and test_acclist store accuracy for every k based on knn
train_acclist = []
test_acclist = []

for k in range(1, 21):
    knn = Knn()
    sum = 0
    knn.train(train, taglist, k)
    for i in range(0, len(train)):
        if train[i][4] == knn.model[i]:
            sum += 1
    train_errlist.append(75 - sum)
    train_acclist.append(float(sum / 75.0))

    sum = 0
    knn.test(test, taglist, k)
    for i in range(0, len(test)):
        if test[i][4] == knn.testresult[i]:
            sum += 1
    test_errlist.append(75 - sum)
    test_acclist.append(float(sum / 75.0))

train_err = np.array(train_errlist)
test_err = np.array(test_errlist)
train_acc = np.array(train_acclist)
test_acc = np.array(test_acclist)

#draw the k vs error plot
x = np.arange(1, 21)
plt.figure(1)
plt.title('KNN classifier Learning curve - k vs error')
plt.plot(x, train_err, 'g-s', x, test_err, 'r-s')
plt.xticks([])
plt.ylim(-1, 10)
plt.ylabel('Error')

#draw the k vs train error and test accuracy table
cell_text = []
cell_text.append(['%d' % x for x in train_errlist])
cell_text.append(['%d' % x for x in test_errlist])
columns = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')
rows = ['train', 'test']
colors = ['g', 'r']
table = plt.table(cellText=cell_text,
                  rowLabels=rows,
                  rowColours=colors,
                  colLabels=columns,
                  loc='bottom')
table.set_fontsize(10)

#draw the k vs accuracy plot
x = np.arange(1, 21)
plt.figure(2)
plt.title('KNN classifier Learning curve - k vs accuracy')
plt.plot(x, train_acc, 'g-s', x, test_acc, 'r-s')
plt.xticks([])
plt.ylim(0.5, 1.20)
plt.ylabel('Accuracy')

#draw the k vs train accuracy and test accuracy table
cell_text = []
cell_text.append(['%1.2f' % x for x in train_acclist])
cell_text.append(['%1.2f' % x for x in test_acclist])
columns = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20')
rows = ['train', 'test']
colors = ['g', 'r']
table = plt.table(cellText=cell_text,
                  rowLabels=rows,
                  rowColours=colors,
                  colLabels=columns,
                  loc='bottom')
table.set_fontsize(10)
plt.show()

file.close()
