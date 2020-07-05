import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

test_file = open("test.bin","rb")
train_file = open("train.bin","rb")

test_num = list(test_file.read())
train_num = list(train_file.read())

#print(test_num[0::617])
#print(train_num[0::617])

test = np.array(test_num[0::617])
train = np.array(train_num[0::617])

test_hist = np.histogram(test, bins=np.linspace(0,19,21))
train_hist = np.histogram(train, bins=np.linspace(0,19,21))

print("distribution of the test dataset :\n ", test_hist[0])
print("distribution of the train dataset :\n ", train_hist[0])

print("sum test : ", sum(test_hist[0]))
print("sum train : ", sum(train_hist[0]))
plt.hist(test_hist[0])
plt.title("Distribution of the labels in test dataset")


plt.bar(["Classe" + str (i) for i in range(19)],[test_hist[0][i] for i in range (19)])
plt.bar(["Classe" + str (i) for i in range(19)],[train_hist[0][i] for i in range(19)])
plt.show()

#image = mpimg.imread(test[0])
#plt.imshow(image)


test_file.close()
train_file.close()
