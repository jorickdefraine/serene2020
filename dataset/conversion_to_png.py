import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

train = open("train.bin", "rb")
test = open("test.bin", "rb")

train_num = list(train.read())
test_num = list(test.read())

def label_class(label):
    association = ['0','1','2','3','4','5','6','7','8','9','A','D','G','H','M','N','O','X','unknown','space']
    return association[label]


for i in range (112861):
    image = np.array(train_num[i*617+1:(i+1)*617])
    image.shape = (28,22)
    image = np.pad(image,((2,2),(5,5))).astype(np.uint8)
    label = train_num[i*617]
    classe = label_class(label)
    img = Image.fromarray(image)
    img.save('train/'+classe+'/'+str(i)+'.png')
    
for i in range (12540):
    image = np.array(test_num[i*617+1:(i+1)*617])
    image.shape = (28,22)
    image = np.pad(image, ((2,2),(5,5))).astype(np.uint8)
    label = test_num[i*617]
    classe = label_class(label)
    img = Image.fromarray(image)
    img.save('test/'+classe+'/'+str(i)+'.png')
