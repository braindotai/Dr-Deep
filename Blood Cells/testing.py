from mxnet import nd
from bloodcells import model
import os, cv2
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt

def accuracy(predictions, targets):
	predictions = nd.argmax(predictions, 1)
	return nd.mean(nd.equal(predictions, targets)).asscalar() * 100

x = []
y = []
z = []
names = []
for wbc_type in os.listdir('dataset'):
    if 'neutro' in wbc_type:
        label = 0
        label2 = 0
    elif 'eosin' in wbc_type:
        label = 1
        label2 = 0
    elif 'mono' in wbc_type:
        label = 2  
        label2 = 1
    elif 'lympho' in wbc_type:
        label = 3
        label2 = 1
    img = mx.image.imread('dataset/' + wbc_type)
    img = mx.image.imresize(img, 80, 60)
    names.append(wbc_type)
    x.append(nd.moveaxis(img, 2, 0).asnumpy())
    y.append(label)
    z.append(label2)

x = nd.array(x)
y = nd.array(y)
z = nd.array(z)
x /= 255.0

print(x.shape)
print(y.shape)

plt.imshow(mx.nd.moveaxis(x[0], 0, 2).asnumpy())
plt.show()

predictions_y, predictions_z = model.predict(x)
print('Test Accuracy-y:', accuracy(predictions_y, y))
print('Test Accuracy-z:', accuracy(predictions_z, z))

for i, (name, predictiony, predictionz, labely, labelz) in enumerate(zip(names, predictions_y, predictions_z, y, z)):
    print(i + 1)
    print(name)
    print('predictiony: ', predictiony.asnumpy().argmax())
    print('labely: ', labely.asnumpy())
    print('predictionz: ', predictionz.asnumpy().argmax())
    print('labelz: ', labelz.asnumpy())


