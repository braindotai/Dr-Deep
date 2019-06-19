from mxnet import nd
from skincancer import model
import os, cv2
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt

def accuracy(predictions, targets):
	predictions = nd.argmax(predictions, 1)
	return nd.mean(nd.equal(predictions, targets)).asscalar() * 100


x = []
y = []
names = []

['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

for wbc_type in os.listdir('dataset'):
    if 'akiec' in wbc_type:
        label = 0
    elif 'bcc' in wbc_type:
        label = 1
    elif 'bkl' in wbc_type:
        label = 2
    elif 'df' in wbc_type:
        label = 3
    elif 'mel' in wbc_type:
        label = 4
    elif 'nv' in wbc_type:
        label = 5
    elif 'vasc' in wbc_type:
        label = 6
    img = mx.image.imread('dataset/' + wbc_type).astype('float32')
    img = mx.image.imresize(img, 256, 256)
    names.append(wbc_type)
    x.append(nd.moveaxis(img, 2, 0).asnumpy())
    y.append(label)

x = nd.array(x)
y = nd.array(y)

x /= 255.0

print(x.shape)
print(y.shape)

plt.imshow(mx.nd.moveaxis(x[0], 0, 2).asnumpy())
plt.show()

predictions = model.predict(x)

cc = 0
for i, (name, prediction, label) in enumerate(zip(names, predictions, y)):
    print(i + 1)
    print(name)
    print('top three:', prediction.argsort()[-3:].asnumpy())
    print('prediction: ', prediction.argmax(0).asnumpy())
    print('label: ', label.asnumpy())
    if label.asnumpy() in prediction.argsort()[-3:].asnumpy():
        cc += 1.0

print('Top1 Accuracy:', accuracy(predictions, y))
print('Top3 Accuracy:', (cc/i) * 100)


