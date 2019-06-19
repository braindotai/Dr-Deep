from mxnet import nd
from retinaloct import model
import mxnet as mx
import os
import matplotlib.pyplot as plt

def accuracy(predictions, targets):
	predictions = nd.argmax(predictions, 1)
	return nd.mean(nd.equal(predictions, targets)).asscalar() * 100


x = []
y = []
names = []

for wbc_type in os.listdir('dataset'):
    if 'CNV' in wbc_type:
        label = 0
    elif 'DME' in wbc_type:
        label = 1
    elif 'DRUSEN' in wbc_type:
        label = 2
    elif 'NORMAL' in wbc_type:
        label = 3
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
    print('prediction: ', prediction.asnumpy())
    print('label: ', label.asnumpy())

print('Accuracy:', accuracy(predictions, y))



