import pandas as pd
import mxnet as mx
from mxnet import nd

dataset = pd.read_csv("./Dataset/data.csv")
for i in range(10):
	dataset = dataset.sample(frac = 1).reset_index(drop = True)

features = nd.array(dataset.drop(['drinks'], axis = 1))
labels = nd.array(dataset.drinks.values).reshape((-1, 1))

print('features shape:', features.shape)
print('labels shape:', labels.shape)

train_x = features[:280]
mini = train_x.min(0)
maxi = train_x.max(0)
# train_x = (train_x - mini)/(maxi - mini)
train_y = labels[:280]

test_x = features[280:]
test_y = labels[280:]

mx.nd.save('./dataset/training_data', {'features': train_x, 'labels': train_y})
mx.nd.save('./dataset/testing_data', {'features': test_x, 'labels': test_y})
mx.nd.save('./dataset/moments', {'min': mini, 'max': maxi})

print(train_x[:5])
print(train_y[:5])

print(test_x[:5])
print(test_y[:5])
