import pandas as pd
import mxnet as mx
from mxnet import nd

dataset = pd.read_csv("./Dataset/heart.csv")
dataset = dataset.sample(frac = 1).reset_index(drop = True)

features = nd.array(dataset.drop(['target'], axis = 1))
labels = nd.array(dataset.target.values)

train_x = features[:297]
test_x = features[297:]

# mean = train_x.mean(0)
# std = (train_x - train_x.mean(0)).square().mean(0).sqrt()

mini = train_x.min(axis = 0)
maxi = train_x.max(axis = 0)

train_x = (train_x - mini) / (maxi - mini)

train_y = labels[:297].reshape((-1, 1))
test_y = labels[297:].reshape((-1, 1))

print('train_x:', train_x.shape)
print('train_y:', train_y.shape)

print('test_x:', test_x.shape)
print('test_y:', test_y.shape)

mx.nd.save('./dataset/training_data', {'features': train_x, 'labels': train_y})
mx.nd.save('./dataset/testing_data', {'features': test_x, 'labels': test_y})
mx.nd.save('./dataset/moments.info', {'min': mini, 'max': maxi})

# print(train_x[:5])
# print(test_x[:5])