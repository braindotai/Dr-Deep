from mxnet import nd
from heartattack import model
from dataset import testing_data

def accuracy(predictions, targets):
	# predictions = nd.argmax(predictions, 1)
	# return nd.mean(nd.equal(predictions, targets)).asscalar() * 100
	predictions = nd.where(predictions > 0.5, nd.ones_like(predictions), nd.zeros_like(predictions))
	return 100.0 - nd.mean(nd.abs(predictions - targets)).asscalar() * 100

predictions = model.predict(testing_data['features'])
print('Test Accuracy:', accuracy(predictions, testing_data['labels']))

for predictions, label in zip(predictions, testing_data['labels']):
	print('-'*50)
	print('Predictions:-')
	print(predictions.asnumpy())
	print('Labels:-')
	print(label.asnumpy())


