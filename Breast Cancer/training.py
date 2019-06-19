import mxnet as mx
from mxnet import nd, autograd
from dataset import training_data, testing_data

def load_model():
	from breastcancer import model
	return model

load = str(input('Wanna load the model: '))
if load == 'y' or load == 'yes':
	model = load_model()
else:
	network = mx.gluon.nn.HybridSequential()
	network.add(mx.gluon.nn.Dense(512, 'relu'))
	network.add(mx.gluon.nn.Dense(256, 'relu'))
	network.add(mx.gluon.nn.Dense(256, 'relu'))
	network.add(mx.gluon.nn.Dense(128, 'relu'))
	network.add(mx.gluon.nn.Dense(1, 'sigmoid'))
	network.collect_params().initialize(ctx = mx.cpu(), init = mx.init.Xavier())
	network.hybridize()

	moments = mx.nd.load('./dataset/moments.info')
	class model:
		def predict(inputs, training = False):
			if training: return network(inputs)
			else:
				inputs = (inputs - moments['min']) / (moments['max'] - moments['min'])
				return network(inputs)

		def collect_params():
			return network.collect_params()

		def export(name, epoch = 0):
			network.export(name, epoch = epoch)

def loss(predictions, targets):
	return -nd.mean((targets * nd.log(predictions)) + ((1 - targets) * nd.log(1 - predictions)))
	# return -nd.mean((targets * nd.log(nd.softmax(predictions))) + ((1 - targets) * nd.log(1 - nd.softmax(predictions))))

def accuracy(predictions, targets):
	# predictions = nd.argmax(predictions, 1)
	# return nd.mean(nd.equal(predictions, targets)).asscalar() * 100
	predictions = nd.where(predictions > 0.5, nd.ones_like(predictions), nd.zeros_like(predictions))
	return 100 - nd.mean(nd.abs(predictions - targets)).asscalar() * 100

optimizer = mx.gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.0001})

epochs = 50
batch_size = 128

dataset = mx.io.NDArrayIter(data = training_data['features'], label = training_data['labels'], batch_size = batch_size, shuffle = True)

for epoch in range(epochs):
	dataset.reset()
	cum_loss = 0.0
	for batch in dataset:
		with autograd.record():
			predictions = model.predict(batch.data[0], True)
			cost = loss(predictions, batch.label[0])
		cost.backward()
		optimizer.step(16)
		cum_loss += cost.asscalar()
	acc = accuracy(model.predict(training_data['features'], True), training_data['labels'])
	print('Epoch:', epoch, 'Cost:', cum_loss/(training_data['features'].shape[0]//batch_size), 'Accucacy:', acc)
	print('Test Accuracy:', accuracy(model.predict(testing_data['features']), testing_data['labels']))

def save_model():
	print('Model is saved')
	model.export('breastcancer', epoch = 0)

save = str(input('Wanna save the model: '))
if save == 'y' or save == 'yes':
	save_model()

load = str(input('Wanna load the model: '))
if load == 'y' or load == 'yes':
	reloaded_model = load_model()
	print('Test Accuracy:', accuracy(reloaded_model.predict(testing_data['features']), testing_data['labels']))

