import mxnet as mx
from mxnet import nd, autograd
from dataset import training_data, testing_data

def load_model():
	from liver import model
	return model

load = str(input('Wanna load the model: '))
if load == 'y' or load == 'yes':
	model = load_model()
else:
	network = mx.gluon.nn.HybridSequential()
	network.add(mx.gluon.nn.Dense(512, 'softrelu'))
	network.add(mx.gluon.nn.BatchNorm())
	network.add(mx.gluon.nn.Dense(512, 'softrelu'))
	network.add(mx.gluon.nn.BatchNorm())
	network.add(mx.gluon.nn.Dropout(0.25))
	network.add(mx.gluon.nn.Dense(512, 'softrelu'))
	network.add(mx.gluon.nn.BatchNorm())
	network.add(mx.gluon.nn.Dense(1))
	network.collect_params().initialize(ctx = mx.cpu(), init = mx.init.Xavier())
	network.hybridize()

	moments = mx.nd.load('./dataset/moments')
	class model:
		def predict(inputs, training = False):
			if training: return network(inputs)
			else:
				# inputs = (inputs - moments['min'])/(moments['max'] - moments['min'])
				return network(inputs)

		def collect_params():
			return network.collect_params()

		def export(name, epoch = 0):
			network.export(name, epoch = epoch)

def loss(predictions, targets):
	# return -nd.mean(targets * nd.log(predictions))
	# return -nd.mean((targets * nd.log(predictions)) + ((1 - targets) * nd.log(1 - predictions)))
	return nd.mean(nd.square(predictions - targets))
	# return -nd.mean((targets * nd.log(nd.softmax(predictions))) + ((1 - targets) * nd.log(1 - nd.softmax(predictions))))

def accuracy(predictions, targets):
	# predictions = nd.argmax(predictions, 1)
	# targets = nd.argmax(targets, 1)
	# return nd.mean(nd.equal(predictions, targets)).asscalar() * 100
	predictions = nd.where(predictions > 0.5, nd.ones_like(predictions), nd.zeros_like(predictions))
	return 100 - nd.mean(nd.abs(predictions - targets)).asscalar() * 100

optimizer = mx.gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.0001})

epochs = 1000
batch_size = 32

dataset = mx.io.NDArrayIter(data = training_data['features'], label = training_data['labels'], batch_size = batch_size, shuffle = True)

for epoch in range(epochs):
	dataset.reset()
	cum_loss = 0.0
	for batch in dataset:
		with autograd.record():
			predictions = model.predict(batch.data[0], True)
			cost = loss(predictions, batch.label[0])
		cost.backward()
		optimizer.step(batch_size)
		cum_loss += cost.asscalar()
	# acc = accuracy(model.predict(training_data['features'], True), training_data['labels'])
	print('Epoch:', epoch, 'Cost:', cum_loss/(training_data['features'].shape[0]//batch_size))
	print('Test Loss:', loss(model.predict(testing_data['features']), testing_data['labels']).asscalar())

def save_model():
	print('Model is saved')
	model.export('liver', epoch = 0)

save = str(input('Wanna save the model: '))
if save == 'y' or save == 'yes':
	save_model()

load = str(input('Wanna load the model: '))
if load == 'y' or load == 'yes':
	reloaded_model = load_model()
	print('Test Loss:', loss(reloaded_model.predict(testing_data['features']), testing_data['labels']).asscalar())
	# print('Test Accuracy:', accuracy(reloaded_model.predict(testing_data['features']), testing_data['labels']))

# model = mx.gluon.nn.HybridSequential()
# model.add(mx.gluon.nn.Conv2D(32, (3,3), activation = 'relu'))
# model.add(mx.gluon.nn.MaxPool2D(2,2))
# model.add(mx.gluon.nn.BatchNorm(axis = -1))
# model.add(mx.gluon.nn.Dropout(0.2))

# model.add(mx.gluon.nn.Conv2D(32, (3,3), activation = 'relu'))
# model.add(mx.gluon.nn.MaxPool2D(2,2))
# model.add(mx.gluon.nn.BatchNorm(axis = -1))
# model.add(mx.gluon.nn.Dropout(0.2))

# model.add(mx.gluon.nn.Conv2D(32, (3,3), activation = 'relu'))
# model.add(mx.gluon.nn.MaxPool2D(2,2))
# model.add(mx.gluon.nn.BatchNorm(axis = -1))
# model.add(mx.gluon.nn.Dropout(0.2))

# model.add(mx.gluon.nn.Flatten())

# model.add(mx.gluon.nn.Dense(512, activation = 'relu'))
# model.add(mx.gluon.nn.BatchNorm(axis = -1))
# model.add(mx.gluon.nn.Dropout(0.2))
# model.add(mx.gluon.nn.Dense(1, 'sigmoid'))

# model.collect_params().initialize(ctx = mx.gpu())
# model.hybridize(static_alloc = True, static_shape = True)