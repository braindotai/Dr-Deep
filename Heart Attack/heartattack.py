import mxnet as mx

moments = mx.nd.load('./dataset/moments.info')
network = mx.gluon.nn.SymbolBlock.imports('heartattack-symbol.json', ['data'], 'heartattack-0000.params', ctx = mx.cpu())
print('model is loaded')
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