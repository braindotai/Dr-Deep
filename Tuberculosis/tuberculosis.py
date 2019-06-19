import mxnet as mx

network = mx.gluon.nn.SymbolBlock.imports('tuberculosis-symbol.json', ['data'], 'tuberculosis-0000.params', ctx = mx.cpu())
print('model is loaded')

class model:
	def predict(inputs):
		return network(inputs)