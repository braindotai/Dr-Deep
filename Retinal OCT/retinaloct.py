import mxnet as mx

network = mx.gluon.nn.SymbolBlock.imports('retinaloct-symbol.json', ['data'], 'retinaloct-0000.params', ctx = mx.cpu())
print('model is loaded')

class model:
	def predict(inputs):
		return network(inputs)