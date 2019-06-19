import mxnet as mx

network_y = mx.gluon.nn.SymbolBlock.imports('bloodcells-y-symbol.json', ['data'], 'bloodcells-y-0000.params', ctx = mx.cpu())
network_z = mx.gluon.nn.SymbolBlock.imports('bloodcells-z-symbol.json', ['data'], 'bloodcells-z-0000.params', ctx = mx.cpu())
print('model is loaded')

class model:
	def predict(inputs):
		return network_y(inputs), network_z(inputs)