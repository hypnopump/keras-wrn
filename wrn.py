import keras
import keras.backend as K 
from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, BatchNormalization, Activation
from keras.layers import  Conv2D, MaxPooling2D, AveragePooling2D, Flatten


def first_block(x):
	x = Conv2D(16, (3,3))(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def normal_block(x, n_block):
	# Alternative branch
	x = Conv2D(32*(2**n_block), (1,1))
	# Main branch
	x_res = Conv2D(32, (3,3))(x)
	x_res = BatchNormalization()(x)
	x_res = Activation('relu')(x)
	x_res = Conv2D(32*(2**n_block), (3,3))(x)
	# Merge Branches
	x = Add()([x_res, x])
	return x

def residual_block(x, n_block):
	""" Applies 2x:
			- BatchNorm
			- Relu
			- Conv(3x3)
	"""
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(32*(2**n_block), (3,3))

	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(32*(2**n_block), (3,3))

	return x

def build_model(n, k, input_dims):
	""" Builds the model. Params:
			- n: number of layers. WRNs are of the form WRN-N-K
				 It must satisfy that (N-4)%6 = 0
			- k: Widening factor. WRNs are of the form WRN-N-K
				 It must satisfy that K%2 = 0
			- input_dims: input dimensions for the model
	"""

	# This returns a tensor input to the model
	inputs = Input(shape=(input_dims))

	# Head of the model
	x = first_block(x)

	# Rest of Blocks (normal-residual)
	for i in n_blocks:
		x = normal_block(x, i)
		x_res = residual_block(x, i)
		x = Add()([x, x_res])
		# Inter block part
		x = BatchNormalization()(x)
		x = Activation('relu')

	x = AveragePooling2D((8,8))
	x = Flatten()(x)
	outputs = Dense(128)(x)

	model = Model(inputs=inputs, outputs=outputs)
	return model


if __name__ == "__main__":
	model = build_model(1, (32,32,3))
	model.summary()