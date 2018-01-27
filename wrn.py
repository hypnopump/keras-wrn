import keras
import keras.backend as K 
from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, BatchNormalization, Activation
from keras.layers import  Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2

def first_block(x):
	x = Conv2D(16, (3,3), padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def normal_block(x, k, n_block, j): # Main branch - if 0, strides=1, else=2
	if n_block==0 or j>0:
		x_res = Conv2D(16*k*(2**n_block), (3,3), padding="same")(x) # , kernel_regularizer=l2(5e-4)
	else:
		x_res = Conv2D(16*k*(2**n_block), (3,3), strides=(2,2), padding="same")(x)
	x_res = BatchNormalization()(x_res)
	x_res = Activation('relu')(x_res)
	x_res = Conv2D(16*k*(2**n_block), (3,3), padding="same")(x_res)
	# Alternative branch
	if n_block==0 or j>0:
		x = Conv2D(16*k*(2**n_block), (1,1))(x)
	else:
		x = Conv2D(16*k*(2**n_block), (1,1), strides=(2,2))(x)
	# Merge Branches
	x = Add()([x_res, x])
	return x

def residual_block(x, k, n_block, dropout):
	""" Applies 2x:
			- BatchNorm
			- Relu
			- Conv(3x3)
	"""
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(16*k*(2**n_block), (3,3), padding="same")(x)

	if dropout: x = Dropout(dropout)(x)

	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(16*k*(2**n_block), (3,3), padding="same")(x)

	return x

def build_model(input_dims, output_dim, n, k, dropout=None):
	""" Builds the model. Params:
			- n: number of layers. WRNs are of the form WRN-N-K
				 It must satisfy that (N-4)%6 = 0
			- k: Widening factor. WRNs are of the form WRN-N-K
				 It must satisfy that K%2 = 0
			- input_dims: input dimensions for the model
	"""
	# Ensure n & k are correct
	assert (n-4)%6 == 0
	assert k%2 == 0
	# This returns a tensor input to the model
	inputs = Input(shape=(input_dims))

	# Head of the model
	x = first_block(inputs)

	# Rest of Blocks (normal-residual)
	for i in range(3):
		for j in range((n-4)//6-1):
			x = normal_block(x, k, i, j)
			x_res = residual_block(x, k, i, dropout)
			x = Add()([x, x_res])
		# Inter block part
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

	x = AveragePooling2D((8,8))(x)
	x = Flatten()(x)
	outputs = Dense(output_dim, activation="softmax")(x)

	model = Model(inputs=inputs, outputs=outputs)
	return model


if __name__ == "__main__":
	model = build_model((32,32,3), 10, 16, 8)
	model.summary()