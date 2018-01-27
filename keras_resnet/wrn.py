import keras
import keras.backend as K 
from keras.models import Model
from keras.layers import Dense, Dropout, Add, Input, BatchNormalization, Activation
from keras.layers import  Conv2D, MaxPooling2D, AveragePooling2D, Flatten
from keras.regularizers import l2

def main_block(x, filters, n, strides, dropout):
	# Normal part
	x_res = Conv2D(filters, (3,3), strides=strides, padding="same")(x)# , kernel_regularizer=l2(5e-4)
	x_res = BatchNormalization()(x_res)
	x_res = Activation('relu')(x_res)
	x_res = Conv2D(filters, (3,3), padding="same")(x_res)
	# Alternative branch
	x = Conv2D(filters, (1,1), strides=strides)(x)
	# Merge Branches
	x = Add()([x_res, x])

	for i in range(n-1):
		# Residual conection
		x_res = BatchNormalization()(x)
		x_res = Activation('relu')(x_res)
		x_res = Conv2D(filters, (3,3), padding="same")(x_res)
		# Apply dropout if given
		if dropout: x_res = Dropout(dropout)(x)
		# Second part
		x_res = BatchNormalization()(x_res)
		x_res = Activation('relu')(x_res)
		x_res = Conv2D(filters, (3,3), padding="same")(x_res)
		# Merge branches
		x = Add()([x, x_res])

	# Inter block part
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x

def build_model(input_dims, output_dim, n, k, act= "relu", dropout=None):
	""" Builds the model. Params:
			- n: number of layers. WRNs are of the form WRN-N-K
				 It must satisfy that (N-4)%6 = 0
			- k: Widening factor. WRNs are of the form WRN-N-K
				 It must satisfy that K%2 = 0
			- input_dims: input dimensions for the model
			- output_dim: output dimensions for the model
			- dropout: dropout rate - default=0 (not recomended >0.3)
			- act: activation function - default=relu. Build your custom
				   one with keras.backend (ex: swish, e-swish)
	"""
	# Ensure n & k are correct
	assert (n-4)%6 == 0
	assert k%2 == 0
	n = (n-4)//6 
	# This returns a tensor input to the model
	inputs = Input(shape=(input_dims))

	# Head of the model
	x = Conv2D(16, (3,3), padding="same")(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# 3 Blocks (normal-residual)
	x = main_block(x, 16*k, n, (1,1), dropout) # 0
	x = main_block(x, 32*k, n, (2,2), dropout) # 1
	x = main_block(x, 64*k, n, (2,2), dropout) # 2
			
	# Final part of the model
	x = AveragePooling2D((8,8))(x)
	x = Flatten()(x)
	outputs = Dense(output_dim, activation="softmax")(x)

	model = Model(inputs=inputs, outputs=outputs)
	return model


if __name__ == "__main__":
	model = build_model((32,32,3), 10, 22, 8)
	model.summary()