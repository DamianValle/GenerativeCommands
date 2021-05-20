import pickle
import warnings

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers


class VectorQuantizer(K.layers.Layer):
	def __init__(self, k, **kwargs):
		super(VectorQuantizer, self).__init__(**kwargs)
		self.k = k

	def build(self, input_shape):
		self.d = int(input_shape[-1])
		rand_init = K.initializers.VarianceScaling(distribution="uniform")
		self.codebook = self.add_weight(shape=(self.k, self.d), initializer=rand_init, trainable=True)

	def call(self, inputs):
		# Map z_e of shape (b, w,, h, d) to indices in the codebook
		lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
		z_e = tf.expand_dims(inputs, -2)
		dist = tf.norm(z_e - lookup_, axis=-1)
		k_index = tf.argmin(dist, axis=-1)
		return k_index

	def sample(self, k_index):
		# Map indices array of shape (b, w, h) to actual codebook z_q
		lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
		k_index_one_hot = tf.one_hot(k_index, self.k)
		z_q = lookup_ * k_index_one_hot[..., None]
		z_q = tf.reduce_sum(z_q, axis=-2)
		return z_q

class Block(layers.Layer):
	def __init__(self, kernel, strides, filters=256, padding='valid'):
		super(Block, self).__init__()
		self.lrelu = layers.LeakyReLU()
		self.bn = layers.BatchNormalization()
		self.conv = layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides, padding=padding)

	def call(self, x):
		x = self.bn(x)
		x = self.lrelu(x)
		x = self.conv(x)
		return x


class Unblock(layers.Layer):
	def __init__(self, kernel, strides, filters=256, padding='valid', activation='sigmoid'):
		super(Unblock, self).__init__()
		if activation == 'sigmoid':
			self.activation = K.activations.sigmoid
		self.activation = layers.LeakyReLU()
		self.bn = layers.BatchNormalization()
		self.conv = layers.Conv2DTranspose(filters=filters, kernel_size=kernel, strides=strides, padding=padding)

	def call(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.activation(x)
		return x


def encoder_pass(inputs, d):
	x = inputs
	x = Block(kernel=(3, 3), strides=(1, 1))(x)
	x = Block(kernel=(3, 3), strides=(2, 2), padding='same')(x)
	x = Block(kernel=(1, 3), strides=(2, 2))(x)
	x = Block(kernel=(1, 3), strides=(1, 2))(x)
	x = Block(kernel=(1, 3), strides=(1, 1))(x)
	x = Block(kernel=(1, 3), strides=(1, 2))(x)
	x = Block(kernel=(1, 3), strides=(1, 1))(x)
	z_e = K.layers.Conv2D(filters=d, kernel_size=1, padding='valid', activation=None, strides=(1, 1), name='z_e')(x)
	return z_e


def decoder_pass(inputs):
	x = inputs
	x = Unblock(kernel=(1, 1), strides=(1, 1))(x)
	x = Unblock(kernel=(1, 3), strides=(1, 1))(x)
	x = Unblock(kernel=(1, 3), strides=(1, 1))(x)
	x = Unblock(kernel=(3, 3), strides=(1, 2), padding='same')(x)
	x = Unblock(kernel=(3, 3), strides=(2, 2), padding='same')(x)
	x = Unblock(kernel=(3, 3), strides=(2, 2), padding='same')(x)
	x = Unblock(filters=1, kernel=(1, 2), strides=(1, 2), activation='sigmoid')(x)
	return x


def build_vqvae(k, d, input_shape):
	global SIZE
	global SIZE2
	## Encoder
	encoder_inputs = K.layers.Input(shape=input_shape, name='encoder_inputs')
	z_e = encoder_pass(encoder_inputs, d)
	SIZE = int(z_e.get_shape()[1])
	SIZE2 = int(z_e.get_shape()[2])

	## Vector Quantization
	vector_quantizer = VectorQuantizer(k, name="vector_quantizer")
	codebook_indices = vector_quantizer(z_e)
	encoder = K.Model(inputs=encoder_inputs, outputs=codebook_indices, name='encoder')

	## Decoder
	decoder_inputs = K.layers.Input(shape=(SIZE, SIZE2, d), name='decoder_inputs')
	decoded = decoder_pass(decoder_inputs)
	decoder = K.Model(inputs=decoder_inputs, outputs=decoded, name='decoder')

	## VQVAE Model (training)
	sampling_layer = K.layers.Lambda(lambda x: vector_quantizer.sample(x), name="sample_from_codebook")
	z_q = sampling_layer(codebook_indices)
	codes = tf.stack([z_e, z_q], axis=-1)
	codes = K.layers.Lambda(lambda x: x, name='latent_codes')(codes)
	straight_through = K.layers.Lambda(lambda x: x[1] + tf.stop_gradient(x[0] - x[1]),
	                                   name="straight_through_estimator")
	straight_through_zq = straight_through([z_q, z_e])
	print(straight_through_zq.shape)
	reconstructed = decoder(straight_through_zq)
	vq_vae = K.Model(inputs=encoder_inputs, outputs=[reconstructed, codes], name='vq-vae')

	## VQVAE model (inference)
	codebook_indices = K.layers.Input(shape=(SIZE, SIZE2), name='discrete_codes', dtype=tf.int32)
	z_q = sampling_layer(codebook_indices)
	generated = decoder(z_q)
	vq_vae_sampler = K.Model(inputs=codebook_indices, outputs=generated, name='vq-vae-sampler')

	## Transition from codebook indices to model (for training the prior later)
	indices = K.layers.Input(shape=(SIZE, SIZE2), name='codes_sampler_inputs', dtype='int32')
	z_q = sampling_layer(indices)
	codes_sampler = K.Model(inputs=indices, outputs=z_q, name="codes_sampler")

	## Getter to easily access the codebook for vizualisation
	indices = K.layers.Input(shape=(), dtype='int32')
	vector_model = K.Model(inputs=indices, outputs=vector_quantizer.sample(indices[:, None, None]), name='get_codebook')

	def get_vq_vae_codebook():
		codebook = vector_model.predict(np.arange(k))
		codebook = np.reshape(codebook, (k, d))
		return codebook

	return vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, get_vq_vae_codebook


def mse_loss(ground_truth, predictions):
	mse_loss = tf.reduce_mean((ground_truth - predictions) ** 2, name="mse_loss")
	return mse_loss


def latent_loss(dummy_ground_truth, outputs):
	global BETA
	BETA = 1.0
	del dummy_ground_truth
	z_e, z_q = tf.split(outputs, 2, axis=-1)
	vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q) ** 2)
	commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q)) ** 2)
	latent_loss = tf.identity(vq_loss + BETA * commit_loss, name="latent_loss")
	return latent_loss


def zq_norm(y_true, y_pred):
	del y_true
	_, z_q = tf.split(y_pred, 2, axis=-1)
	return tf.reduce_mean(tf.norm(z_q, axis=-1))


def ze_norm(y_true, y_pred):
	del y_true
	z_e, _ = tf.split(y_pred, 2, axis=-1)
	return tf.reduce_mean(tf.norm(z_e, axis=-1))


def train_vqvae_model(i, x_train, epochs, batch_size,  k, d, input_shape, lr):

	vq_vae, vq_vae_sampler, encoder, decoder, codes_sampler, get_vq_vae_codebook = build_vqvae(
		k, d, input_shape=input_shape)

	vq_vae.compile(loss=[mse_loss, latent_loss], metrics={"latent_codes": [zq_norm, ze_norm]},
	               optimizer=K.optimizers.Adam(lr))

	vq_vae.build(input_shape)

	# EarlyStoppingCallback.
	esc = K.callbacks.EarlyStopping(monitor='loss', min_delta=1e-4,
	                                    patience=5, verbose=0, mode='auto',
	                                    baseline=None, restore_best_weights=True)

	history = vq_vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[esc])

	vq_vae.save_weights(f'../models/vq-vae_{i}.h5')

	with open(f'../models/train_history_{i}', 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
