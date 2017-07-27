# denoising_autoencoder.py															26-Jul-2017
#
#

import tensorflow as tf
import numpy as np


def weight_variable(shape, name=None):
   """
   Create a weight matrix
   """

   return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)


def bias_variable(shape, name=None):
   """
   Create a bias variable
   """

   return tf.Variable(tf.constant(0.01, shape=shape), name=name)


def linear(x):
	"""
	"""

	return x


class DenoisingAutoencoder:
	"""
	A denoising autoencoder
	"""

	def __init__(self, input_size, code_size, **kwargs):
		"""
		"""

		# Create all the needed tensorflow stuff
		self.sess = kwargs.get('session', tf.InteractiveSession())
		hidden_activation = kwargs.get('hidden_activation', tf.nn.relu)
		output_activation = kwargs.get('output_activation', tf.nn.relu)
		self.name = kwargs.get('name', None)

		with tf.variable_scope(self.name):
			# Input to the network
			self.input = tf.placeholder(tf.float32, (None, input_size), name='input_'+self.name)
			self.dropout_prob = tf.placeholder(tf.float32, None, name='dropout_probability_' + self.name)
			self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate_' + self.name)

			# Weights and biases of the network
			self.W1 = weight_variable((input_size, code_size), 'W_encode_'+self.name)
			self.b1 = bias_variable((code_size,), 'b_encode_' + self.name)
			self.W2 = weight_variable((code_size, input_size), 'W_decode_'+self.name)
			self.b2 = bias_variable((input_size,), 'b_decode_' + self.name)

			x = tf.nn.dropout(self.input, self.dropout_prob)

			# Code layer and output layer
			self.code = hidden_activation(tf.matmul(x, self.W1) + self.b1)

			h = tf.nn.dropout(self.code, self.dropout_prob)

			output = output_activation(tf.matmul(h, self.W2) + self.b2)

			# Build an optimizer
	        self.loss = tf.reduce_sum(tf.square(output - self.input), name='objective_' + self.name)
	        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


	def train(self, dataset, dropout_prob = 0.2, learning_rate = 0.1):
		"""
		"""

		fd = {self.input: dataset, self.dropout_prob: dropout_prob, self.learning_rate: learning_rate}

		self.sess.run(self.train_step, feed_dict=fd)


	def get_loss(self, dataset, dropout_prob = 0.0):
		"""
		"""

		fd = {self.input: dataset, self.dropout_prob: dropout_prob}

		return self.sess.run(self.loss, feed_dict=fd)


	def get_code(self, dataset, dropout_prob = 0.0):
		"""
		"""

		fd = {self.input: dataset, self.dropout_prob: dropout_prob}

		return self.sess.run(self.code, feed_dict=fd)







