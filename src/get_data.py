import numpy as np
import tensorflow.keras as K


class Normalizer:

	def fit_transform(self, data):
		self.data_min = data.min()
		data -= self.data_min
		self.data_max_after = data.max()
		data /= self.data_max_after

	def transform(self, data):
		data -= self.data_min
		data /= self.data_max_after

	def unnormalize(self, data):
		ret_data = data * self.data_max_after
		ret_data += self.data_min
		return ret_data


def get_data():
	commands = 'bed bird cat dog down eight five four go happy house left marvin nine no off on one right seven sheila six stop three tree two up wow yes zero'

	idx2command = []
	for command in commands.split(' '):
		idx2command.append(command)

	command2idx = {}
	for idx, command in enumerate(idx2command):
		command2idx[command] = idx

	with open('../data/train_x.npy', 'rb') as f:
		train_x = np.load(f)

	normalizer = Normalizer()
	normalizer.fit_transform(train_x)
	#normalizer.transform(val_x)
	#normalizer.transform(test_x)

	return train_x
	#return {'data': (train_x, val_x, test_x), 'labels': (Y_train, Y_val, Y_test)}
