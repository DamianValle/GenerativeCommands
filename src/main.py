from get_data import get_data
from vq_vae import train_vqvae_model

data = get_data()

x_train, _, _ = data['data']

del data

# Hyperparameters
k = 256  # Number of codebook entries
d = 2  # Dimension of each codebook entries

input_shape = x_train.shape[1:]


batch_size = 128  # Batch size for training the VQVAE
epochs = 3  # Number of epochs
lr = 3e-4  # Learning rate

params = [(64, 4), (256, 2), (256, 4), (512, 30), (512, 1)]

for i, (k, d) in enumerate(params):

	train_vqvae_model(i, x_train, epochs, batch_size,  k, d, input_shape, lr)