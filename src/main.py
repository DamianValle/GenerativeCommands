from get_data import get_data
from vq_vae import train_vqvae_model

x_train = get_data()

#x_train, _, _ = data['data']

#del data

# Hyperparameters
k = 256  # Number of codebook entries
d = 2  # Dimension of each codebook entries

input_shape = x_train.shape[1:]


batch_size = 128  # Batch size for training the VQVAE
epochs = 3  # Number of epochs
lr = 3e-4  # Learning rate

params = [(128, 64), (256, 64), (512, 64)]
beta = [0.25, 0.75]

for i, (k, d) in enumerate(params):
	for b in beta:
		train_vqvae_model(i, x_train, epochs, batch_size,  k, d, input_shape, lr, b)