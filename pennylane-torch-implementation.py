# %%
from datetime import datetime
import pennylane as qml
from pennylane import numpy as np

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import time
from os.path import join
import os

from sklearn.decomposition import PCA

import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torchvision.utils import save_image

import argparse

# %%
# parameters
parser = argparse.ArgumentParser(
    description='learns gan on mnist dataset compressed with pca')

parser.add_argument(
    '--batch_size',type=int, default=10)
parser.add_argument("--dimensions", type=int, default=4, 
                    help="dimension of pca compressed data")
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--model",  type=str, choices=["q", "c"])

config, unknown = parser.parse_known_args()
if unknown:
    print("Warning, ignored unknown args:", *unknown)
# %%
size_dataset = 1000
data_dimensions = config.dimensions

# Hyperparameters
batch_size = 10
g_lr = 1e-1
d_lr = 1e-2

if config.model == "q":
    quantum = True
elif config.model == "c":
    quantum = False
    
# parameters
log_step = 10

# %% [markdown]
# Dataset preprocessing
# %%
# create name for data folder

time_stamp = datetime.now().strftime("%d.%m_%H:%M:%S")
folder = join("logs", f'{"q" if quantum else "c"}GAN__d={data_dimensions}__{time_stamp}')
os.makedirs(folder)
# %%
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST(root="~/.mnist", train=True, download=True,
                         transform=transform)

# %%
train_images = torch.stack([image.flatten() for image, label in dataset])
train_labels = torch.tensor([label for image, label in dataset])


# %%
# --------------------------------------------------
# ---------------- PCA Section ---------------------
# --------------------------------------------------
k = data_dimensions
pca = PCA(n_components=k)
pca.fit(train_images)
pca_data = pca.transform(train_images).astype(np.float32)

valid_labels = (train_labels == 3) | (train_labels == 6) | (train_labels == 9)

pca_data = pca_data[valid_labels][:size_dataset]
data_labels = train_labels[valid_labels][:size_dataset]
# %%
# visualizing different dimensions reconstruction images
n_images = 10

plt.figure(figsize=(4*n_images, 4*2))
original_images = train_images[:n_images]

plt.suptitle(f"Real images vs {data_dimensions} dimensions PCA")
for i in range(n_images):
    plt.subplot(2, n_images, i+1)
    plt.imshow(original_images[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="none")

pca.fit(train_images)
mixed_images = pca.inverse_transform(pca.transform(original_images))

for i in range(n_images):
    plt.subplot(2, n_images, n_images + i+1)
    plt.imshow(mixed_images[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="none")
plt.savefig(join(folder, "pca_effect"))
plt.close()


# %%
# --------------------------------------------------
# --------  Normalize vector components ------------
# --------------------------------------------------
pca_descaler = [[] for _ in range(k)]
for i in range(k):
    if pca_data[:, i].min() < 0:
        pca_descaler[i].append(pca_data[:, i].min())
        pca_data[:, i] += np.abs(pca_data[:, i].min())
    else:
        pca_descaler[i].append(pca_data[:, i].min())
        pca_data[:, i] -= pca_data[:, i].min()
    pca_descaler[i].append(pca_data[:, i].max())
    pca_data[:, i] /= pca_data[:, i].max()


train_dataset = pca_data

print(f"The Total Explained Variance of {k} Dimensions is {sum(pca.explained_variance_ratio_).round(3)}")

# --------------------------------------------------
# Define a function that can take in PCA'ed data and return an image
# --------------------------------------------------


def descale_points(d_point, scales=pca_descaler, tfrm=pca):
    for col in range(d_point.shape[1]):
        d_point[:, col] *= scales[col][1]
        d_point[:, col] += scales[col][0]
    reconstruction = tfrm.inverse_transform(d_point)
    return reconstruction


# %% [markdown]
if quantum:
    # ### Pennylane quantum model

    n_qubits = data_dimensions
    use_noise = True

    dev = qml.device("lightning.qubit", wires=n_qubits)
    diff_method = "adjoint"

    d_params = torch.from_numpy(np.random.uniform(0, 2*np.pi, size=(3, n_qubits))
                                ).requires_grad_(True)  # todo add multiple layers support
    g_params = torch.from_numpy(np.random.uniform(0, 2*np.pi, size=(3, n_qubits))).requires_grad_(True)

    def generator_circ(noise, params):
        if use_noise:
            for i in range(n_qubits):
                qml.RX(noise[i], wires=i)

        params = params.reshape(-1, n_qubits)
        for i in range(n_qubits):
            qml.RY(params[0, i], wires=i)
        for i in range(n_qubits):
            qml.IsingYY(params[1, i], wires=[i, (i+1) % n_qubits])
        for i in range(n_qubits):
            qml.CRY(params[1, i], wires=[i, (i+1) % n_qubits])
        # todo try with different axes rotations

    def data_loading_circ(values):
        assert len(values) == n_qubits
        assert min(values) >= -1, max(values) <= 1
        for i in range(n_qubits):
            qml.RY(np.arccos(values[i]), wires=i)

    def discriminator(params):
        for i in range(n_qubits):
            qml.RY(params[0, i], wires=i)
        for i in range(n_qubits):
            qml.IsingYY(params[1, i], wires=[i, (i+1) % n_qubits])
        for i in range(n_qubits):
            qml.CRY(params[1, i], wires=[i, (i+1) % n_qubits])

    @qml.qnode(dev, diff_method=diff_method)
    def sample(noise, params):
        generator_circ(noise, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=diff_method)
    def discriminate_generated_circ(input, real=True):
        if real:
            data_loading_circ(input)
        else:
            generator_circ(input, g_params)
        qml.adjoint(discriminator)(d_params)
        return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

    # todo compare speed of this with:
        # return qml.probs(wires=range(n_qubits))
        # + take [0] element when using (but doesn't support adjoint diff)
        # and so far thows error "QuantumFunctionError: Adjoint differentiation method does not support expectation return type mixed with other return types"

    projector = torch.zeros((2**n_qubits, 2**n_qubits))
    projector[0, 0] = 1

    def discim_real(data):
        '''generates probabilities of discriminating'''
        return torch.stack([discriminate_generated_circ(x, real=True)[None] for x in data]).type(data.dtype)

    def discrim_fake(z):
        '''generates probabilities of discriminating'''
        return torch.stack([discriminate_generated_circ(noise, real=False)[None] for noise in z]).type(z.dtype)

    def gen_data(noise):
        return [sample(x, g_params) for x in noise]

    d_optimizer = optim.Adam([d_params], lr=d_lr)
    g_optimizer = optim.Adam([g_params], lr=g_lr)

    latent_size = n_qubits
    device = torch.device('cpu')

## %%
# ### classical model
else:
    import torch.nn as nn

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    image_size = data_dimensions
    latent_size = 10
    hidden_size = 10

    # Generator model
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Linear(hidden_size, image_size),
                nn.Sigmoid()  # To ensure pixel values are in the range [0, 1]
            )

        def forward(self, x):
            return self.model(x)

    # Discriminator model
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(image_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

        def predict_real(self, data):
            return self.model(data)

        def predict_fake(self, noise):
            return self.model(self.gen(noise).detach())

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)
    g_optimizer = optim.Adam(generator.parameters(), lr=g_lr)

    def discim_real(data):
        return discriminator(data)

    def discrim_fake(noise):
        return discriminator(generator(noise))

    def gen_data(noise):
        return generator(noise)


# %% [markdown]
# ## Learning

# %%
# ----- Learning -------
criterion = torch.nn.BCELoss()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def get_latent(size):
    return torch.randn(size, latent_size).to(device)


def fit_epoch():
    for i, images in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.view(batch_size, -1).to(device)

        # Create real and fake labels for the loss functions
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train the discriminator
        # Real images
        outputs = discim_real(images)
        d_loss_real = criterion(outputs, real_labels)

        # Fake images
        z = get_latent(batch_size)
        outputs = discrim_fake(z)
        d_loss_fake = criterion(outputs, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        # Backprop and optimize discriminator
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train the generator
        z = get_latent(batch_size)
        outputs = discrim_fake(z)   # todo check whether it is possible to reuse previous computation ouf this outputs
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize generator
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Print progress
        if (i + 1) % log_step == 0:
            print(f'Epoch [{epoch+1}/{config.num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}')


def inference():
    # generate images
    with torch.no_grad():
        fake_data = gen_data(get_latent(64))
        fake_images = torch.tensor(pca.inverse_transform(fake_data), dtype=torch.float32)
        os.makedirs(join(folder, "gen_images"), exist_ok=True)
        save_image(fake_images.view(fake_images.size(0), 1, 28, 28),
                   join(folder, "gen_images", f'generated_images_epoch_{epoch}.jpg'))
        plt.figure(figsize=(4*10, 4))
        for i in range(10):
            plt.subplot(1, 10, i+1)
            plt.imshow(fake_images[i].view(28, 28), cmap='gray', vmin=0, vmax=1, interpolation="none")
            plt.axis('off')
        plt.savefig(join(folder, "gen_images", f'my_generated_images_epoch_{epoch}.jpg'))
        plt.close()

    # compare distributions

    vis_dims = [0, 1]
    n_samples = 100
    with torch.no_grad():
        fake_data = np.array(gen_data(get_latent(n_samples)))

    labels = data_labels.unique()
    for l in labels:
        index = data_labels == l
        plt.scatter(pca_data[index, vis_dims[0]],
                    pca_data[index, vis_dims[1]], label=l.item(), alpha=0.2, s=8)
    plt.title("PCA values of different images")
    plt.scatter(fake_data[:, vis_dims[0]],
                fake_data[:, vis_dims[1]], label="fake", alpha=0.2, s=8)

    plt.legend()
    os.makedirs(join(folder, "distribution"), exist_ok=True)
    plt.savefig(join(folder, "distribution", f'scatter_distribution_epoch_{epoch}.jpg'))
    plt.close()



# Training loop
for epoch in range(config.num_epochs):
    start = time.time()
    # Save generated images each epoch
    inference()
    print('Time for inference is {} sec'.format(time.time() - start))

    start = time.time()
    fit_epoch()
    print('Time for Epoch {} is {} sec'.format(epoch + 1, time.time() - start))

epoch+=1
inference()

# Save models
if quantum:
    torch.save(g_params, join(folder, 'generator_model.pth'))
    torch.save(d_params, join(folder, 'discriminator_model.pth'))
else:
    torch.save(generator.state_dict(), join(folder, 'generator_model.pth'))
    torch.save(discriminator.state_dict(), join(folder, 'discriminator_model.pth'))


#%%
# temp cell
# counting kl divergence and heidegger distance



