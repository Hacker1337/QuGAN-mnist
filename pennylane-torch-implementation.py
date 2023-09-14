# %%
from datetime import datetime
import pennylane as qml
from pennylane import numpy as np

# %%
import matplotlib.pyplot as plt
import time
from os.path import join
import os

from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal

import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
from torchvision.utils import save_image

import argparse
torch.manual_seed(0)
np.random.seed(1)

# %%
# parameters
parser = argparse.ArgumentParser(
    description='learns gan on mnist dataset compressed with pca',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--batch_size',type=int, default=10)
parser.add_argument("--dimensions", type=int, default=4,
                    help="dimension of pca compressed data")
parser.add_argument("--epoch",       default=1,     type=int, help="number of epochs to learn")
parser.add_argument("--d_lr",        default=1e-3,  type=float, help="learning rate of the discriminator")
parser.add_argument("--g_lr",        default=1e-3,  type=float, help="learning rate of the generator")
parser.add_argument("--model",       default="q_exp",   type=str, choices=["q_exp", "q_sample", "c", "h_exp", "h_sample"],
                    help="c -- classial GAN,\n" +
                    "q_exp -- expectation value base quantum model with classical noise\n"+
                    "q_sample -- quantum sample based model. Uses quantum randomness\n"+
                    "h_exp -- hybrid model. Quantum generator with exp_val measurement\n"+
                    "    and classical discriminator\n"+
                    "h_sample -- hybrid model. Quantum generator with sample output\n"+
                    "    and classical discriminator"
                    )
parser.add_argument("--d_layers",    default=1,     type=int, help="Number of layers of quantum discriminator")
parser.add_argument("--g_layers",    default=1,     type=int, help="Number of layers of quantum generator")
parser.add_argument("--dataset_size",default=1000,  type=int,)
parser.add_argument("--Finite_diff_step",default=0.1,  type=float, help="relevant only for h_sample model")
parser.add_argument("--number_of_averaged_samples",default=20,  type=int, help="relevant only for h_sample model")

params, unknown = parser.parse_known_args()
if unknown:
    print("Warning, ignored unknown args:", *unknown)
interactive = False
params.dataset_size = min(60_000, params.dataset_size)

# #%%
# # handmade parameter adjust
# params.epoch = 50
# params.model = "q"
# params.d_lr = 1e-2
# params.g_lr = 0*1e-3
# params.dimensions = 2
# params.d_layers = 1
# params.g_layers = 9
# interactive = True

#%%
# wandb
import wandb
from dotenv import load_dotenv
import os

load_dotenv("wandb.env")
wandb.login(key=os.environ["WANDB_API_KEY"])

wandb.init(
    project="QuGAN-mnist",
    notes="",
    config=params,
    save_code=True,
)
# wandb.run.log_code(include_fn=lambda path: path.endswith("translate_train.ipynb"))
# wandb.run.log_code(
#     include_fn=lambda path: path.endswith(".ipynb") or path.endswith(".py")
# )
config = wandb.config

# %%
data_dimensions = config.dimensions

# Hyperparameters
batch_size = config.batch_size
g_lr = config.g_lr
d_lr = config.d_lr

if config.model == "q_exp":
    quantum = True
    use_noise = True
elif config.model == "q_sample":
    quantum = True
    use_noise = False
elif config.model == "c":
    quantum = False
elif config.model == "h_exp":
    use_noise = True
    quantum = True
elif config.model == "h_sample":
    use_noise = False
    quantum = True
else:
    print("Warning! No configurations for such model")

# parameters
log_step = 10

# %% [markdown]
# Dataset preprocessing
# %%
# create name for data folder

time_stamp = datetime.now().strftime("%d.%m_%H:%M:%S")
folder = join("logs", f'{config.model}GAN__d={data_dimensions}__{time_stamp}')
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

pca_data = pca_data[valid_labels][:config.dataset_size]
data_labels = train_labels[valid_labels][:config.dataset_size]
# %%
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
# scale data to be in [-1, 1]
pca_descaler = [[] for _ in range(k)]
for i in range(k):
    a, b = pca_data[:, i].min(), pca_data[:, i].max(),
    pca_descaler[i] = [(a+b)/2, (b-a)/2] # mean, scale
    pca_data[:, i] -= pca_descaler[i][0]
    pca_data[:, i] /= pca_descaler[i][1]

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


# %%
if params.model in ("h_sample", "h_exp"):
    # ### Quantum generator, classical discriminator
    n_qubits = data_dimensions

    dev = qml.device("lightning.qubit", wires=n_qubits)
    shot_dev = qml.device("lightning.qubit", wires=n_qubits)
    diff_method = "adjoint"

    g_params = torch.from_numpy(np.random.uniform(0, 2*np.pi, size=(config.g_layers, 3, n_qubits))).requires_grad_(True)
    if config.model == "h_sample":
        g_params.requires_grad = False
    
    def generator_circ(noise, params):

        for l in range(1):
            for i in range(n_qubits):
                qml.RY(params[l, 0, i], wires=i)
            for i in range(n_qubits):
                qml.IsingYY(params[l, 1, i], wires=[i, (i+1) % n_qubits])
            for i in range(n_qubits):
                qml.CRY(params[l, 2, i], wires=[i, (i+1) % n_qubits])

        if use_noise:
            for i in range(n_qubits):
                qml.RX(noise[i], wires=i)

        for l in range(1, config.g_layers):
            for i in range(n_qubits):
                qml.RY(params[l, 0, i], wires=i)
            for i in range(n_qubits):
                qml.IsingYY(params[l, 1, i], wires=[i, (i+1) % n_qubits])
            for i in range(n_qubits):
                qml.CRY(params[l, 2, i], wires=[i, (i+1) % n_qubits])
        # todo try with different axes rotations



    @qml.qnode(dev, diff_method=diff_method)
    def latent_sample(noise, params):
        generator_circ(noise, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    @qml.qnode(shot_dev)
    def quantum_sample(params):
        generator_circ(None, params)
        return qml.sample()

    def gen_data(noise):
        if use_noise:
            return torch.stack([torch.stack(latent_sample(x, g_params)) for x in noise]).float()
        else:
            shots = config.number_of_averaged_samples
            shot_dev.shots = shots*noise.shape[0]
            res = quantum_sample(g_params).reshape(shots, noise.shape[0], n_qubits) \
                .float().mean(axis=0)
            res = (res - 0.5)*12
            # from [0, 1] to [min, max]
            return res


    g_optimizer = optim.Adam([g_params], lr=g_lr)

    latent_size = n_qubits
    device = torch.device('cpu')

    number_of_generator_params = g_params.numel()

    ## classical discriminator

    import torch.nn as nn

    # Hyperparameters
    image_size = data_dimensions
    hidden_size = 8

    # Discriminator model
    class Discriminator(nn.Module):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(image_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.2),
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
    discriminator = Discriminator().to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=d_lr)

    def discrim_real(data):
        return discriminator(data)

    def discrim_fake(noise):
        return discriminator(gen_data(noise))

    number_of_discriminator_params, = \
    (sum(p.numel() for p in model.parameters() if p.requires_grad)
     for model in [discriminator])

elif quantum:
    # ### Pennylane quantum model

    n_qubits = data_dimensions

    dev = qml.device("lightning.qubit", wires=n_qubits)
    shot_dev = qml.device("lightning.qubit", wires=n_qubits)
    diff_method = "adjoint"

    g_params = torch.from_numpy(np.random.uniform(0, 2*np.pi, size=(config.g_layers, 3, n_qubits))).requires_grad_(True)
    d_params = torch.from_numpy(np.random.uniform(0, 2*np.pi, size=(config.d_layers, 3, n_qubits))).requires_grad_(True)

    def generator_circ(noise, params):

        for l in range(1):
            for i in range(n_qubits):
                qml.RY(params[l, 0, i], wires=i)
            for i in range(n_qubits):
                qml.IsingYY(params[l, 1, i], wires=[i, (i+1) % n_qubits])
            for i in range(n_qubits):
                qml.CRY(params[l, 2, i], wires=[i, (i+1) % n_qubits])

        if use_noise:
            for i in range(n_qubits):
                qml.RX(noise[i], wires=i)

        for l in range(1, config.g_layers):
            for i in range(n_qubits):
                qml.RY(params[l, 0, i], wires=i)
            for i in range(n_qubits):
                qml.IsingYY(params[l, 1, i], wires=[i, (i+1) % n_qubits])
            for i in range(n_qubits):
                qml.CRY(params[l, 2, i], wires=[i, (i+1) % n_qubits])
        # todo try with different axes rotations

    def data_loading_circ(values):
        assert len(values) == n_qubits
        assert min(values) >= -1, max(values) <= 1
        for i in range(n_qubits):
            qml.RY(np.arccos(values[i]), wires=i)

    def discriminator(params):
        for l in range(config.d_layers):
            for i in range(n_qubits):
                qml.RY(params[l, 0, i], wires=i)
            for i in range(n_qubits):
                qml.IsingYY(params[l, 1, i], wires=[i, (i+1) % n_qubits])
            for i in range(n_qubits):
                qml.CRY(params[l, 2, i], wires=[i, (i+1) % n_qubits])

    @qml.qnode(dev, diff_method=diff_method)
    def latent_sample(noise, params):
        generator_circ(noise, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    @qml.qnode(shot_dev,)
    def quantum_sample(params):
        generator_circ(None, params)
        return qml.sample()

    @qml.qnode(dev, diff_method=diff_method, interface="torch")
    def discriminate_generated_circ(input, real=True):
        if real:
            data_loading_circ(input)
        else:
            generator_circ(input, g_params)
        qml.adjoint(discriminator)(d_params)
        return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

        # return qml.probs(wires=range(n_qubits))
        # + take [0] element when using (but doesn't support adjoint diff)
        # and so far throws error "QuantumFunctionError: Adjoint differentiation method does not support expectation return type mixed with other return types"

    projector = torch.zeros((2**n_qubits, 2**n_qubits))
    projector[0, 0] = 1

    def discrim_real(data):
        '''generates probabilities of discriminating'''
        return torch.stack([discriminate_generated_circ(x, real=True)[None] for x in data]).type(data.dtype)

    def discrim_fake(z):
        '''generates probabilities of discriminating'''
        return torch.stack([discriminate_generated_circ(noise, real=False)[None] for noise in z]).type(z.dtype)

    def gen_data(noise):
        if use_noise:
            return [latent_sample(x, g_params) for x in noise]
        else:
            shots = 20
            shot_dev.shots = shots*noise.shape[0]
            res = quantum_sample(g_params).reshape(shots, noise.shape[0], n_qubits) \
                .float().mean(axis=0)
            res = res*2 - 1  # from [0, 1] to [-1, 1]
            return res


    d_optimizer = optim.Adam([d_params], lr=d_lr)
    g_optimizer = optim.Adam([g_params], lr=g_lr)

    latent_size = n_qubits
    device = torch.device('cpu')

    number_of_generator_params = g_params.numel()
    number_of_discriminator_params = d_params.numel()
## %%
# ### classical model
else:
    import torch.nn as nn

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    image_size = data_dimensions
    latent_size = 4
    hidden_size = 4

    # Generator model
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_size, image_size),
                nn.Tanh()  # To ensure pixel values are in the range [-1, 1]
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
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.02),
                nn.Linear(hidden_size, hidden_size),
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

    def discrim_real(data):
        return discriminator(data)

    def discrim_fake(noise):
        return discriminator(generator(noise))

    def gen_data(noise):
        return generator(noise)

    number_of_generator_params, number_of_discriminator_params = \
    (sum(p.numel() for p in model.parameters() if p.requires_grad)
     for model in [generator, discriminator])

# %% some additional functions

def kl_divergance(p, q):
    eps = 0.1/p.size    # to be smaller, the uniform proba
    p = np.maximum(eps, p)
    p /= np.sum(p)
    q = np.maximum(eps, q)
    q /= np.sum(q)
    return np.sum(p * np.log(p / q))

def fake_hellinger_distance(p: np.array, q: np.array):
    results = []
    n = len(p.shape)
    for i in range(n):
        prior_p = np.sum(p, axis=tuple(range(i)) + tuple(range(i+1, n)))
        prior_q = np.sum(q, axis=tuple(range(i)) + tuple(range(i+1, n)))
        results.append(hellinger_distance(prior_p, prior_q))
    return np.mean(results)

def hellinger_distance(p, q):
    pointwise_dist =  (np.sqrt(p) - np.sqrt(q))**2
    return np.sqrt(np.sum(pointwise_dist)/2)

def visualize_discriminator():
    assert data_dimensions == 2
    n = 10
    x = np.linspace(*data_ranges[0], n, dtype=np.float32)
    y = np.linspace(*data_ranges[1], n, dtype=np.float32)
    grids = np.meshgrid(x, y)
    grids = [t.reshape(-1, 1) for t in grids]
    xy = np.concatenate(grids, axis=1)
    with torch.no_grad():
        z = discrim_real(torch.from_numpy(xy))
    z = z.reshape((n, n))
    full_frame()
    plt.contourf(z, vmin=0, vmax=1, levels=12)
    cbar = plt.colorbar()
    cbar.set_label('probability to be real')

def estimate_density(samples, bins=10):
    # Compute the histogram of samples
    hist, _ = np.histogramdd(samples, bins=bins, range=data_ranges)
    return hist/len(samples)

data_ranges = list(zip(np.min(train_dataset, axis=0), np.max(train_dataset, axis=0)))

true_dft = estimate_density(train_dataset) # for future computations

def fit_multidimensional_normal(data):
    # Compute the mean and covariance of the data
    mean = np.mean(data, axis=0)
    covariance_matrix = np.cov(data, rowvar=False)

    # Create a multivariate normal distribution with the computed mean and covariance
    fitted_distribution = multivariate_normal(mean=mean, cov=covariance_matrix)

    return fitted_distribution

def full_frame():
    """helper function to plot without axis and margins"""
    import matplotlib as mpl
    mpl.rcParams['savefig.pad_inches'] = 0
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)

# %% [markdown]
# ## Learning

# %%
# ----- Learning functions -------
criterion = torch.nn.BCELoss()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

wandb.run.summary["g_params"] = number_of_generator_params
wandb.run.summary["d_params"] = number_of_discriminator_params

def get_latent(size):
    return torch.randn(size, latent_size).to(device)

def eval_gradients(batch_size, h):
    global g_params
    gradients = torch.zeros_like(g_params)

    real_labels = torch.ones(batch_size, 1).to(device)
    def eval_loss():
        z = get_latent(batch_size)
        outputs = discrim_fake(z)
        g_loss = criterion(outputs, real_labels)
        return g_loss

    f0 = eval_loss()

    for index in np.ndindex(g_params.shape):
        # save old value in memory for the case of precision problems
        old_value = g_params[index].detach().clone()
        g_params[index] += h

        f1 = eval_loss()
        gradients[index] = (f1-f0)/h

        g_params[index] = old_value

    return gradients, f0

def step_generator(lr, h, batch_size=10_000):
    global g_params
    gradients, loss = eval_gradients(batch_size, h)
    g_params -= gradients*lr
    return loss


def fit_epoch():
    for i, images in enumerate(train_loader):
        batch_size = images.size(0)
        images = images.view(batch_size, -1).to(device)

        # Create real and fake labels for the loss functions
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train the discriminator
        # Real images
        outputs = discrim_real(images)
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
        if config.model in ("h_sample"):
            g_loss = step_generator(g_lr, h=config.Finite_diff_step)
        else:
            z = get_latent(batch_size)
            outputs = discrim_fake(z)
            g_loss = criterion(outputs, real_labels)

            # Backprop and optimize generator
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # Print progress
        if (i + 1) % log_step == 0:
            print(f'Epoch [{epoch+1}/{config.epoch}], Step [{i+1}/{len(train_loader)}], '
                  f'Discr Loss: {d_loss.item():.4f} (r: {d_loss_real:.2f}, f: {d_loss_fake:.2f}), Gen Loss: {g_loss.item():.4f}')
            wandb.log({"d_loss": d_loss.item(), "g_loss": g_loss.item()})


def inference():
    # generate images
    if quantum and use_noise:
        n_samples = 200
    else:
        n_samples = 2000

    points_alpha = min(1, 0.2*1e3/config.dataset_size)
    with torch.no_grad():
        fake_data = np.array(gen_data(get_latent(n_samples)))

    # n_visual = 10
    # fake_images = torch.tensor(pca.inverse_transform(fake_data[:n_visual]), dtype=torch.float32)
    # # os.makedirs(join(folder, "gen_images"), exist_ok=True)
    # # save_image(fake_images.view(fake_images.size(0), 1, 28, 28),
    # #             join(folder, "gen_images", f'generated_images_epoch_{epoch}.jpg'))
    # plt.figure(figsize=(2*n_visual, 2))
    # for i in range(n_visual):
    #     plt.subplot(1, n_visual, i+1)
    #     plt.imshow(fake_images[i].view(28, 28), cmap='gray', vmin=0, vmax=1, interpolation="none")
    #     plt.axis('off')
    # wandb.log({"generated_images": wandb.Image(plt)}, commit=False)

    # # plt.savefig(join(folder, "gen_images", f'my_generated_images_epoch_{epoch}.jpg'))
    # plt.close()

    # compare distributions on plot

    vis_dims = [0, 1]

    for l in data_labels.unique():
        plt.scatter(pca_data[data_labels == l, vis_dims[0]],
                    pca_data[data_labels == l, vis_dims[1]], label=l.item(), c='blue', alpha=points_alpha, s=8)
    plt.title("PCA values of different images")
    plt.scatter(fake_data[:, vis_dims[0]],
                fake_data[:, vis_dims[1]], label="fake", alpha=points_alpha, s=8)

    plt.legend(loc='upper right')
    wandb.log({"distributions":  wandb.Image(plt)}, commit=False)
    plt.close()

    # plot density function approximation
    sampled_data = fake_data[:, vis_dims]

    # # normal distribution approximation
    # fitted_distribution = fit_multidimensional_normal(sampled_data)
    # x, y = np.meshgrid(np.linspace(*data_ranges[vis_dims[0]], 100),
    #                    np.linspace(*data_ranges[vis_dims[1]], 100))
    # pos = np.dstack((x, y))
    # pdf_values = fitted_distribution.pdf(pos)
    # plt.contourf(x, y, pdf_values.T, cmap='Reds', alpha=0.7)
    # plt.title('2D Density Function of Fitted 2D Normal Distribution')
    # wandb.log({"dft_approx":  wandb.Image(plt)}, commit=False)
    # plt.close()

    # handmade hist for full control and contourf plot
    full_frame()
    pdf_values, edges = np.histogramdd(sampled_data, range=[data_ranges[j] for j in vis_dims], density=True)
    plt.contourf(pdf_values.T, cmap='Reds', alpha=0.7,
                 extent=[edges[0][0], edges[0][-1], edges[1][0], edges[1][-1]])
    # plt.title('Density Function Contourf plot')
    for l in data_labels.unique():
        plt.scatter(pca_data[data_labels == l, vis_dims[0]],
                    pca_data[data_labels == l, vis_dims[1]], label=l.item(), c='blue', alpha=points_alpha, s=8)
    plt.legend(loc='upper right')
    wandb.log({"dft_contourf":  wandb.Image(plt)}, commit=False)
    if interactive:
        plt.show()
    plt.close()

    # full_frame()
    # # plt.title('Density Function histogram plot')
    # plt.hist2d(sampled_data[:, 0], sampled_data[:, 1], range=[data_ranges[j] for j in vis_dims],
    #            alpha=0.7, cmap="Reds", bins=15)
    # for l in data_labels.unique():
    #     plt.scatter(pca_data[data_labels == l, vis_dims[0]],
    #                 pca_data[data_labels == l, vis_dims[1]], label=l.item(), c='blue', alpha=points_alpha, s=8)
    # plt.legend(loc='upper right')
    # wandb.log({"dft_approx_hist":  wandb.Image(plt)}, commit=False)
    # if interactive:
    #     plt.show()
    # plt.close()

    if data_dimensions == 2:
        # visualize discriminator thoughts
        visualize_discriminator()
        wandb.log({"disciminator_predictions":  wandb.Image(plt)}, commit=False)
        if interactive:
            plt.show()
        plt.close()


    # compute metrics
    metrics = {}
    fake_dft = estimate_density(fake_data)
    with torch.no_grad():
        metrics["kl_divergence"] = kl_divergance(true_dft, fake_dft).item()
        metrics["hellinger_distance"] = hellinger_distance(true_dft, fake_dft).item()
        metrics["fake_hellinger_distance"] = fake_hellinger_distance(true_dft, fake_dft).item()
        wandb.log(metrics, commit=False)
    print(metrics)

    return metrics



# %%
# Training loop
for epoch in range(config.epoch):
    start = time.time()
    # Save generated images each epoch
    inference()

    print('Time for inference is {} sec'.format(time.time() - start))

    start = time.time()
    fit_epoch()
    print('Time for Epoch {} is {} sec'.format(epoch + 1, time.time() - start))

epoch+=1
inference()

# # Save models
# if quantum: # replace with universal try/catch block
#     torch.save(g_params, join(folder, 'generator_model.pth'))
#     torch.save(d_params, join(folder, 'discriminator_model.pth'))
# else:
#     torch.save(generator.state_dict(), join(folder, 'generator_model.pth'))
#     torch.save(discriminator.state_dict(), join(folder, 'discriminator_model.pth'))

# %%
wandb.finish()
# %%
# # Save models
# torch.save(g_params, join(folder, 'generator_model.pth'))
# torch.save(discriminator.state_dict(), join(folder, 'discriminator_model.pth'))
# # %%
# # Load models
# folder = "logs/h_sampleGAN__d=2__09.09_16:20:13"
# g_params = torch.load(join(folder, 'generator_model.pth'))
# discriminator.load_state_dict(torch.load(join(folder, 'discriminator_model.pth')))
# # %%
# inference()
