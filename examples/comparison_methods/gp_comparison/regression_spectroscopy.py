import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import gpytorch
from matplotlib import pyplot as plt
from scipy.io import loadmat

from src.data import SpectralData

data = loadmat("examples/data/flodat.mat")
del(data["SDisk"])

for key, item in data.items():
    data[str(key)] = torch.Tensor(item)




all_spectra_tensor = torch.hstack([data["spec30"],
    data["spec40"],
    data["spec50"],
    data["spec60"],
    data["spec70"],
]).T
all_spectra_dataset = SpectralData(data["wl"], all_spectra_tensor)
all_spectra_dataset.snv()
all_spectra_dataset.trim_wavelengths(800,1000)

cov_matrix = torch.matmul(all_spectra_dataset.spectra.T, all_spectra_dataset.spectra) / (all_spectra_dataset.spectra.size(0) - 1)

# Perform Singular Value Decomposition
U, S, V = torch.svd(cov_matrix)

# Calculate cumulative variance
total_variance = torch.sum(S)
cumulative_variance = torch.cumsum(S, 0)/total_variance

# Find the number of components to reach 99% of variance
print(cumulative_variance)
num_pca_dims = 1 
while cumulative_variance[num_pca_dims-1] < 1 - 1e-4: 
    num_pca_dims += 1 

print("num_components ", num_pca_dims)
# %%
all_spectra_dataset.spectra, _, _ = torch.pca_lowrank(all_spectra_dataset.spectra, num_pca_dims)


all_temperatures = torch.vstack([
    data["temper30"],
    data["temper40"],
    data["temper50"],
    data["temper60"],
    data["temper70"],
])

all_components = data["conc"].repeat(5,1)

train_x, test_x, train_y, test_y = train_test_split(all_spectra_dataset.spectra, all_components, test_size=0.5)


num_latents = 2
num_tasks = 3

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_latents, num_tasks):
        # Let's use a different set of inducing points for each latent function
        inducing_points = inducing_points

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


model = MultitaskGPModel(torch.rand(num_latents, 50, num_pca_dims), num_latents, num_tasks)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

output = model(train_x)
print(output.__class__.__name__, output.event_shape)


num_epochs = 2000


model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=1e-2)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
epochs_iter = range(num_epochs)
loss_list = []
for i in epochs_iter:
    print(i)
    # Within each iteration, we will go over each minibatch of data
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()
# %%
# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():

    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

with torch.no_grad():
    plt.plot(loss_list)

# %%
# Initialize plots
# fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
# for task, ax in enumerate(axs):
#     print(task)
#     # Plot training data as black stars
#     axs[task].scatter(test_y[:,task], mean[:,task])
#     axs[task].set_title(f'Task {task + 1}')

# %%

# print("log_prob:", predictions.log_prob(test_y)/test_y.shape[0])
# print("msep:", torch.mean((mean - test_y)**2))

# %%

print("log_prob:", predictions.log_prob(test_y))
print("msep:", torch.mean((mean - test_y)**2).item())


with open("examples/comparison_methods/gp_comparison/msep/regression_spectroscopy.txt", 'a') as file:
    file.write(f"{((mean- test_y)**2).mean()}\n")

# %%

with open("examples/comparison_methods/gp_comparison/lpd/regression_spectroscopy.txt", 'a') as file:
    file.write(f"{predictions.log_prob(test_y)}\n")