import math
import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt

training_spectra = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrn.txt'))
training_labels = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrnLbls.txt')).type(torch.int)
training_components = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrnFrctns.txt'))
training_components = torch.hstack([training_components, (1 - training_components.sum(axis=1)).reshape(-1,1)])
test_spectra = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrn.txt'))
test_labels = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrnLbls.txt')).type(torch.int)
test_components = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrnFrctns.txt'))
test_components = torch.hstack([test_components, (1 - test_components.sum(axis=1)).reshape(-1,1)])

num_training = 1000

train_x = training_spectra[:num_training]
train_y = training_components[:num_training]

print(train_x.shape, train_y.shape)

num_latents = 3
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


model = MultitaskGPModel(torch.rand(num_latents, 50, 12), num_latents, num_tasks)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

output = model(train_x)
print(output.__class__.__name__, output.event_shape)



import os
smoke_test = ('CI' in os.environ)
num_epochs = 5000


model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
epochs_iter = range(num_epochs)
for i in epochs_iter:
    print(i)
    # Within each iteration, we will go over each minibatch of data
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()

# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():

    test_x = test_spectra
    test_y = test_components
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# %%
# Initialize plots
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
for task, ax in enumerate(axs):
    print(task)
    # Plot training data as black stars
    axs[task].scatter(test_components[:,task], mean[:,task])
    axs[task].set_title(f'Task {task + 1}')


# %%


print("log_prob:", predictions.log_prob(test_y))
print("msep:", torch.mean((mean - test_y)**2).item())


with open("examples/comparison_methods/gp_comparison/msep/regression_oil_flow.txt", 'a') as file:
    file.write(f"{((mean- test_y)**2).mean()}\n")

# %%

with open("examples/comparison_methods/gp_comparison/lpd/regression_oil_flow.txt", 'a') as file:
    file.write(f"{predictions.log_prob(test_y)}\n")