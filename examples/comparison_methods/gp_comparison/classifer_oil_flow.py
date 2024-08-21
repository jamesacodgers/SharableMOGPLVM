

from matplotlib import pyplot as plt
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import MultitaskGaussianLikelihood
import numpy as np

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
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([num_latents]))
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

training_spectra = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrn.txt'))
training_labels = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrnLbls.txt')).type(torch.int)
training_components = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrnFrctns.txt'))
training_components = torch.hstack([training_components, (1 - training_components.sum(axis=1)).reshape(-1,1)])
test_spectra = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTst.txt'))
test_labels = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTstLbls.txt')).type(torch.int)
test_components = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTstFrctns.txt'))
test_components = torch.hstack([test_components, (1 - test_components.sum(axis=1)).reshape(-1,1)])

# Assume X_train (12-dimensional input) and y_train (labels) are your data
X_train = training_spectra
y_train = training_labels.argmax(dim=-1)

# Instantiate model and likelihood
num_latents = 2
num_tasks = 3
model =MultitaskGPModel(torch.rand(num_latents, 50, 12), num_latents, 3)
# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=3, num_features=3)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=1000)


# Training routine
model.train()
likelihood.train()

# Use your favorite optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

training_iterations = 2000
# Train the model
loss_list = []
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss_list.append(loss.item())
    loss.backward()
    optimizer.step()
    print(i)

# %%
with torch.no_grad():
    plt.plot(loss_list)


# %%
model.eval()
likelihood.eval()
n = 10_000
with torch.no_grad():
    test_latent_functions = model(test_spectra)
    test_preds = likelihood(test_latent_functions)
    preds = test_preds.sample_n(n)
    preds = preds.reshape(-1,1000)
    cat_1 = (preds ==0).sum(dim=0)
    cat_2 = (preds ==1).sum(dim=0)
    cat_3 = (preds ==2).sum(dim=0)
    preds_final = torch.vstack([cat_1, cat_2, cat_3]).T
    print("accuracy:", torch.mean((preds_final.argmax(dim=1) == test_labels.argmax(dim=1)).type(torch.float)))
    print("approx_log_prob:", torch.log(((preds_final[test_labels.type(torch.bool)]+1)/(preds.shape[0] + 2 ))).sum())

# %%

with open("examples/comparison_methods/gp_comparison/acc/oil_flow.txt", 'a') as file:
    file.write(f"{torch.mean((preds_final.argmax(dim=1) == test_labels.argmax(dim=1)).type(torch.float))}\n")

# %%

with open("examples/comparison_methods/gp_comparison/lp/oil_flow.txt", 'a') as file:
    file.write(f"{torch.log(((preds_final[test_labels.type(torch.bool)]+1)/(preds.shape[0] + 2 ))).sum()}\n")