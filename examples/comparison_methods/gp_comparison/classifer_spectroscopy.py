

from matplotlib import pyplot as plt
import pandas as pd
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.likelihoods import MultitaskGaussianLikelihood
import numpy as np

from src.data import SpectralData

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

training_df = pd.read_csv("examples/data/UCRArchive_2018/Rock/Rock_TRAIN.tsv", delimiter="\t", header  = None)
test_df = pd.read_csv("examples/data/UCRArchive_2018/Rock/Rock_TEST.tsv", delimiter="\t", header  = None)

training_components_one_hot = torch.Tensor(pd.get_dummies(training_df[0]).to_numpy())
training_spectra = torch.Tensor(training_df.drop(0, axis= 1).to_numpy())
test_components_one_hot = torch.Tensor(pd.get_dummies(test_df[0]).to_numpy())
test_spectra = torch.Tensor(test_df.drop(0, axis= 1).to_numpy())


combined_spectra = torch.concat([training_spectra, test_spectra])

# %%

training_spectral_data = SpectralData(torch.Tensor(training_df.columns[1:]).reshape(-1,1), training_spectra)
training_spectral_data.trim_wavelengths(0,2250)
training_spectral_data.snv()
# training_spectral_data = IndependentObservations(training_spectral_data.spectra)
test_spectral_data = SpectralData(torch.Tensor(test_df.columns[1:]).reshape(-1,1), test_spectra)
test_spectral_data.trim_wavelengths(0,2250)
test_spectral_data.snv()


cov_matrix = (torch.matmul(test_spectral_data.spectra.T, test_spectral_data.spectra) + torch.matmul(training_spectral_data.spectra.T, training_spectral_data.spectra)) / (training_spectral_data.num_data_points + test_spectral_data.num_data_points - 1)

# Perform Singular Value Decomposition
U, S, V = torch.svd(cov_matrix)

# Calculate cumulative variance
total_variance = torch.sum(S)
cumulative_variance = torch.cumsum(S, 0)/total_variance

# Find the number of components to reach 99% of variance
print(cumulative_variance)
num_pca_dims = 1 
while cumulative_variance[num_pca_dims-1] < 1 - 1e-2: 
    num_pca_dims += 1 

print("num_components ", num_pca_dims)


test_spectra, _, _ = torch.pca_lowrank(test_spectral_data.spectra, num_pca_dims)
training_spectra, _, _ = torch.pca_lowrank(training_spectral_data.spectra, num_pca_dims)


# Assume X_train (12-dimensional input) and y_train (labels) are your data
X_train = training_spectra
y_train = training_components_one_hot.argmax(dim=-1)

# Instantiate model and likelihood
num_latents = 3
num_tasks = 4
model =MultitaskGPModel(torch.rand(num_latents, 50, num_pca_dims), num_latents, 4)
# likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_classes=4, num_features=4)
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

test_labels = test_components_one_hot

with torch.no_grad():
    test_latent_functions = model(test_spectra)
    test_preds = likelihood(test_latent_functions)
    preds = test_preds.sample_n(1000)
    preds = preds.reshape(-1,50)
    cat_1 = (preds ==0).sum(dim=0)
    cat_2 = (preds ==1).sum(dim=0)
    cat_3 = (preds ==2).sum(dim=0)
    cat_4 = (preds ==3).sum(dim=0)
    preds_final = torch.vstack([cat_1, cat_2, cat_3, cat_4]).T
    print("accuracy:", (preds_final.argmax(dim=1) == test_labels.argmax(dim=1)).type(torch.float64).mean())
    print("approx_log_prob:", torch.log((preds_final[test_labels.type(torch.bool)]  +1)/10002 ).sum())

# %%

with open("examples/comparison_methods/gp_comparison/acc/spectroscopy.txt", 'a') as file:
    file.write(f"{(preds_final.argmax(dim=1) == test_labels.argmax(dim=1)).type(torch.float64).mean()}\n")

# %%

with open("examples/comparison_methods/gp_comparison/lp/spectroscopy.txt", 'a') as file:
    file.write(f"{torch.log((preds_final[test_labels.type(torch.bool)]  +1)/10002 ).sum()}\n")