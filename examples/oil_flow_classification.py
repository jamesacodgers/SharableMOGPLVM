# Code required for oil flow classification


# %%
import copy
from scipy.io import loadmat
import torch
from src.mogplvm import MOGPLVM
import matplotlib.pyplot as plt
import pickle
from itertools import chain
from sklearn.model_selection import train_test_split

from src.data import Dataset, IndependentObservations, ObservedComponents, SpectralData, VariationalClassifier, VariationalDirichletDistribution, get_init_values_for_latent_variables
from src.utils.plot_utils import SpectraPlot, project_covar_to_dirichlet, project_vect_to_dirichlet
from src.utils.prob_utils import dirichlet_cov
from src.utils.tensor_utils import log_linspace
from src.utils.train_utils import lbfgs_training_loop, train_bass_on_spectral_data

from sklearn.cross_decomposition import PLSRegression

import numpy as np
import pandas as pd
# %%


torch.set_default_dtype(torch.float64)


torch.manual_seed(1234)
np.random.seed(1234)

num_training = 1e9
num_test = 1e9

training_spectra = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrn.txt'))
training_labels = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrnLbls.txt')).type(torch.int)
training_components = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTrnFrctns.txt'))
training_components = torch.hstack([training_components, (1 - training_components.sum(axis=1)).reshape(-1,1)])
test_spectra = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTst.txt'))
test_labels = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTstLbls.txt')).type(torch.int)
test_components = torch.Tensor(np.loadtxt(fname='examples/data/3phData/DataTstFrctns.txt'))
test_components = torch.hstack([test_components, (1 - test_components.sum(axis=1)).reshape(-1,1)])



# %%
# cmap = plt.get_cmap("plasma")

# fig, ax = plt.subplots()
# for y,r,x in zip(training_spectra, training_components, training_labels):
#     ax.plot(y[[0,2,4,6,8,10,1,3,5,7,9,11]], color = cmap(r[2]), )
#%%


# %%

training_spectral_data = IndependentObservations(training_spectra)
# training_components_data = ObservedComponents(training_components[:num_training])
training_components_data = ObservedComponents(training_labels)

test_spectral_data =IndependentObservations(test_spectra)
test_components_data = VariationalDirichletDistribution(torch.ones(test_spectral_data.num_data_points, 3))
# test_components_data = VariationalClassifier(torch.ones(test_spectral_data.num_data_points, 3)/3)

# training_spectral_data.snv()
# test_spectral_data.snv()
# %%

num_latent_dims = 10
x_init_train = get_init_values_for_latent_variables(training_spectral_data, training_spectral_data, training_components_data, num_latent_dims)



training_dataset = Dataset(
                training_spectral_data, 
                training_components_data, 
                x_init_train,
                Sigma_x = 1*torch.ones(training_spectral_data.num_data_points, num_latent_dims
                ))


# %%
test_dataset = Dataset(
                test_spectral_data, 
                test_components_data, 
                # x_test_init,
                torch.zeros(test_spectral_data.num_data_points, num_latent_dims),
                Sigma_x = 1*torch.ones(test_spectral_data.num_data_points, num_latent_dims
                ))

# %%



bass = MOGPLVM(
    beta = torch.ones(x_init_train.shape[1])*1, 
    # gamma=torch.ones(1)*1e-1, 
    sigma2 = torch.ones(1)*1,
    sigma2_s = torch.ones(1)*1,
    v_x = torch.randn(16,num_latent_dims),
    # v_l= torch.arange(12).reshape(-1,1).type(torch.float64)
)



loss = []

adam_bass = torch.optim.Adam([bass._log_beta, bass._log_sigma2, bass._log_sigma2_s, bass.v_x], lr = 1e-2)
# adam_training = torch.optim.Adam(chain([bass._log_beta, bass._log_sigma2_s, bass.v_x, bass._log_sigma2], training_dataset.parameters()), lr = 5e-3)
adam_training = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters()), lr = 5e-3)



# %%
loss.extend(train_bass_on_spectral_data(bass, [training_dataset], adam_training, 4000))

# loss.extend(lbfgs_training_loop(bass, [training_dataset], chain(training_dataset.parameters()) , 2))
# while  loss[-1] - loss[-2] > 1:
#     loss.extend(lbfgs_training_loop(bass, [training_dataset, ], chain(test_dataset.parameters()) , 1))
# %%

noise_schedule = log_linspace(1,bass.sigma2.item(), 20)
for s in noise_schedule:
    print(s)
    bass.sigma2 = s
    loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters()) , 2))
    # while  loss[-1] - loss[-2] > 1:
    #     loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters()) , 1))

# %%
    
test_dataset.components_distribution = VariationalClassifier(test_dataset.get_r())

adam_all = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters(), test_dataset.parameters()), lr = 1e-2)

# for s in noise_schedule:
#     print(s)
#     bass.sigma2 = s
#     loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters()) , 2))
    # while  loss[-1] - loss[-2] > 1:
    #     loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters()) , 1))
loss.extend(train_bass_on_spectral_data(bass, [training_dataset, test_dataset], adam_all, 5000))
# %%
loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(training_dataset.parameters(), test_dataset.parameters(), bass.parameters()) , 2))
while  loss[-1] - loss[-2] > 1:
    loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(training_dataset.parameters(), test_dataset.parameters(), bass.parameters()) , 1))
# %%
# fig, ax = plt.subplots()
# for y,r,x in zip(training_dataset.observations.get_observations(), training_components, training_labels):
#     ax.plot(y, color = cmap(r[2]), )

# %%

# args = bass.beta.argsort()
# with torch.no_grad():
#     plt.scatter(training_dataset.mu_x[:,args[0]], training_dataset.mu_x[:,args[1]], c = training_labels.argmax(dim=1))
#     plt.scatter(test_dataset.mu_x[:,args[0]], test_dataset.mu_x[:,args[1]], c = training_labels.argmax(dim=1), marker="x")
#     # plt.scatter(bass.v_x[:,args[0]], bass.v_x[:,args[1]], c = "black", marker="*")
#     # plt.savefig(f"examples/figs/oil_flow/mixture_latent_space.pdf", bbox_inches = "tight")
#     theta = torch.arange(0, 2*torch.pi, 0.01)
#     d = torch.sqrt(5**2/((torch.cos(theta)**2) + (torch.sin(theta)**2)) )
#     # plt.plot(d*np.cos(theta), d*np.sin(theta))
# %%


# with torch.no_grad():
#     vars = torch.Tensor([])
#     for a in test_dataset.components_distribution.alpha:
#         dist = torch.distributions.Dirichlet(a)
#         vars = torch.concat((vars,dist.variance))
#     vars = vars.reshape(-1,3)
#     for i in range(3):
#         plt.figure()
#         plt.errorbar(test_components[:,i], test_components_data.get_r()[:, i], 2*torch.sqrt(vars[:,i]), linestyle="None", capsize=2, marker= ".", alpha = 0.5) 
#         plt.plot(test_components[:,i], test_components[:,i])
#         # plt.scatter(test_components[:,i], y_hat[:,i], c = "black")
#         plt.xlabel("true mixture")
#         plt.ylabel("estimated mixture")
#         plt.savefig(f"examples/figs/oil_flow/r_{i+1}.pdf", bbox_inches = "tight")
#         plt.show()
# %%

# torch.save(bass.state_dict(),"examples/models/oil/bass.pth")
# torch.save(training_dataset.state_dict(),"examples/models/oil/training.pth")
# torch.save(test_dataset.state_dict(),"examples/models/oil/test.pth")

# bass.load_state_dict(torch.load("examples/models/oil/bass.pth"))
# training_dataset.load_state_dict(torch.load("examples/models/oil/training.pth"))
# test_dataset.load_state_dict(torch.load("examples/models/oil/test.pth"))



# %%
args = bass.beta.argsort()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # You can change this to "sans-serif" if needed
    "font.serif": ["Computer Modern Roman"],  # You can change this to your preferred serif font
    "font.size": 20  # Adjust the font size as needed
})

with torch.no_grad():
    plt.figure(figsize=(4,4))
    plt.scatter(training_dataset.mu_x[:,args[0]], training_dataset.mu_x[:,args[1]], c = training_components)
    plt.scatter(test_dataset.mu_x[:,args[0]], test_dataset.mu_x[:,args[1]], c = test_components, marker="x")
    # plt.scatter(bass.v_x[:,args[0]], bass.v_x[:,args[1]], c = "black", marker="*")
    # plt.savefig(f"examples/figs/oil_flow/classification_latent_space.pdf", bbox_inches = "tight")
    plt.show()
# %%
print("acc", torch.mean((test_dataset.get_r().argmax(dim=1) == test_labels.argmax(dim=1)).type(torch.float)))
print("LP", torch.log(test_dataset.get_r()[test_labels.type(torch.bool)]).sum())
# %%

# with open("examples/acc/oil_flow.txt", 'a') as file:
#     file.write(f"{torch.mean((test_dataset.get_r().argmax(dim=1) == test_labels.argmax(dim=1)).type(torch.float))}\n")
# with open("examples/lp/oil_flow.txt", 'a') as file:
#     file.write(f"{torch.log(test_dataset.get_r()[test_labels.type(torch.bool)]).sum()}\n")

# %%
