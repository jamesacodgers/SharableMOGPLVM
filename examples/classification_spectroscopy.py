# Code required for the classification hyperspecral example in the paper

# %%
import copy
from itertools import chain
import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
import torch
from src.mogplvm import MOGPLVM
import pandas as pd

from src.data import Dataset, IndependentObservations, ObservedComponents, SpectralData, VariationalClassifier, VariationalDirichletDistribution, get_init_values_for_latent_variables, predict_components_from_static_spectra
from src.utils.plot_utils import SpectraPlot
from src.utils.tensor_utils import log_linspace
from src.utils.train_utils import lbfgs_training_loop, train_bass_on_spectral_data

torch.set_default_dtype(torch.float64)

training_df = pd.read_csv("examples/data/UCRArchive_2018/Rock/Rock_TRAIN.tsv", delimiter="\t", header  = None)
test_df = pd.read_csv("examples/data/UCRArchive_2018/Rock/Rock_TEST.tsv", delimiter="\t", header  = None)

training_components_one_hot = torch.Tensor(pd.get_dummies(training_df[0]).to_numpy())
training_spectra = torch.Tensor(training_df.drop(0, axis= 1).to_numpy())
test_components_one_hot = torch.Tensor(pd.get_dummies(test_df[0]).to_numpy())
test_spectra = torch.Tensor(test_df.drop(0, axis= 1).to_numpy())


num_latent_dims = 2
combined_spectra = torch.concat([training_spectra, test_spectra])

# %%

training_spectral_data = SpectralData(torch.Tensor(training_df.columns[1:]).reshape(-1,1), training_spectra)
training_spectral_data.trim_wavelengths(0,2250)
training_spectral_data.snv()
test_spectral_data = SpectralData(torch.Tensor(test_df.columns[1:]).reshape(-1,1), test_spectra)
test_spectral_data.trim_wavelengths(0,2250)
test_spectral_data.snv()
test_spectral_data = IndependentObservations( test_spectral_data.spectra)
training_spectral_data = IndependentObservations(training_spectral_data.spectra)


training_components_object = ObservedComponents(training_components_one_hot)
test_components = VariationalDirichletDistribution(torch.ones_like(test_components_one_hot)/test_components_one_hot.shape[1])



# %%

x_init_train = get_init_values_for_latent_variables(training_spectral_data, training_spectral_data, training_components_object, num_latent_dims)
x_init_test = get_init_values_for_latent_variables(test_spectral_data, training_spectral_data, training_components_object, num_latent_dims)



# %%
training_dataset = Dataset(training_spectral_data, training_components_object, x_init_train, torch.ones_like(x_init_train))
test_dataset = Dataset(test_spectral_data, test_components, torch.zeros_like(x_init_test), torch.ones(test_spectral_data.num_data_points, num_latent_dims))

# %%
from matplotlib.cm import get_cmap

cmap = get_cmap("tab10")
training_colors = cmap(training_components_one_hot.argmax(dim = 1))
test_colors = cmap(test_components_one_hot.argmax(dim = 1))



fig, ax = plt.subplots()
for i in range(training_dataset.num_data_points):
        if training_components_one_hot[i,2]:
            ax.plot(
                    training_spectral_data.get_observations()[i], 
                    c=training_colors[i],
                    alpha= 0.5
                )
for i in range(training_dataset.num_data_points):
    if training_components_one_hot[i,2]:
        ax.plot(
            test_spectral_data.get_observations()[i], 
            c=training_colors[i],
            ls = ":",
            alpha = 0.5
        )
plt.show()
# %%

bass = MOGPLVM(
    beta = torch.ones(x_init_train.shape[1])*5, 
    # gamma=torch.ones(1)*200**2, 
    sigma2 = torch.ones(1)*1, 
    sigma2_s = torch.ones(1)*1,
    # v_x = torch.randn(16,num_latent_dims)*1,
    v_x = torch.randn(16,num_latent_dims)*1,
    # v_l= torch.linspace(training_dataset.observations.wavelengths.min(),training_dataset.observations.wavelengths.max(),30).reshape(-1,1)
)


# %%
loss = []
#%%



adam_gp = torch.optim.Adam([bass._log_beta], lr=1e-2)
adam_train = torch.optim.Adam(chain(training_dataset.parameters(), bass.parameters()), lr=1e-5)
adam_bass = torch.optim.Adam(bass.parameters(), lr=1e-2)

# %%
# sigma_2 = copy.deepcopy(bass.sigma2.item())
adam_all = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters(), test_dataset.parameters()), lr=1e-3)
# %%
noise_schedule = log_linspace(1,0.2, 100)

for i in range(1):
    for s in noise_schedule:
        print(s)
        bass.sigma2 = s
        loss.extend(train_bass_on_spectral_data(bass, [test_dataset, training_dataset], adam_all, 100))



# %%
test_components_classifier = VariationalClassifier(test_dataset.get_r())
test_dataset.components_distribution = test_components_classifier

adam_all = torch.optim.Adam(chain(bass.parameters(), training_dataset.parameters(), test_dataset.parameters()), lr=1e-3)
for i in range(1):
    for s in noise_schedule:
        print(s)
        bass.sigma2 = s
        loss.extend(train_bass_on_spectral_data(bass, [test_dataset, training_dataset], adam_all, 100))


loss.extend(train_bass_on_spectral_data(bass, [training_dataset, test_dataset], adam_all, 1000))

#  %%
while loss[-1] - loss[-2] >1:
    loss.extend(lbfgs_training_loop(bass, [training_dataset, test_dataset], chain(test_dataset.parameters(), training_dataset.parameters(), bass.parameters()), 1)) 

# %%
# with torch.no_grad():
#     plt.plot(loss)
# args = bass.beta.argsort()

# with torch.no_grad():
#     plt.figure()
#     plt.scatter(training_dataset.mu_x[:, args[0]], training_dataset.mu_x[:, args[1]], c = training_components_one_hot.argmax(axis = 1 ))
#     plt.scatter(test_dataset.mu_x[:, args[0]], test_dataset.mu_x[:,args[1]], c = test_components_one_hot.argmax(axis = 1 ), marker= "x" )
#     plt.scatter(bass.v_x[:, args[0]], bass.v_x[:, args[1]], c = "red", marker="*")
#     plt.title("2 significant Latent variables colored by temperature")
#     plt.xlabel("lv1")
#     plt.ylabel("lv2")
#     plt.show()


# %%

# training_samples = bass.get_sample_mean(training_dataset, [training_dataset, test_dataset])
# test_samples = bass.get_sample_mean(test_dataset, [training_dataset, test_dataset])
# %%
# with torch.no_grad():
#     for k in range(1):
#         plt.figure()
#         for i in range(test_dataset.num_data_points):
#             plt.vlines(bass.v_l,-5,5)
#             # if test_components_one_hot[i,k]:
#             #     plt.plot(color = "purple",alpha = 0.5)
#             #     plt.plot(test_dataset.observations.get_observations()[i,:], color = "red",alpha = 0.5)
#         for i in range(training_dataset.num_data_points):
#             plt.vlines(bass.v_l,-5,5)
#             if training_components_object.get_r()[i,k]:
#                 plt.plot(training_samples[i,:,k],color = "blue",alpha = 0.5)
#                 plt.plot(training_dataset.observations.get_observations()[i,:], color = "orange",alpha = 1)
# plt.show()
# %%

accuracy = (test_dataset.get_r().argmax(axis = 1) == test_components_one_hot.argmax(axis = 1 )).sum()
print("accuracy is ", (accuracy/ test_dataset.num_data_points).item())

print("log_prob:", torch.log(test_dataset.get_r()[test_components_one_hot.type(torch.bool)]).sum())


# with open("examples/acc/spectroscopy.txt", 'a') as file:
#     file.write(f"{torch.mean((test_dataset.get_r().argmax(dim=1) == test_components_one_hot.argmax(dim=1)).type(torch.float).mean())}\n")
# with open("examples/lp/spectroscopy.txt", 'a') as file:
#     file.write(f"{torch.log(test_dataset.get_r()[test_components_one_hot.type(torch.bool)]).sum()}\n")
