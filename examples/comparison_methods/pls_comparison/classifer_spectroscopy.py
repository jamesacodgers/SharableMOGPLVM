

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
import numpy as np

from src.data import SpectralData



training_df = pd.read_csv("examples/data/UCRArchive_2018/Rock/Rock_TRAIN.tsv", delimiter="\t", header  = None)
test_df = pd.read_csv("examples/data/UCRArchive_2018/Rock/Rock_TEST.tsv", delimiter="\t", header  = None)

training_components_one_hot = pd.get_dummies(training_df[0]).to_numpy()
training_spectra = torch.Tensor(training_df.drop(0, axis= 1).to_numpy())
test_components_one_hot = pd.get_dummies(test_df[0]).to_numpy()
test_spectra = torch.Tensor(test_df.drop(0, axis= 1).to_numpy())


combined_spectra = torch.concat([training_spectra, test_spectra])

# %%

training_spectral_data = SpectralData(torch.Tensor(training_df.columns[1:]).reshape(-1,1), training_spectra)
training_spectral_data.snv()
training_spectral_data.trim_wavelengths(0,2250)
# training_spectral_data = IndependentObservations(training_spectral_data.spectra)
test_spectral_data = SpectralData(torch.Tensor(test_df.columns[1:]).reshape(-1,1), test_spectra)
test_spectral_data.snv()
test_spectral_data.trim_wavelengths(0,2250)





train_x = training_spectral_data.spectra
train_y = training_components_one_hot

test_x = test_spectral_data.spectra
test_y = test_components_one_hot


nA = 12
msep = np.zeros([nA])

kf = KFold(n_splits=10, shuffle=True)

for i in range(nA):
    for train_idx, test_idx in kf.split(train_x):
        pls_reg = PLSRegression(i+1)
        pls_reg.fit(train_x[train_idx], train_y[train_idx])
        # y_hat = pls_reg.predict(test_spectra.spectra)
        y_hat = pls_reg.predict(train_x[test_idx])
        msep[i] += ((y_hat - train_y[test_idx])**2).sum()

min_err_components = np.argmin(msep)

pls_reg = PLSRegression(min_err_components.item()+1)
pls_reg.fit(train_x, train_y)
y_hat = pls_reg.predict(test_x)
fig, ax = plt.subplots(3)
for i in range(3):
    ax[i].scatter(test_y, y_hat)
plt.show()


print("acc:", np.sum(np.argmax(y_hat,axis=1) == np.argmax(test_y, axis = 1 ))/test_y.shape[0])

# %%

with open("examples/comparison_methods/pls_comparison/classifier_spectroscopy.txt", 'a') as file:
    file.write(f"{np.sum(np.argmax(y_hat,axis=1) == np.argmax(test_y, axis = 1 ))/test_y.shape[0]}\n")