import math
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, train_test_split
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
num_pca_dims = 10
all_spectra_dataset = SpectralData(data["wl"], all_spectra_tensor)
all_spectra_dataset.trim_wavelengths(800,1000)
all_spectra_dataset.snv()
# all_spectra_dataset.spectra, _, _ = torch.pca_lowrank(all_spectra_dataset.spectra, num_pca_dims)





all_temperatures = torch.vstack([
    data["temper30"],
    data["temper40"],
    data["temper50"],
    data["temper60"],
    data["temper70"],
])

all_components = data["conc"].repeat(5,1)

train_x, test_x, train_y, test_y = train_test_split(all_spectra_dataset.spectra.numpy(), all_components.numpy(), test_size=0.5)


nA = 10
msep = np.zeros([nA,3])

kf = KFold(n_splits=10, shuffle=True)

for i in range(nA):
    for train_idx, test_idx in kf.split(train_x):
        pls_reg = PLSRegression(i+1)
        pls_reg.fit(train_x[train_idx], train_y[train_idx])
        # y_hat = pls_reg.predict(test_spectra.spectra)
        y_hat = pls_reg.predict(train_x[test_idx])
        msep[i] += ((y_hat - train_y[test_idx])**2).sum()

min_err_components = np.argmin(msep.sum(axis=1))

pls_reg = PLSRegression(min_err_components.item()+1)
pls_reg.fit(train_x, train_y)
y_hat = pls_reg.predict(test_x)
fig, ax = plt.subplots(3)
for i in range(3):
    ax[i].scatter(y_hat, test_y)
plt.show()

print("msep:", np.mean((y_hat - test_y)**2))

# %%

with open("examples/comparison_methods/pls_comparison/regression_spectroscopy.txt", 'a') as file:
    file.write(f"{np.mean((y_hat - test_y)**2)}\n")
