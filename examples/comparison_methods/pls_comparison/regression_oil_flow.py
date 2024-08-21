# %%
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from matplotlib import pyplot as plt

training_spectra = np.loadtxt(fname='examples/data/3phData/DataTrn.txt')
training_labels = np.loadtxt(fname='examples/data/3phData/DataTrnLbls.txt')
training_components = np.loadtxt(fname='examples/data/3phData/DataTrnFrctns.txt')
training_components = np.hstack([training_components, (1 - training_components.sum(axis=1)).reshape(-1,1)])
test_spectra = np.loadtxt(fname='examples/data/3phData/DataTrn.txt')
test_labels = np.loadtxt(fname='examples/data/3phData/DataTrnLbls.txt')
test_components = np.loadtxt(fname='examples/data/3phData/DataTrnFrctns.txt')
test_components = np.hstack([test_components, (1 - test_components.sum(axis=1)).reshape(-1,1)])

num_training = 1000

train_x = training_spectra[:num_training]
train_y = training_components[:num_training]

test_x = test_spectra
test_y = test_components


nA = 12
msep = np.zeros([nA,3])

d1,d2, r1, r2 =  train_test_split(train_x, train_y)

for i in range(nA):
    pls_reg = PLSRegression(i+1)
    pls_reg.fit(d1, r1)
    # y_hat = pls_reg.predict(test_spectra.spectra)
    y_hat = pls_reg.predict(d2)
    # plt.figure()
    for j in range(3):
        # msep[i,j] = ((test_components_tensor[:,j] - y_hat[:,j])**2).sum()
        msep[i,j] = ((r2[:,j] - y_hat[:,j])**2).sum()
    print(msep)

min_err_components = np.argmin(msep.sum(axis=1))

pls_reg = PLSRegression(min_err_components.item()+1)
pls_reg.fit(training_spectra, training_components)
y_hat = pls_reg.predict(test_x)
fig, ax = plt.subplots(3)
for i in range(3):
    ax[i].scatter(y_hat, test_y)
plt.show()


print("msep:", ((y_hat - test_y)**2).mean())


with open("examples/comparison_methods/pls_comparison/regression_oil_flow.txt", 'a') as file:
    file.write(f"{((y_hat - test_y)**2).mean()}\n")