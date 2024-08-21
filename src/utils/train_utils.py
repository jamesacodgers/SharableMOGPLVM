import torch
from typing import List

from src.mogplvm import MOGPLVM
from src.data import Dataset, SpectralData

def train_bass_on_spectral_data(
        model: MOGPLVM, 
        data: List[Dataset], 
        optimizer, 
        epochs: int
    ):
    elbo_list = []
    for epoch in range(epochs):
        # Compute the loss        
        loss =  - model.elbo(data)
        if torch.isnan(loss):
            print(list(model.named_parameters()))
        elbo_list.append(-loss.detach().numpy())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print the loss after every 100 epochs
        
        if (epoch + 1) % 50 == 0:
            print("Epoch [{}/{}], ELBO: {:.4f}".format(epoch + 1, epochs, -loss.item()))

    return elbo_list

def lbfgs_training_loop(
    model: MOGPLVM, 
    data: List[SpectralData], 
    params,
    epochs: int
):
    optimizer = torch.optim.LBFGS(
                                params, 
                                history_size=100, 
                                max_iter=100, 
                                line_search_fn="strong_wolfe"
                                )

    def closure():
        optimizer.zero_grad()
        loss = - model.elbo(data)  # Forward pass
        loss.backward()  # Backpropagate the gradients
        return loss

    loss_list = []
    for epoch in range(epochs):
        # inducing_T2 = (model.v_x**2).sum(axis=1)
        # active = inducing_T2< 25
        # replacement = torch.randn()
        # model.v_x = torch.nn.Parameter(model.v_x[active])


        loss_list.append(-optimizer.step(closure))
        
        print("Epoch [{}/{}], ELBO: {:.4f}".format(epoch + 1, epochs, loss_list[-1]))

    return loss_list



# %%
class TestModule(torch.nn.Module):
    def __init__(self, x, y):
        super(TestModule, self).__init__()
        self.x = torch.nn.Parameter(x)
        self.y = torch.nn.Parameter(y)

    def forward(self):
        return (self.x**2).sum() + self.y**2 
# %%

x_init = torch.Tensor([1,2])
y_init = torch.Tensor([1])

model = TestModule(x_init, y_init)

loss = model()

params = list(model.parameters())


param_vec = torch.cat([param.view(-1) for param in params])
new_params = [torch.empty_like(param) for param in model.parameters()]
start = 0 
for i,p in enumerate(new_params):
    end = start + p.numel()
    param_vec

    
# %%
