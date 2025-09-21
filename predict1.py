import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
import numpy as np
import pyro.poutine as poutine


# Bayesian Neural Network definition
class BNN(PyroModule):
    def __init__(self, in_features=34, hidden=64, out_features=2):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_features, hidden)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden, in_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden]).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](hidden, hidden)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden, hidden]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden]).to_event(1))
        
        self.out = PyroModule[nn.Linear](hidden, out_features)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, hidden]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        # ensure x is 2D
        if x.dim() == 3:  # flatten if 3D [num_samples, batch, features]
            x = x.view(-1, x.shape[-1])
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        logits = self.out(x)
        with pyro.plate("data", size=logits.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

def main():
    # Instantiate BNN
    bnn = BNN()

    # Ask user for 34 feature values
    print("=== Bayesian Landslide Prediction ===")
    raw_input = input("You need to enter 34 feature values:\n> ")
    features = np.array([float(x.strip()) for x in raw_input.split(",")])
    if len(features) != 34:
        raise ValueError("You must enter exactly 34 features.")
    
    X_new = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # shape: [1, 34]

    # Create guide and predictive
    guide = AutoDiagonalNormal(poutine.block(bnn, hide=['obs']))
    predictive = Predictive(bnn, guide=guide, num_samples=500)

    # Get predictions
    samples = predictive(X_new)["obs"]  # shape: [num_samples, batch]
    # Compute probabilities for class 1 (landslide)
    landslide_prob = (samples == 1).float().mean().item()
    
    print(f"Predicted probability of landslide: {landslide_prob:.4f}")

if __name__ == "__main__":
    main()
