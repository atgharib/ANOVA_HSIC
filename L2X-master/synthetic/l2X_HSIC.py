import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random
import os
import time
import argparse

BATCH_SIZE = 1000
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# The number of key features for each dataset.
ks = {'orange_skin': 4, 'XOR': 2, 'nonlinear_additive': 4, 'switch': 5}

def create_data(datatype, n=1000):
    """
    Create train and validation datasets.
    """
    from make_data2 import generate_data
    x_train, y_train, _ = generate_data(n=n, datatype=datatype, seed=0)
    x_val, y_val, datatypes_val = generate_data(n=10**5, datatype=datatype, seed=1)

    input_shape = x_train.shape[1]

    return torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), \
           torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32), \
           datatypes_val, input_shape

def create_rank(scores, k):
    """
    Compute rank of each feature based on weight.
    """
    scores = torch.abs(scores)
    n, d = scores.shape
    ranks = []
    for i in range(n):
        # Random permutation to avoid bias due to equal weights.
        idx = torch.randperm(d)
        permuted_weights = scores[i, idx]
        permuted_rank = (-permuted_weights).argsort().argsort() + 1
        rank = permuted_rank[torch.argsort(idx)]
        ranks.append(rank)

    return torch.stack(ranks)

def compute_median_rank(scores, k, datatype_val=None):
    ranks = create_rank(scores, k)
    if datatype_val is None:
        median_ranks = torch.median(ranks[:, :k], dim=1).values
    else:
        datatype_val = datatype_val[:len(scores)]
        median_ranks1 = torch.median(ranks[datatype_val == 'orange_skin', :][:, torch.tensor([0, 1, 2, 3, 9])], dim=1).values
        median_ranks2 = torch.median(ranks[datatype_val == 'nonlinear_additive', :][:, torch.tensor([4, 5, 6, 7, 9])], dim=1).values
        median_ranks = torch.cat((median_ranks1, median_ranks2), dim=0)
    return median_ranks
# HSIC function implementation
def hsic(X, Y, sigma=1.0):
    """
    Compute the Hilbert-Schmidt Independence Criterion (HSIC) between X and Y.
    
    Args:
        X: torch.Tensor, shape (n, d1) - Input features.
        Y: torch.Tensor, shape (n, d2) - Target features.
        sigma: float - Bandwidth parameter for the Gaussian kernel.
    
    Returns:
        hsic_value: torch.Tensor - The HSIC score.
    """
    n = X.size(0)

    # Compute Gaussian kernel matrices
    KX = torch.exp(-torch.cdist(X, X, p=2)**2 / (2 * sigma**2))
    KY = torch.exp(-torch.cdist(Y, Y, p=2)**2 / (2 * sigma**2))

    # Center the kernel matrices
    H = torch.eye(n) - (1/n) * torch.ones(n, n)
    KX_centered = H @ KX @ H
    KY_centered = H @ KY @ H

    # Compute HSIC
    hsic_value = (1 / (n - 1)**2) * torch.trace(KX_centered @ KY_centered)
    return hsic_value

class SampleConcrete(nn.Module):
    """
    Layer for sampling Concrete / Gumbel-Softmax variables.
    """
    def __init__(self, tau0, k):
        super(SampleConcrete, self).__init__()
        self.tau0 = tau0
        self.k = k

    def forward(self, logits):
        # logits: [BATCH_SIZE, d]
        logits_ = logits.unsqueeze(1)  # [BATCH_SIZE, 1, d]

        batch_size = logits_.shape[0]
        d = logits_.shape[2]
        uniform = torch.rand(batch_size, self.k, d, device=logits.device).clamp(min=torch.finfo(torch.float32).eps, max=1.0)

        gumbel = -torch.log(-torch.log(uniform))
        noisy_logits = (gumbel + logits_) / self.tau0
        samples = F.softmax(noisy_logits, dim=-1)
        samples = samples.max(dim=1).values

        # Expand samples to match x's dimensions
        repeat_count = logits.shape[1] // samples.shape[1]
        samples = samples.repeat(1, repeat_count)

        # Explanation Stage output.
        threshold = logits.topk(self.k, dim=1, sorted=True).values[:, -1].unsqueeze(1)
        discrete_logits = (logits >= threshold).float()

        return samples if self.training else discrete_logits

class L2XModel(nn.Module):
    def __init__(self, input_shape, k, tau0, activation='relu'):
        super(L2XModel, self).__init__()
        activation_fn = F.relu if activation == 'relu' else F.selu

        # P(S|X) network
        self.dense1 = nn.Linear(input_shape, 100)
        self.dense2 = nn.Linear(100, 100)
        self.logits = nn.Linear(100, input_shape)
        self.sample_concrete = SampleConcrete(tau0, k)

        # q(X_S) network
        self.dense3 = nn.Linear(input_shape, 200)
        self.batch_norm1 = nn.BatchNorm1d(200)
        self.dense4 = nn.Linear(200, 200)
        self.batch_norm2 = nn.BatchNorm1d(200)
        self.output = nn.Linear(200, 2)

        self.activation_fn = activation_fn

    def forward(self, x):
        # P(S|X) network
        x1 = self.activation_fn(self.dense1(x))
        x1 = self.activation_fn(self.dense2(x1))
        logits = self.logits(x1)

        # Sampling using Concrete distribution
        samples = self.sample_concrete(logits)

        # q(X_S) network
        xs = x * samples

  


        # x = self.activation_fn(self.dense3(x))
        # x = self.batch_norm1(x)
        # x = self.activation_fn(self.dense4(x))
        # x = self.batch_norm2(x)
        
        
        ###### this section should be written
        preds = F.softmax(self.output(xs), dim=-1)
        hsic(xs, preds)

        return preds, samples


def L2X(datatype, train=True):
    x_train, y_train, x_val, y_val, datatype_val, input_shape = create_data(datatype, n=int(1e6))

    activation = 'relu' if datatype in ['orange_skin', 'XOR'] else 'selu'
    k = ks[datatype]
    tau = 0.1

    model = L2XModel(input_shape, k, tau, activation)

    if train:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        best_val_acc = 0.0
        for epoch in range(1):  # Run for 1 epoch
            model.train()
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                preds, _ = model(inputs)
                loss = criterion(preds, labels.argmax(dim=1))
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    preds, _ = model(inputs)
                    val_correct += (preds.argmax(dim=1) == labels.argmax(dim=1)).sum().item()

            val_acc = val_correct / len(val_loader.dataset)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'L2X_{datatype}.pth')
                print(f'Best model saved with accuracy: {val_acc:.4f}')

    else:
        model.load_state_dict(torch.load(f'L2X_{datatype}.pth'))

    model.eval()
    with torch.no_grad():
        _, scores = model(x_val)
    
    median_ranks = compute_median_rank(scores, k=ks[datatype], datatype_val=datatype_val)
    return median_ranks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datatype', type=str, choices=['orange_skin', 'XOR', 'nonlinear_additive', 'switch'], default='orange_skin')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    args.train = True
    median_ranks = L2X(datatype=args.datatype, train=args.train)
    print(f'Median Ranks: {median_ranks}')
