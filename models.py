import torch
import torch.nn as nn
import torch.nn.functional as F


class Glove2Sparse(nn.Module):
    def __init__(self):
        super(Glove2Sparse, self).__init__()
        self.layer1 = nn.Linear(100, 500)
        self.layer2 = nn.Linear(500, 5000)
        self.layer3 = nn.Linear(5000, 500)
        self.layer4 = nn.Linear(500, 100)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x


class Glove2kSparse(nn.Module):
    def __init__(self):
        super(Glove2kSparse, self).__init__()
        self.encoder = nn.Linear(100, 1000)
        self.decoder = nn.Linear(1000, 100)

    def encode(self, x):
        x = self.encoder(x)
        topk = torch.topk(x, 25).indices
        debug_count = 0
        for index in range(len(x)):
            if index not in topk:
                x[index] = 0
                debug_count += 1
        return x

    def forward(self, x):  # Unsqueeze / squeeze?
        return self.decoder(self.encode(x))

    def view_grads(self):
        enc_grads = net.encoder.weight.grad.resize(100000)
        dec_grads = net.decoder.weight.grad.resize(100000)
        return sparse_vec_to_dict(torch.cat([enc_grads, dec_grads]))
