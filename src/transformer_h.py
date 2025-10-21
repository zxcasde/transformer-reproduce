import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding()