import torch
import numpy as np
from torch import nn

song_id_map = torch.load('meta/song_id_map.pt')
id_song_map = torch.load('meta/id_song_map.pt')
X = np.load('meta/X.npy')
y = np.load('meta/y.npy')
unique_songs = np.unique(X).shape[0] + 1

sd = torch.load('model_sg.pt')

class CBOW(nn.Module):
    def __init__(self, unique_songs):
        super(CBOW, self).__init__()
        self.emb = nn.Embedding(unique_songs, 64)
        self.lin = nn.Linear(64, unique_songs)
        self.act = nn.functional.log_softmax
    
    def forward(self, x):
        x = x.reshape(-1,4)
        x = self.emb(x)
        x = x.mean(1)
        x = self.lin(x)
        x = self.act(x)
        return x

model = CBOW(unique_songs)

input1 = nn.Embedding(unique_songs, 64)
# layer1 = nn.Linear(64, 256)
# activation = nn.ReLU()
layer2 = nn.Linear(64, unique_songs)
softmax = nn.functional.log_softmax
model = nn.Sequential(*[input1, layer2])
model.load_state_dict(sd)

emb_layer = model[0]
context = int(input())
t1 = torch.Tensor([context]).int()
emb1 = emb_layer(t1).detach().numpy().reshape(-1,)
t1d = np.dot(emb1,emb1)**0.5

ele = list(map(int, input().split()))
for i in ele:
    t2 = torch.Tensor([i]).int()
    emb2 = emb_layer(t2).detach().numpy().reshape(-1,)
    t2d = np.dot(emb2,emb2)**0.5
    print(context, i, np.dot(emb1, emb2)/(t1d*t2d))

fn = lambda x: song_id_map[x]