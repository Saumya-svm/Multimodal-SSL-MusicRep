import torch
import numpy as np
import argparse
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchsummary import summary
import torch.optim.lr_scheduler as lr_scheduler
from random import shuffle

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
        # x = self.act(x, dim=1)
        return x

class Song(Dataset):
	def __init__(self , X, y, song_id_map):
		print(X.shape, y.shape)
		self.X = X
		self.y = y
		self.map = song_id_map

	def __getitem__(self, x):
		return torch.Tensor([self.X[x]]).int(), torch.Tensor([self.y[x]]).long()

	def __len__(self):
		return self.X.shape[0]


def main():
	emb_dim = 64
	emb = torch.randn(64, emb_dim)

	print(emb.shape)

	song_id_map = torch.load('meta/song_id_map.pt')
	id_song_map = torch.load('meta/id_song_map.pt')
	X = np.load('meta/X.npy')
	y = np.load('meta/y.npy')
	unique_songs = np.unique(X).shape[0] + 1
	y = np.vectorize(lambda x: song_id_map[x])(y)
	
	if args.method == 'sg':
		input1 = nn.Embedding(unique_songs, 64)
		# layer1 = nn.Linear(64, 256)
		# activation = nn.ReLU()
		layer2 = nn.Linear(64, unique_songs)
		softmax = nn.functional.log_softmax
		model = nn.Sequential(*[input1, layer2])
	else:
		model = CBOW(unique_songs)
		X = np.load('meta/new_x.npy', allow_pickle=True)
		y = np.load('meta/new_y.npy', allow_pickle=True)


	indices = [i for i in range(X.shape[0])]
	shuffle(indices)
	train_size = int(0.8*X.shape[0])
	train_indices = indices[:train_size]
	test_indices = indices[train_size:]

	train_dataset = Song(X[train_indices], y[train_indices], song_id_map)
	train_loader = DataLoader(train_dataset, shuffle=False, batch_size=32)
	test_dataset = Song(X[test_indices], y[test_indices], song_id_map)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

	print(model)

	#loss
	loss_fn = nn.NLLLoss()

	#opt
	opt = optim.Adam(model.parameters(), lr=0.006)
	mult = lambda epoch : 0.95
	lr_sched = lr_scheduler.MultiplicativeLR(opt, mult)
	lr_sched = lr_scheduler.ReduceLROnPlateau(opt, 'min')

	state_dict = torch.load(f'model_{args.method}.pt')
	model.load_state_dict(state_dict)
	device = torch.device('mps')
	model.to(device)
	epochs = 300
	#training loop

	model.train()

	for i in range(epochs):
		epoch_loss = 0
		for j, (inp,out) in enumerate(train_loader):
			model.zero_grad()
			inp = inp.reshape(-1,).to(device)
			out = out.reshape(-1,).to(device)
			# print(inp.shape)
			output = model(inp)
			output = F.log_softmax(output, dim=1)
			loss = loss_fn(output, out)
			loss.backward()
			opt.step()

			epoch_loss += loss.item()


		lr_sched.step(epoch_loss)
		print(f'Epoch {i} Loss {epoch_loss}')
		

	torch.save(model.state_dict(), f'model_{args.method}.pt')

	model.eval()
	num_corrects = 0
	total = 0
	for i, (inp,out) in enumerate(train_loader):
		inp = inp.reshape(-1,).to(device)
		out = out.reshape(-1,).to(device)
		output = model(inp)
		output = output.argmax(dim=1)
		for j in range(output.shape[0]):
			total += 1
			if output[j].item() == out[j].item():
				num_corrects += 1

	

	print(total, num_corrects, num_corrects/total)

	num_corrects = 0
	total = 0

	for i, (inp,out) in enumerate(test_loader):
		inp = inp.reshape(-1,).to(device)
		out = out.reshape(-1,).to(device)
		output = model(inp)
		output = output.argmax(dim=1)

		for j in range(output.shape[0]):
			total += 1
			if output[j].item() == out[j].item():
				num_corrects += 1
	

	print(total, num_corrects, num_corrects/total)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--unique_songs', type=int, default=1)
	parser.add_argument('--method', type=str, default='sg')

	args = parser.parse_args()
	main()
