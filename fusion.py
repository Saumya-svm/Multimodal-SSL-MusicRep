import argparse
import torch
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from random import shuffle


song_id_map = torch.load('meta/song_id_map.pt')
id_song_map = torch.load('meta/id_song_map.pt')

class Fusion(nn.Module):
	def __init__(self, enc_nodes, dec_nodes, inp_dim, unique_songs, latent_dim=64):
		super(Fusion, self).__init__()
		
		if latent_dim != enc_nodes[-1]:
			raise DimensionError('Latent dimension does not match the last encoder node.')

		enc_layers = self._make_layer(enc_nodes, inp_dim)
		self.enc = nn.Sequential(*enc_layers)

		dec_layers = self._make_layer(dec_nodes, enc_nodes[-1])
		self.dec = nn.Sequential(*dec_layers[:-1])


	def forward(self, x):
		x = self.enc(x)
		x = self.dec(x)

		x = F.log_softmax(x, dim=1)
		return x

	def _make_layer(self, enc_nodes, inp_dim):
		print(enc_nodes)
		enc_nodes.insert(0, inp_dim)
		layers = []
		for i in range(1,len(enc_nodes)):
			layers.append(nn.Linear(enc_nodes[i-1], enc_nodes[i]))
			layers.append(nn.ReLU())
		return layers


class Embedding(Dataset):
	def __init__(self, X, y, song_emb, lyric_emb, operation='con'):
		self.X = np.array(X)
		self.y = np.array(y)
		self.song_emb = song_emb
		self.lyric_emb = lyric_emb
		self.operation = operation

	def __getitem__(self, i):
		emb1 = self.song_emb[self.X[i]]
		# emb1 = torch.randn(384)
		# emb1 = self.norm(emb1) 

		lyric = self.lyric_emb[self.X[i]]
		# lyric = self.norm(lyric)
		if self.operation == 'con':
			shared = torch.cat([emb1, lyric])
		elif self.operation == 'add':
			shared = emb1+lyric
		return self.X[i], shared, song_id_map[self.y[i]]

	def __len__(self):
		return self.X.shape[0]

	def norm(self, x):
		mini = x.min()
		maxi = x.max()

		norm_x = (x-mini)/(maxi-mini)
		return norm_x

def load_lyrics():
	lyrics = torch.load('song_lyrics.pt')
	return np.array(lyrics)

def train(model, loader, test_loader):
	device = torch.device('cpu')

	opt = optim.Adam(model.parameters(), lr=0.0007)
	sched = ReduceLROnPlateau(opt, mode='min')
	loss_fn = nn.NLLLoss()
	epochs = 100

	torch.save(model.state_dict(), 'fusion_ini.pt')

	model.train()

	for epoch in range(epochs):
		epoch_loss = 0
		for i, (id,inp, out) in enumerate(loader):
			# print(inp, out)
			model.zero_grad()
			inp, out = inp.to(device), out.to(device)
			# print(id,out)
			output = model(inp)
			loss = loss_fn(output, id.to(device))
			epoch_loss += loss.item()
			loss.backward()
			opt.step()

			# val_loss = eval(model, test_loader)

		sched.step(loss)

		accuracy = eval(model, loader)
		print(f'Epoch {epoch} Loss {epoch_loss} Accuracy{accuracy}')


	torch.save(model.state_dict(), 'fusion_final1.pt')

def eval(model, loader):
	num_corrects = 0
	total = 0

	device = torch.device('cpu')

	model.eval()

	for i, (id,inp, out) in enumerate(loader):
		inp, out = inp.to(device), out.to(device)
		output = model(inp)

		pred = torch.argmax(output, dim=1)

		for i in range(id.shape[0]):
			total += 1
			if id[i] == pred[i]:
				num_corrects += 1

	print(f'Total {total} Num Corrects {num_corrects}')
	return num_corrects/total

def main():
	# load lyrics
	# lyrics = load_lyrics()
	# lyrics = np.array(lyrics)
	# indices = np.where(lyrics != 'nan')[0]

	# compute embeddings
	start = time.time()

	# store embeddings
	X = np.load('meta/X.npy')
	y = np.load('meta/y.npy')

	indices = [i for i in range(X.shape[0])]
	shuffle(indices)

	size = X.shape[0]
	train_ratio = 0.8
	train_size = int(train_ratio*size)
	train_indices = indices[:train_size]
	test_indices = indices[train_size:]

	trainX, trainY = X[train_indices], y[train_indices]
	testX, testY = X[test_indices], y[test_indices]

	song_emb = torch.load('song_embeddings_384.pt')
	lyric_emb = torch.load('lyrics_embeddings.pt')
	print(type(lyric_emb))

	train_data = Embedding(trainX, trainY, song_emb, lyric_emb)
	train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

	test_data = Embedding(testX, testY, song_emb, lyric_emb)
	test_loader = DataLoader(test_data, shuffle=True, batch_size=64)
 
	unique_songs = np.unique(X).shape[0] + 1
	print(unique_songs)
	enc_nodes = [128, 64]
	dec_nodes = [64, 128, unique_songs]
	inp_dim = 384*2
	model = Fusion(enc_nodes, dec_nodes, inp_dim, unique_songs)
	# x = torch.randn(384*2)
	# summary(model, x.shape)
	n = next(iter(train_loader))
	print(n[1])

	device = torch.device('cpu')
	model.to(device)

	print(args.train)
	if args.train:
		# sd = torch.load('fusion_final.pt')
		# model.load_state_dict(sd)
		train(model, train_loader, test_loader)
	else:
		sd1 = torch.load('fusion_final1.pt')
		model.load_state_dict(sd1)
		eval(model, train_loader)
		eval(model, test_loader)

		# sd2 = torch.load('fusion_ini.pt')
		# model.load_state_dict(sd2)
		# eval(model, train_loader)
		# eval(model, test_loader)


	end = time.time()
	print(end-start)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_version', type=str, default='paraphrase-MiniLM-L6-v2')
	parser.add_argument('--train', type=bool, default=False)
	args = parser.parse_args()

	main()

