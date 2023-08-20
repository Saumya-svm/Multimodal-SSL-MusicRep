import numpy as np
from torch import nn
from fusion import Fusion, Embedding
from torch.utils.data import DataLoader
import torch

def shared_rep(i, song_emb, lyric_emb):
	emb1 = song_emb[i]
	lyric = lyric_emb[i]
	shared = torch.cat([emb1, lyric])
	return shared

def main():
	song_id_map = torch.load('meta/song_id_map.pt')
	id_song_map = torch.load('meta/id_song_map.pt')
	X = np.load('meta/X.npy')
	y = np.load('meta/y.npy')
	song_emb = torch.load('song_embeddings_384.pt')
	lyric_emb = torch.load('lyrics_embeddings.pt')

	unique_songs = np.unique(X).shape[0] + 1
	enc_nodes = [128, 64]
	dec_nodes = [unique_songs]
	inp_dim = 384*2

	model = Fusion(enc_nodes, dec_nodes, inp_dim, unique_songs)
	sd = torch.load('fusion_final.pt')
	# sd = torch.load('fusion_ini.pt')
	model.load_state_dict(sd)

	for param in model.parameters():
		param.requires_grad = False

	train_data = Embedding(X, y, song_emb, lyric_emb)
	train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

	emb_layer = model.enc
	context = int(input())
	emb1 = emb_layer(shared_rep(context, song_emb, lyric_emb)).detach().numpy()
	# emb1 = shared_rep(context, song_emb, lyric_emb)[384:]
	t1d = np.linalg.norm(emb1)

	ele = list(map(int, input().split()))
	for i in ele:
	    emb2 = emb_layer(shared_rep(i, song_emb, lyric_emb)).detach().numpy()
	    # emb2 = shared_rep(i, song_emb, lyric_emb)[384:]
	    t2d = np.linalg.norm(emb2)
	    print(context, i, np.dot(emb1, emb2)/(t1d*t2d))


	all_list = [i for i in range(unique_songs)]
	all_sim = {}
	for i in all_list:
		try:
		    emb2 = emb_layer(shared_rep(i, song_emb, lyric_emb)).detach().numpy()
		    # emb2 = shared_rep(i, song_emb, lyric_emb)[384:]
		    t2d = np.linalg.norm(emb2)
		    all_sim[i] = (np.dot(emb1, emb2)/(t1d*t2d))
		except:
			continue

	sorted_dict = dict(sorted(all_sim.items(), key=lambda item: item[1]))
	all_keys = list(sorted_dict.keys())[::-1]
	for i in range(20):
		print(all_keys[i], sorted_dict[all_keys[i]])


if __name__ == "__main__":
	main()

