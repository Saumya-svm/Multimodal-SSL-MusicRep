import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from random import shuffle
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, LinearLR
from numpy import dot
from numpy.linalg import norm
import argparse
import wandb

neigh = torch.load('neighbours.pt')

class CrossAttention(nn.Module):
	def __init__(self):
		super(CrossAttention, self).__init__()

	def forward(self, q, k, v):
		# d_k = q.size(-1)
		# x = torch.matmul(q, k.transpose(-2,-1)/(d_k**0.5))
		# print("ca",q.shape, k.shape, x.shape)
		# x = F.softmax(x, dim=-1)
		# out = torch.matmul(x, v)
		d_k = q.size(-1)
		x = torch.matmul(q, k.transpose(-2,-1)/(d_k**0.5))
		# print("ca",q.shape, v.shape, x.shape)
		# x = F.softmax(x, dim=-1)
		out = torch.matmul(x,v)
		# print(out.shape)
		# out = 0
		return out.squeeze(1)

class SelfAttention(nn.Module):
	def __init__(self, n_emb):
		super(SelfAttention, self).__init__()
		self.key = nn.Linear(n_emb, n_emb)
		self.query = nn.Linear(n_emb, n_emb)
		self.value = nn.Linear(n_emb, n_emb)


	def forward(self, x):
		q = self.query(x)
		k = self.key(x)
		v = self.value(x)
		x = torch.matmul(q, k.transpose(-2,-1))
		x = F.softmax(x, dim=-1)
		out = torch.matmul(x, v)
		# print('hello')
		return out


class preAttention(nn.Module):
	def __init__(self, n_emb):
		super(preAttention, self).__init__()
		self.key = nn.Linear(n_emb, n_emb)
		self.query = nn.Linear(n_emb, n_emb)
		self.value = nn.Linear(n_emb, n_emb)

	def forward(self, x):
		query = self.query(x)
		key = self.key(x)
		value = self.value(x)
		return query, key, value


class AudioEncoder(nn.Module):
    def __init__(self, n_emb):
        super(AudioEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 128),  # Adjusted input size to 11
            nn.ReLU(),
            nn.Linear(128, n_emb),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layers(x)
        return out

class Encoder(nn.Module):
	def __init__(self, n_emb, n_out):
		super(Encoder, self).__init__()
		self.audioEnc = AudioEncoder(n_emb)

		self.n_emb = n_emb
		self.n_out = n_out
		self.pa1 = preAttention(n_emb)
		self.pa2 = preAttention(n_emb)
		self.ca1 = CrossAttention()
		self.ca2 = CrossAttention()
		# self.sa1 = SelfAttention(n_emb)
		# self.sa2 = SelfAttention(n_emb)
		self.ffn1 = nn.Linear(n_emb, n_out)
		self.ffn2 = nn.Linear(n_emb, n_out)
		self.t = nn.Linear(768,n_emb)
		self.relu = nn.ReLU()
		self.norm = torch.nn.LayerNorm(n_emb)
		self.batch_norm = nn.BatchNorm1d(2*n_out)

		self.ffn = nn.Linear(2*n_out, 2*n_out)

	def forward(self, text, audio):
		audio = 0.5*self.audioEnc(audio)
		text = self.relu(self.t(text))

		bs = text.shape[0]

		if args.include_attention:
			textQuery, textKey, textValue = self.pa1(text)
			audioQuery, audioKey, audioValue = self.pa2(audio)
			textQuery, textKey, textValue = textQuery.reshape(bs, 1, -1), textKey.reshape(bs, 1, -1), textValue.reshape(bs, 1, -1)
			audioQuery, audioKey, audioValue = audioQuery.reshape(bs, 1, -1), audioKey.reshape(bs, 1, -1), audioValue.reshape(bs, 1, -1)

			temp = self.ca1(textQuery, audioKey, textValue)
			# print(textValue.reshape(bs,-1).shape, temp.shape)
			textValue = self.norm(textValue.reshape(bs,-1) + temp)
			# # textValue = textValue + self.sa1(textValue)
			# text = self.norm(text+self.relu(self.ffn1(textValue)))
			text = self.relu(self.ffn1(textValue))

			# audioValue = self.norm(audioValue + self.ca2(audioQuery, textKey, audioValue))
			audioValue =  self.norm(self.ca2(audioQuery, textKey, audioValue))
			# # audioValue = audioValue + self.sa2(audioValue)
			# audio = self.relu(self.ffn2(self.norm(audioValue+audio)))
			audio = self.norm(self.relu(self.ffn2(audioValue)))
			# print(audio.shape, text.shape)

		encoded = torch.cat([audio, text], dim=-1)
		encoded = self.relu(self.batch_norm(self.ffn(encoded)))
		encoded = self.relu(self.batch_norm(self.ffn(encoded)))
		return encoded



def weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0)


class Decoder(nn.Module):
	def __init__(self, n_emb, num_classes):
		super(Decoder, self).__init__()
		self.classifier = nn.Linear(n_emb, num_classes)
		self.tar_emb = nn.Linear(num_classes, n_emb)
		weight_matrix = self.tar_emb.weight
		weight_shape = weight_matrix.shape

	def forward(self, x):
		# logits = self.classifier(x)
		# prob = torch.sigmoid(logits)
		logits = self.classifier(x)
		logits = F.log_softmax(logits, dim=1)
		# prob = F.softmax(logits, dim=1)
		return logits


class Embedding(Dataset):
	def __init__(self, X, y, audio, text, num_classes, operation='con'):
		self.X = np.array(X)
		self.y = np.array(y)
		self.audio = audio
		self.text = text
		self.size = self.X.shape[0]

		self.neg_samples = args.neg_samples
		self.num_classes = num_classes
		# print('one_hot', self.one_hot.shape)

	def __getitem__(self, i):
		song = self.audio[self.X[i]]
		lyric = self.text[self.X[i]]
		neg_idxs = torch.randint(0, self.num_classes, (self.neg_samples,))
		context = np.zeros((self.num_classes,))
		context[self.y[i]] = 1
		one_hot = np.zeros((self.neg_samples, self.num_classes))
		for i, j in enumerate(neg_idxs):
			one_hot[i][j] = 1

		# if True:
		# 	l = []
		# 	for i, j in enumerate(neg_idxs):
		# 		l.append([self.text[j], self.audio[j]])
		# 	one_hot = l

		if True:
			context = [self.text[self.y[i]], self.audio[self.y[i]]]



		return self.X[i], lyric, song, context, one_hot
		return self.X[i], lyric, song, song_id_map[self.y[i]]

	def __len__(self):
		return self.X.shape[0]

	def norm(self, x):
		mini = x.min()
		maxi = x.max()

		norm_x = (x-mini)/(maxi-mini)
		return norm_x

class SkipGram_NegSample_Loss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, 
              input_vectors, 
              output_vectors, 
              noise_vectors=None):
      
    batch_size, embed_size = input_vectors.shape
    
    input_vectors = input_vectors.view(batch_size, embed_size, 1)   # batch of column vectors
    output_vectors = output_vectors.view(batch_size, 1, embed_size) # batch of row vectors
    out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()
    
    # incorrect log-sigmoid loss
    noise_loss = 0
    noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
    noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors
    # print(noise_vectors.shape, torch.bmm(noise_vectors.neg(), input_vectors).shape, out_loss.shape, noise_loss.shape)
    # print(f"Out Loss {out_loss[0]} {torch.bmm(output_vectors, input_vectors).sigmoid()[0]}")

    return -(out_loss + noise_loss).mean()  # average batch loss

cos_sim = lambda a,b: np.dot(a, b)/(norm(a)*norm(b))

class Model(nn.Module):
	def __init__(self, n_emb, n_out, num_classes):
		super(Model, self).__init__()
		self.enc = Encoder(n_emb, n_out)
		self.dec = Decoder(2*n_out, num_classes)
		self.logging = None

	def forward(self, text, audio):
		x = self.enc(text, audio)
		out = self.dec(x)
		return out

	def forward_context(self,x):
		return self.dec.tar_emb(x)


def eval(model, loader,isLoss=False):
	num_corrects = 0
	total = 0

	device = torch.device('cpu')

	model.eval()
	model.to(device)
	criterion = SkipGram_NegSample_Loss()

	s = 0
	l = 0
	for i, (id,inp1, inp2, y, neg) in enumerate(loader):
		inp1, inp2 = inp1.to(device), inp2.to(device)
		input_emb = model.enc(inp1, inp2)
		# target_emb = model.forward_context(y.float())
		neg_emb = model.forward_context(neg.float())

		if True:
			y[0], y[1] = y[0].to(device).float(), y[1].to(device).float()
			target_emb = model.enc(y[0], y[1])
			# neg_emb = torch.rand(64, 4, 128)
			# l = []
			# neg_emb = model.enc(neg[0][0], neg[0][1])
			# temp = neg_emb.unsqueeze(1)
			# for i in range(1,len(neg)):
			# 	neg_emb = model.enc(neg[i][0], neg[i][1])
			# 	temp = torch.concat([temp, neg_emb.unsqueeze(1)], axis=1)
			# neg_emb = temp
		else:
			target_emb = model.forward_context(y)

		loss = criterion(input_emb, target_emb, neg_emb)
		# print(loss.item(), loss, id)
		l += loss.item()
		l += loss
		# print(neg_emb.shape)
		for i in range(id.shape[0]):
			total += 1
			temp = 0
			a = input_emb[i].detach().numpy()
			b = target_emb[i].detach().numpy()
			c = (abs(cos_sim(a,b)+1)/2)
			c = cos_sim(a,b)
			# print(c)

			temp +=  c
			# for j in range(args.neg_samples):
			# 	b = neg_emb[i][j].detach().numpy()
			# 	t  = ((1-(abs(1+cos_sim(a, b))/2)))/4
			# 	# print(t)
			# 	temp += t
			# # print(temp,s)

			# temp = temp/2
			s += temp
			# print(s)

	if isLoss:
		return l/len(loader)


	# print(f'Total {total} Num Corrects {num_corrects}', num_corrects/total)
	# print(s/total)
	return s/total

def save_ckpt(model, opt, sched, filename='ca_fusion', version='7'):
	ckpt = {
	'state_dict':  model.state_dict(),
	'opt': opt.state_dict(),
	'sched': sched.state_dict()
	}

	torch.save(ckpt, filename+'_'+version+'.pt')

def train(model, loader, test_loader):
	device = torch.device('cpu')
	l = {19: 0.0002, 30: 0.00007 ,45: 0.00002}
	opt = optim.Adam(model.parameters(), lr=args.learning_rate)
	sched = ReduceLROnPlateau(opt, mode='min')
	from torch.optim.lr_scheduler import CosineAnnealingLR

	sched = CosineAnnealingLR(opt,
                              T_max = 100, # Maximum number of iterations.
                             eta_min = 1e-6) # Minimum learning rate.
	if args.resume:
		print('Loading model ....')
		d = torch.load('ca_fusion'+'_'+args.version+'.pt')
		model.load_state_dict(d['state_dict'])
		opt.load_state_dict(d['opt'])
		sched.load_state_dict(d['sched'])

	# sched = LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=200)
	# sched = StepLR(opt, 
    #                step_size = 15,
    #                gamma = 0.15)
	loss_fn = torch.nn.functional.binary_cross_entropy
	loss_fn = torch.nn.CrossEntropyLoss()
	loss_fn = nn.NLLLoss()
	criterion = SkipGram_NegSample_Loss()

	torch.save(model.state_dict(), 'ca_fusion_ini.pt')

	model.train()

	min_val = 10

	for epoch in range(args.epochs):
		epoch_loss = 0
		for i, (id,inp1, inp2, y, neg) in enumerate(loader):
			model.zero_grad()
			inp1, inp2 = inp1.to(device), inp2.to(device)
			input_emb = model.enc(inp1, inp2)
			if True:
				neg_emb = model.forward_context(neg.float())

			# ind = torch.argmax(y, dim=-1)
			if True:
				y[0], y[1] = y[0].to(device).float(), y[1].to(device).float()
				target_emb = model.enc(y[0], y[1])
				# neg_emb = torch.rand(64, 4, 128)
				# neg_emb = model.enc(neg[0][0], neg[0][1])
				# temp = neg_emb.unsqueeze(1)
				# for i in range(1,len(neg)):
				# 	neg_emb = model.enc(neg[i][0], neg[i][1])
				# 	temp = torch.concat([temp, neg_emb.unsqueeze(1)], axis=1)
				# neg_emb = temp
			else:
				target_emb = model.forward_context(y)

			# output = output.to(device)
			loss = criterion(input_emb, target_emb, neg_emb)
			# print(loss)
			epoch_loss += loss.item()
			loss.backward(retain_graph=True)
			opt.step()

		epoch_loss /= len(loader)

		val_loss = eval(model, test_loader, True)
# 
		sched.step(loss)

		accuracy = eval(model, loader,False)

		if model.logging:
			model.logging.log({'epoch_loss':epoch_loss, 'val_loss': val_loss, 'accuracy': accuracy})

		print(f"Epoch {epoch} Loss {epoch_loss} Val Loss {val_loss} Accuracy{accuracy} lr {opt.param_groups[0]['lr']}")
		# if epoch in l.keys():
		# 	opt.param_groups[0]['lr'] = l[epoch]

		if args.save:
			save_ckpt(model, opt, sched, version=args.version)

		if val_loss < min_val:
			min_val = val_loss
			save_ckpt(model, opt, sched, version=args.version+'best')

def standardize(data, return_params=False):
	mean = data.mean()
	std = data.std()

	standardized_data = (data - mean) / std
	return standardized_data

def normalize(data, return_params=False):
	mean = data.max(axis=0).values
	std = data.min(axis=0).values

	standardized_data = (data - std) / (mean-std)
	return standardized_data*2-1

def prepare_config():
	config = {
	'model_version': args.version,
	'n_emb': args.n_emb, 
	'epochs': args.epochs, 
	'resume': args.resume,
	'learning_rate': args.learning_rate,
	'neg_samples': args.neg_samples,
	'include_attention': args.include_attention,
	}

	return config

def main():
	song_id_map = torch.load('meta/song_id_map1.pt')
	id_song_map = torch.load('meta/id_song_map1.pt')
	X = np.load('meta/newX.npy')
	y = np.load('meta/newY.npy')

	func = lambda i: song_id_map[i] if i != 'nan' else 0
	y = np.vectorize(func)(y)
	cos_sim = lambda a,b: np.dot(a, b)/(norm(a)*norm(b))



	text = torch.load('lyrics_embeddings.pt')
	audio = torch.tensor(np.load('norm_audio.npy'), dtype=torch.float32)
	# text = torch.load('text_encodings.pt')
	# audio = torch.load('audio_encodings.pt')

	text = torch.tensor(np.load('newText.npy'), dtype=torch.float32)
	audio = torch.tensor(np.load('newAudio.npy'), dtype=torch.float32)

	# text = normalize(text)
	audio = normalize(audio)
	print(text.shape, audio.shape)


	indices = [i for i in range(X.shape[0])]
	shuffle(indices)

	size = X.shape[0]
	train_ratio = 0.8
	train_size = int(train_ratio*size)
	train_indices = indices[:train_size]
	test_indices = indices[train_size:]


	trainX, trainY = X[train_indices], y[train_indices]
	testX, testY = X[test_indices], y[test_indices]

	unique_songs = np.unique(X).shape[0] + 1
	print(unique_songs)

	train_data = Embedding(trainX, trainY, audio, text, unique_songs)
	train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

	test_data = Embedding(testX, testY, audio, text, unique_songs)
	test_loader = DataLoader(test_data, shuffle=True, batch_size=64)
 
	n_emb = args.n_emb
	n_out = n_emb
	enc = Encoder(n_emb, n_out)
	dec = Decoder(128, unique_songs)
	model = Model(n_emb, n_out, unique_songs)
	model.apply(weights_init)

	criterion = SkipGram_NegSample_Loss()
	for i, (id,inp1, inp2, y, neg) in enumerate(train_loader):
		print(inp1.shape, y[0].shape, y[1].shape)
		# input_emb = model.enc(inp1, inp2)
		# neg_emb = model.forward_context(neg.float())
		# # print(torch.argmax(neg,axis=-1).shape)
		# # ind = torch.argmax(y, dim=-1)
		# # print('Hello', len(neg))
		# if True:
		# 	target_emb = model.enc(y[0], y[1])
		# 	# print(neg[1][1].shape)
		# 	# neg_emb = torch.rand(64, 4, 128)
		# 	# l = []
		# 	# neg_emb = model.enc(neg[0][0], neg[0][1])
		# 	# temp = neg_emb.unsqueeze(1)
		# 	# for i in range(1,4):
		# 	# 	neg_emb = model.enc(neg[i][0], neg[i][1])
		# 	# 	temp = torch.concat([temp, neg_emb.unsqueeze(1)], axis=1)

		# else:
		# 	target_emb = model.forward_context(y)
		# loss = criterion(input_emb, target_emb, neg_emb)
		# # print(loss.item())
		# audio = 0.5*model.enc.audioEnc(inp2)
		# text = model.enc.relu(model.enc.t(inp1))

		# textQuery, textKey, textValue = model.enc.pa1(text)
		# audioQuery, audioKey, audioValue = model.enc.pa2(audio)

		# textQuery, textKey, textValue = textQuery.reshape(64, 1, -1), textKey.reshape(64, 1, -1), textValue.reshape(64, 1, -1)
		# audioQuery, audioKey, audioValue = audioQuery.reshape(64, 1, -1), audioKey.reshape(64, 1, -1), audioValue.reshape(64, 1, -1)

		# temp = model.enc.ca1(textQuery, audioKey, textValue)
		# print("cross ", temp.shape, textValue.shape)
		# temp = model.enc.ca2(audioQuery, textKey, audioValue)
		# print("cross ", temp.shape, textValue.shape)

		# print(textQuery.shape, audioKey.T.shape)
		# textValue = model.enc.norm(textValue + model.enc.ca1(textQuery, audioKey, textValue))
		# return
		break

	# return

	# decoded = dec(encoded)
	

	config = prepare_config()
	if not args.run_name:
		args.run_name = ''
		for k, v in config.items():
			args.run_name += k+'_'+str(v)+'_'
	print(args.run_name)
	# return

	if args.train:
		if args.log:
			wandb.init(project='recsys', config=config, name=args.run_name)
			model.logging = wandb
		train(model, train_loader, test_loader)
		if model.logging:
			model.logging.finish()

	# eval(model, test_loader)


	# d = torch.load('ca_fusion'+'_'+args.version+'.pt')
	# model.load_state_dict(d['state_dict'])
	# for i, (id,inp1, inp2, y, neg) in enumerate(train_loader):
	# 	input_emb = model.enc(inp1, inp2)
	# 	# print(y.shape)
	# 	target_emb = model.forward_context(y.float())
	# 	# print(target_emb.shape, neg.shape)
	# 	neg_emb = model.forward_context(neg.float())
	# 	# print(neg_emb.shape)
	# 	total = 0
	# 	s = 0
	# 	for i in range(id.shape[0]):
	# 		total += 1
	# 		temp = 0
	# 		a = input_emb[i].detach().numpy()
	# 		b = target_emb[i].detach().numpy()
	# 		c = cos_sim(a,b)**2
	# 		temp +=  c
	# 		for j in range(args.neg_samples):
	# 			temp += (1-cos_sim(input_emb[i].detach().numpy(), neg_emb[i][j].detach().numpy()))**2
	# 		temp = temp/2
	# 		s += temp
	# 	print(s/total, total)
	# 	break

	if args.eval:
		d = torch.load('ca_fusion'+'_'+args.version+'best.pt')
		model.load_state_dict(d['state_dict'])
		print(eval(model, train_loader, True))
		print(eval(model, train_loader, False))
		print(eval(model, test_loader, True))
		model.eval()
		tensor = model.enc(text, audio)
		tensor_normalized = tensor / torch.norm(tensor, dim=1, keepdim=True)

		# print(eval(model, train_loader, True))
		# print(eval(model, train_loader, False))
		# print(eval(model, test_loader, True))
		# # Compute cosine similarity using matrix multiplication
		cos_sim_matrix = torch.matmul(tensor_normalized, tensor_normalized.t())

		# print(cos_sim_matrix.shape)
		# i = 2231
		top_val , top_idx = torch.topk(cos_sim_matrix[i], 10, largest=True)
		# # print(eval(model, train_loader, True))
		# # print(eval(model, train_loader, False))
		# # print(eval(model, test_loader, True))
		print(top_val, top_idx)
		k = list(neigh[i])+[223,828,3477, 3324, 8476, 4867]
		
		# # k = [3588, 4081, 4375, 4753, 4867, 5778, 5819, 6036, 7656, 8237, 8661,9129, 9397]
		# l = [i for i in range(len(k))]
		s = 0
		for j in k:
			embi = audio[i] 
			lyrici = text[i]
			sharedi = torch.cat([embi, lyrici])

			embj = audio[j] 
			lyricj = text[j]
			sharedj = torch.cat([embj, lyricj])
			# print(lyrici.shape, embi.shape)

			a, b = model.enc(lyrici.reshape(1,-1), embi.reshape(1,-1)).detach().numpy()[0], model.enc(lyricj.reshape(1,-1), embj.reshape(1,-1)).detach().numpy()[0]
			# print(a,b)
			t1 = torch.Tensor(np.zeros((1, 8633)))
			t2 = torch.Tensor(np.zeros((1, 8633)))
			t1[0][i] = 1
			t2[0][j] = 1
			# a, b = model.enc(t1).detach().numpy()[0], model.enc(t2).detach().numpy()[0]
			cos_sim1 = dot(a, b)/(norm(a)*norm(b))
			a, b = audio[i], audio[j]
			cos_sim2 = dot(a, b)/(norm(a)*norm(b))
			a, b = text[i], text[j]
			cos_sim3 = dot(a, b)/(norm(a)*norm(b))


			# print('hello',id_song_map[i], id_song_map[j.item()],i,j.item(),cos_sim1, cos_sim2, cos_sim3)
			print('hello',i,j,cos_sim1, cos_sim2, cos_sim3)
			s += cos_sim1
		# print(s)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_version', type=str, default='7')
	parser.add_argument('--train', type=int, default=0)
	parser.add_argument('--resume', type=int, default=0)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--version', type=str, default='wa')
	parser.add_argument('--eval', type=int, default=1)
	parser.add_argument('--include_attention', type=int, default=0)
	parser.add_argument('--n_emb', type=int, default=64)
	parser.add_argument('--n_out', type=int, default=64)
	parser.add_argument('--neg_samples', type=int, default=4)
	parser.add_argument('--save', type=int, default=1)
	parser.add_argument('--log', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=0.0007)
	parser.add_argument('--project', type=str, default='recsys')
	parser.add_argument('--run_name', type=str, default='')
	parser.add_argument('--train_split', type=float, default=0.8)
	args = parser.parse_args()

	main()