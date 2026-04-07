import os
import module as nn
from dataprocess import Dataloader
from data2 import ImgDataset
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImageModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.layers = [
			nn.Linear(784, 256, random_policy='He'),
			nn.ReLU(),
			nn.Linear(256, 128, random_policy='He'),
			nn.ReLU(),
			nn.Linear(128, 64, random_policy='He'),
			nn.ReLU(),
			nn.Linear(64, 12, random_policy='He')
		]

	def forward(self, x):
		x = x.reshape(-1, 28 * 28)
		for layer in self.layers:
			x = layer.forward(x)
		return x

	def backward(self, d):
		for layer in reversed(self.layers):
			d = layer.backward(d)
		return d

def train_epoch(model, dataloader, criterion, optimizer):
	model.train()
	total_loss = 0
	correct = 0
	total = 0

	for batch_x, batch_y in dataloader:
		pred = model.forward(batch_x)
		loss = criterion.forward(pred, batch_y)

		d_loss = criterion.backward()
		model.backward(d_loss)

		optimizer.step()
		optimizer.zero_grad()

		total_loss += loss

		pred_labels = np.argmax(pred, axis=1)
		correct += np.sum(pred_labels == batch_y)
		total += len(batch_y)

	accuracy = correct / total if total > 0 else 0
	return total_loss / len(dataloader), accuracy

def validate(model, dataloader, criterion):
	model.eval()
	total_loss = 0
	correct = 0
	total = 0

	for batch_x, batch_y in dataloader:
		batch_x = batch_x.reshape(batch_x.shape[0], -1)
		pred = model.forward(batch_x)
		loss = criterion.forward(pred, batch_y)
		total_loss += loss

		pred_labels = np.argmax(pred, axis=1)
		correct += np.sum(pred_labels == batch_y)
		total += len(batch_y)

	accuracy = correct / total if total > 0 else 0
	return total_loss / len(dataloader), accuracy

def plot_curves(train_losses, val_losses, train_accs, val_accs):
	fig, axes = plt.subplots(1, 2, figsize=(15, 5))

	axes[0].plot(train_losses, label='Train Loss', linewidth=2)
	axes[0].plot(val_losses, label='Val Loss', linewidth=2)
	axes[0].set_xlabel('Epoch', fontsize=12)
	axes[0].set_ylabel('Loss', fontsize=12)
	axes[0].set_title('Training and Validation Loss', fontsize=14)
	axes[0].legend(fontsize=10)
	axes[0].grid(True, alpha=0.3)

	axes[1].plot(train_accs, label='Train Accuracy', linewidth=2)
	axes[1].plot(val_accs, label='Val Accuracy', linewidth=2)
	axes[1].set_xlabel('Epoch', fontsize=12)
	axes[1].set_ylabel('Accuracy', fontsize=12)
	axes[1].set_title('Training and Validation Accuracy', fontsize=14)
	axes[1].legend(fontsize=10)
	axes[1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig('./checkpoint_1_2/training_curves.png', dpi=300, bbox_inches='tight')
	print("Training curves saved to ./checkpoint_1_2/training_curves.png")
	plt.show()

def main():
	np.random.seed(42)

	with open('./data_2/train.pkl', 'rb') as f:
		trainset = ImgDataset()
		trainset.load_state_dict(pickle.load(f))

	with open('./data_2/val.pkl', 'rb') as f:
		valset = ImgDataset()
		valset.load_state_dict(pickle.load(f))

	trainloader = Dataloader(trainset, batch_size=64, shuffle=True)
	valloader = Dataloader(valset, batch_size=64, shuffle=False)

	model = ImageModel()
	criterion = nn.CrossEntropyLoss()
	optimizer = nn.Adam(model, lr=0.001)

	num_epochs = 50
	print_every = 1

	train_losses = []
	val_losses = []
	train_accs = []
	val_accs = []

	epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
	for epoch in epoch_pbar:
		train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer)
		val_loss, val_acc = validate(model, valloader, criterion)

		train_losses.append(train_loss)
		val_losses.append(val_loss)
		train_accs.append(train_acc)
		val_accs.append(val_acc)

		if (epoch + 1) % print_every == 0 or epoch == 0:
			epoch_pbar.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Train Acc: {train_acc:.8f}, Val Acc: {val_acc:.8f}")
			epoch_pbar.set_postfix({'train_loss': f'{train_loss:.8f}', 'val_loss': f'{val_loss:.8f}', 'train_acc': f'{train_acc:.8f}', 'val_acc': f'{val_acc:.8f}'})

	final_train_loss, final_train_acc = validate(model, trainloader, criterion)
	final_val_loss, final_val_acc = validate(model, valloader, criterion)
	print(f"Final Train Loss: {final_train_loss:.8f}, Final Val Loss: {final_val_loss:.8f}")
	print(f"Final Train Acc: {final_train_acc:.8f}, Final Val Acc: {final_val_acc:.8f}")

	plot_curves(train_losses, val_losses, train_accs, val_accs)

	os.makedirs('./checkpoint_1_2', exist_ok=True)
	with open('./checkpoint_1_2/model.pkl', 'wb') as f:
		pickle.dump(model.state_dict(), f)
	print("Model saved to ./checkpoint_1_2/model.pkl")

if __name__ == "__main__":
	main()
