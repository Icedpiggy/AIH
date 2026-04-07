import module as nn
from dataprocess import Dataloader
from data1 import SineDataset
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class SineModel(nn.Module):
	def __init__(self, hidden_dim=16):
		super().__init__()
		self.layers = [
			nn.Linear(1, hidden_dim, random_policy='He'),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim, random_policy='He'),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim, random_policy='He'),
			nn.ReLU(),
			nn.Linear(hidden_dim, 1, random_policy='He')
		]

	def forward(self, x):
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
	for batch_x, batch_y in dataloader:
		pred = model.forward(batch_x)
		loss = criterion.forward(pred, batch_y)

		d_loss = criterion.backward()
		model.backward(d_loss)

		optimizer.step()
		optimizer.zero_grad()

		total_loss += loss

	return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
	model.eval()
	total_loss = 0
	for batch_x, batch_y in dataloader:
		pred = model.forward(batch_x)
		loss = criterion.forward(pred, batch_y)
		total_loss += loss

	return total_loss / len(dataloader)

def plot_loss_curves(train_losses, val_losses):
	plt.figure(figsize=(10, 6))
	plt.plot(train_losses, label='Train Loss', linewidth=2)
	plt.plot(val_losses, label='Val Loss', linewidth=2)
	plt.xlabel('Epoch', fontsize=12)
	plt.ylabel('Loss', fontsize=12)
	plt.yscale('log')
	plt.title('Training and Validation Loss (Log Scale)', fontsize=14)
	plt.legend(fontsize=10)
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig('./checkpoint_1_1/loss_curve.png', dpi=300, bbox_inches='tight')
	print("Loss curve saved to ./checkpoint_1_1/loss_curve.png")
	plt.show()

def main():
	np.random.seed(42)

	with open('./data_1/train.pkl', 'rb') as f:
		trainset = SineDataset()
		trainset.load_state_dict(pickle.load(f))

	with open('./data_1/val.pkl', 'rb') as f:
		valset = SineDataset()
		valset.load_state_dict(pickle.load(f))

	trainloader = Dataloader(trainset, batch_size=64, shuffle=True)
	valloader = Dataloader(valset, batch_size=64, shuffle=False)

	model = SineModel(hidden_dim=16)
	criterion = nn.MSELoss()
	optimizer = nn.Adam(model, lr=0.0005)

	num_epochs = 100
	print_every = 1

	train_losses = []
	val_losses = []

	epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
	for epoch in epoch_pbar:
		train_loss = train_epoch(model, trainloader, criterion, optimizer)
		val_loss = validate(model, valloader, criterion)

		train_losses.append(train_loss)
		val_losses.append(val_loss)

		if (epoch + 1) % print_every == 0 or epoch == 0:
			epoch_pbar.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}")
			epoch_pbar.set_postfix({'train_loss': f'{train_loss:.10f}', 'val_loss': f'{val_loss:.10f}'})

	final_train_loss = validate(model, trainloader, criterion)
	final_val_loss = validate(model, valloader, criterion)
	print(f"Final Train Loss: {final_train_loss:.10f}, Final Val Loss: {final_val_loss:.10f}")

	plot_loss_curves(train_losses, val_losses)

	x_test = np.array([[-np.pi], [-np.pi/2], [0], [np.pi/2], [np.pi]])
	y_true = np.sin(x_test)

	model.eval()
	y_pred = model.forward(x_test)

	print("\nTest Results:")
	for i in range(len(x_test)):
		error = abs(y_true[i][0] - y_pred[i][0])
		print(f"x={x_test[i][0]:.10f}: true={y_true[i][0]:.10f}, pred={y_pred[i][0]:.10f}, error={error:.10f}")

	with open('./checkpoint_1_1/model.pkl', 'wb') as f:
		pickle.dump(model.state_dict(), f)
	print("Model saved to ./checkpoint_1_1/model.pkl")

if __name__ == "__main__":
	main()
