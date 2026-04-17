import os
from utils import train_epoch, validate, plot_loss_curves, load_dataset
from sine_model import SineModel
from sine_dataset import SineDataset
import module as nn
from utils import Dataloader
from utils.backend import numpy
from tqdm import tqdm
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
	numpy.random.seed(42)

	trainset = load_dataset(SineDataset, os.path.join(HERE, 'data'), 'train')
	valset = load_dataset(SineDataset, os.path.join(HERE, 'data'), 'val')
	trainloader = Dataloader(trainset, batch_size=64, shuffle=True)
	valloader = Dataloader(valset, batch_size=64, shuffle=False)

	model = SineModel(hidden_dims=(16, 16, 16), activation=nn.Tanh, random_policy='Xavier')
	criterion = nn.MSELoss()
	optimizer = nn.Adam(model, lr=0.0005)

	num_epochs = 100
	print_every = 1

	train_losses = []
	val_losses = []

	epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
	for epoch in epoch_pbar:
		train_loss = train_epoch(model, trainloader, criterion, optimizer, task_type='regression')
		val_loss = validate(model, valloader, criterion, task_type='regression')

		train_losses.append(train_loss)
		val_losses.append(val_loss)

		if (epoch + 1) % print_every == 0 or epoch == 0:
			epoch_pbar.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.10f}, Val Loss: {val_loss:.10f}")
			epoch_pbar.set_postfix({'train_loss': f'{train_loss:.10f}', 'val_loss': f'{val_loss:.10f}'})

	final_train_loss = validate(model, trainloader, criterion, task_type='regression')
	final_val_loss = validate(model, valloader, criterion, task_type='regression')
	print(f"Final Train Loss: {final_train_loss:.10f}, Final Val Loss: {final_val_loss:.10f}")

	checkpoint_dir = os.path.join(HERE, 'checkpoint')
	os.makedirs(checkpoint_dir, exist_ok=True)
	plot_loss_curves(train_losses, val_losses, checkpoint_dir)

	with open(os.path.join(checkpoint_dir, 'model.pkl'), 'wb') as f:
		pickle.dump(model.state_dict(), f)
	print(f"Model saved to {checkpoint_dir}/model.pkl")

if __name__ == "__main__":
	main()
