import os
from utils import train_epoch, validate, plot_training_curves, load_dataset
from mlp_image_model import ImageModel
from image_dataset import ImgDataset, prepare_image_data
import module as nn
from utils import Dataloader
from utils.backend import numpy
from tqdm import tqdm
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
	numpy.random.seed(42)
	prepare_image_data(os.path.join(HERE, '..', 'data_2'), os.path.join(HERE, 'data'))

	trainset = load_dataset(ImgDataset, os.path.join(HERE, 'data'), 'train')
	valset = load_dataset(ImgDataset, os.path.join(HERE, 'data'), 'val')
	trainloader = Dataloader(trainset, batch_size=64, shuffle=True)
	valloader = Dataloader(valset, batch_size=64, shuffle=False)

	model = ImageModel(hidden_dims=(512, 512, 512), dropout_rate=0.2)
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
		train_loss, train_acc = train_epoch(model, trainloader, criterion, optimizer, task_type='classification')
		val_loss, val_acc = validate(model, valloader, criterion, task_type='classification')

		train_losses.append(train_loss)
		val_losses.append(val_loss)
		train_accs.append(train_acc)
		val_accs.append(val_acc)

		if (epoch + 1) % print_every == 0 or epoch == 0:
			epoch_pbar.write(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Train Acc: {train_acc:.8f}, Val Acc: {val_acc:.8f}")
			epoch_pbar.set_postfix({'train_loss': f'{train_loss:.8f}', 'val_loss': f'{val_loss:.8f}', 'train_acc': f'{train_acc:.8f}', 'val_acc': f'{val_acc:.8f}'})

	final_train_loss, final_train_acc = validate(model, trainloader, criterion, task_type='classification')
	final_val_loss, final_val_acc = validate(model, valloader, criterion, task_type='classification')
	print(f"Final Train Loss: {final_train_loss:.8f}, Final Val Loss: {final_val_loss:.8f}")
	print(f"Final Train Acc: {final_train_acc:.8f}, Final Val Acc: {final_val_acc:.8f}")

	checkpoint_dir = os.path.join(HERE, 'checkpoint_mlp')
	os.makedirs(checkpoint_dir, exist_ok=True)
	plot_training_curves(train_losses, val_losses, train_accs, val_accs, checkpoint_dir)

	with open(os.path.join(checkpoint_dir, 'model.pkl'), 'wb') as f:
		pickle.dump(model.state_dict(), f)
	print(f"Model saved to {checkpoint_dir}/model.pkl")

if __name__ == "__main__":
	main()
