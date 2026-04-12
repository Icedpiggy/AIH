import os
from utils import validate, load_dataset
from mlp_image_model import ImageModel
from image_dataset import ImgDataset
import module as nn
from utils import Dataloader
from tqdm import tqdm
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
	testset = load_dataset(ImgDataset, os.path.join(HERE, 'data'), 'test')
	testloader = Dataloader(testset, batch_size=64, shuffle=False)

	model = ImageModel(dropout_rate=0.2)
	with open(os.path.join(HERE, 'checkpoint_mlp', 'model.pkl'), 'rb') as f:
		model.load_state_dict(pickle.load(f))
		print('load')

	criterion = nn.CrossEntropyLoss()

	test_loss, test_acc = validate(model, testloader, criterion, task_type='classification')

	print(f"Test Loss: {test_loss:.8f}, Test Acc: {test_acc:.8f}")

if __name__ == "__main__":
	main()
