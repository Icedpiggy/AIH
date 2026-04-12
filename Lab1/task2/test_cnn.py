import os
from utils import validate, load_dataset
from cnn_image_model import CNNModel
from image_dataset import ImgDataset
import module as nn
from utils import Dataloader
from tqdm import tqdm
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
	testset = load_dataset(ImgDataset, os.path.join(HERE, 'data'), 'test')
	testloader = Dataloader(testset, batch_size=64, shuffle=False)

	model = CNNModel(num_classes=12, dropout_rate=0.3)
	with open(os.path.join(HERE, 'checkpoint_cnn', 'model.pkl'), 'rb') as f:
		model.load_state_dict(pickle.load(f))
		print('load')

	criterion = nn.CrossEntropyLoss()

	test_loss, test_acc = validate(model, testloader, criterion, task_type='classification')

	print(f"Test Loss: {test_loss:.8f}, Test Acc: {test_acc:.8f}")

if __name__ == "__main__":
	main()
