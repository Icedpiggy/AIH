import os
from utils import validate, load_dataset
from sine_model import SineModel
from sine_dataset import SineDataset
import module as nn
from utils import Dataloader
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
	testset = load_dataset(SineDataset, os.path.join(HERE, 'data'), 'test')
	testloader = Dataloader(testset, batch_size=64, shuffle=False)

	model = SineModel(hidden_dims=(16, 16, 16))
	with open(os.path.join(HERE, 'checkpoint', 'model.pkl'), 'rb') as f:
		model.load_state_dict(pickle.load(f))

	criterion = nn.MAELoss()

	test_loss = validate(model, testloader, criterion, task_type='regression')

	print(f"Test Loss: {test_loss:.10f}")

if __name__ == "__main__":
	main()
