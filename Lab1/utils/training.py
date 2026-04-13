import numpy as np
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, task_type='classification'):
	model.train()
	total_loss = 0

	if task_type == 'regression':
		pbar = tqdm(dataloader, leave=True, ncols=120, disable=False)
		for batch_x, batch_y in pbar:
			pred = model(batch_x)
			loss = criterion(pred, batch_y)

			d_loss = criterion.backward()
			model.backward(d_loss)

			optimizer.step()
			optimizer.zero_grad()

			total_loss += loss

			pbar.set_postfix({'loss': f'{loss:.8f}'})

		return total_loss / len(dataloader)

	else:
		correct = 0
		total = 0

		pbar = tqdm(dataloader, leave=True, ncols=120, disable=False)
		for batch_x, batch_y in pbar:
			pred = model(batch_x)
			loss = criterion(pred, batch_y)

			d_loss = criterion.backward()
			model.backward(d_loss)

			optimizer.step()
			optimizer.zero_grad()

			total_loss += loss

			pred_labels = np.argmax(pred, axis=1)
			correct += np.sum(pred_labels == batch_y)
			total += len(batch_y)

			pbar.set_postfix({'loss': f'{loss:.8f}', 'acc': f'{np.sum(pred_labels == batch_y) / len(batch_y):.8f}'})

		accuracy = correct / total if total > 0 else 0
		return total_loss / len(dataloader), accuracy


def validate(model, dataloader, criterion, task_type='classification'):
	model.eval()
	total_loss = 0

	if task_type == 'regression':
		pbar = tqdm(dataloader, leave=True, ncols=120, disable=False)
		for batch_x, batch_y in pbar:
			pred = model(batch_x)
			loss = criterion(pred, batch_y)
			total_loss += loss
			pbar.set_postfix({'loss': f'{loss:.8f}'})

		return total_loss / len(dataloader)

	else:
		correct = 0
		total = 0

		pbar = tqdm(dataloader, leave=True, ncols=120, disable=False)
		for batch_x, batch_y in pbar:
			pred = model(batch_x)
			loss = criterion(pred, batch_y)
			total_loss += loss

			pred_labels = np.argmax(pred, axis=1)
			correct += np.sum(pred_labels == batch_y)
			total += len(batch_y)

			pbar.set_postfix({'loss': f'{loss:.8f}', 'acc': f'{np.sum(pred_labels == batch_y) / len(batch_y):.8f}'})

		accuracy = correct / total if total > 0 else 0
		return total_loss / len(dataloader), accuracy
