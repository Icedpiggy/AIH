import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, val_losses, checkpoint_dir):
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
	plt.savefig(f'{checkpoint_dir}/loss_curve.png', dpi=300, bbox_inches='tight')
	print(f"Loss curve saved to {checkpoint_dir}/loss_curve.png")
	plt.show()


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, checkpoint_dir):
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
	plt.savefig(f'{checkpoint_dir}/training_curves.png', dpi=300, bbox_inches='tight')
	print(f"Training curves saved to {checkpoint_dir}/training_curves.png")
	plt.show()
