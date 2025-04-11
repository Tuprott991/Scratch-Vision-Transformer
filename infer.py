# infer.py

import torch
from torch.utils.data import DataLoader
from ViT_arch import VisionTransformer
from dataset import get_cifar10
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def infer_and_plot(model_path, device):
    _, test_loader = get_cifar10(batch_size=128)

    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        embed_dim=128,
        mlp_dim=256,
        num_layers=6,
        num_classes=10,
        num_heads=8,
        dropout=0.1,
        in_channels=3
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs, _ = model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    plot_confusion_matrix(all_labels, all_preds)

# Calculate Top-1 Accuracy
def calculate_top1_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum()
    total = len(y_true)
    accuracy = correct / total * 100
    return accuracy

# Plot Confusion Matrix including Top-1 Accuracy
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = calculate_top1_accuracy(np.array(y_true), np.array(y_pred))

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Top-1 Accuracy: {accuracy:.2f}%)')
    plt.show()
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_and_plot("model_ckpt/vit_best_1.pth", device)
