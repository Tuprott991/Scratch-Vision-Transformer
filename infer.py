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

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = [str(i) for i in range(10)]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on CIFAR-10 Test Set')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_and_plot("model_ckpt/vit_best.pth", device)
