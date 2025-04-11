# train.py (UPDATED)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from ViT_arch import VisionTransformer
from dataset import get_cifar10
from datetime import datetime

def log_to_file(filename, text):
    with open(filename, "a") as f:
        f.write(f"{text}\n")

def train(model, train_loader, val_loader, device, epochs=50, lr=3e-4, patience=7):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    no_improve_epochs = 0

    os.makedirs('model_ckpt', exist_ok=True)
    log_to_file("train_log.txt", f"Start training at {datetime.now()}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_acc, val_loss = evaluate(model, val_loader, device, criterion)
        log_line = f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        print(log_line)
        log_to_file("eval_log.txt", log_line)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'model_ckpt/vit_best.pth')
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                break

def evaluate(model, data_loader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    avg_loss = total_loss / len(data_loader) if criterion else 0
    return acc, avg_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_cifar10(batch_size=128, augment=True)

    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        embed_dim=256,
        mlp_dim=512,
        num_layers=	12,
        num_classes=10,
        num_heads= 8,
        dropout=0.1,
        in_channels=3
    ).to(device)

    train(model, train_loader, val_loader, device, epochs=50, lr=3e-4, patience=5)
