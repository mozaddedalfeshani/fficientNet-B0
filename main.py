import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet101, ResNet101_Weights
import timm
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy
from timm.scheduler import CosineLRScheduler
import torchmetrics
from PIL import Image
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class TeaLeafDataset(Dataset):
    """Dataset class for tea leaf images organized in folders by class."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Get all class folders
        class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Create class mappings
        for idx, class_folder in enumerate(class_folders):
            class_name = class_folder.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Load all images from this class folder
            image_files = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(idx)
        
        print(f"Loaded {len(self.images)} images from {len(class_folders)} classes")
        print(f"Classes: {list(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(data_dir, batch_size=128, num_workers=4, img_size=224):
    """Create train and validation data loaders with augmentation."""
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Training transforms - Simplified to prevent memory issues
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = TeaLeafDataset(data_dir, transform=None)
    
    # Split into train and validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    # Optimized DataLoader settings for MPS
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable for MPS
        persistent_workers=False if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Disable for MPS
        persistent_workers=False if num_workers > 0 else False
    )
    
    return train_loader, val_loader, full_dataset.class_to_idx


def apply_mixup_cutmix(x, y, mixup_fn):
    """Apply MixUp or CutMix augmentation using timm's Mixup class."""
    if mixup_fn is not None:
        # Ensure labels are long type for Mixup
        y = y.long() if y.dtype != torch.long else y
        x, y = mixup_fn(x, y)
    return x, y


def train_epoch(model, train_loader, criterion, optimizer, device, scaler, mixup_fn, epoch, label_smoothing=0.1):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Apply MixUp/CutMix
        images, labels = apply_mixup_cutmix(images, labels, mixup_fn)
        
        optimizer.zero_grad()
        
        # Forward pass - handle MPS differently
        # timm's Mixup with mode='batch' returns (y_a, y_b, lam) tuple
        outputs = model(images)
        
        # Handle different label formats from Mixup
        if isinstance(labels, tuple) and len(labels) == 3:
            # MixUp/CutMix: (y_a, y_b, lam)
            y_a, y_b, lam = labels
            # Ensure labels are long type (class indices)
            y_a = y_a.long() if y_a.dtype != torch.long else y_a
            y_b = y_b.long() if y_b.dtype != torch.long else y_b
            # Calculate loss
            loss_a = criterion(outputs, y_a)
            loss_b = criterion(outputs, y_b)
            loss = lam * loss_a + (1 - lam) * loss_b
        elif labels.dim() == 2 and labels.size(1) > 1:
            # One-hot encoded labels (shouldn't happen with mode='batch', but handle it)
            # Convert to class indices
            labels_long = torch.argmax(labels, dim=1).long()
            loss = criterion(outputs, labels_long)
        else:
            # Regular class index labels
            labels_long = labels.long() if labels.dtype != torch.long else labels
            loss = criterion(outputs, labels_long)
        
        # Add label smoothing effect
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
        smooth_loss = -log_probs.mean(dim=1).mean()
        loss = (1 - label_smoothing) * loss + label_smoothing * smooth_loss
        
        # Backward pass
        if device.type == 'cuda' and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Clear cache periodically for MPS to prevent hanging
        if device.type == 'mps' and total % 100 == 0:
            torch.mps.empty_cache()
        
        running_loss += loss.item()
        
        # Calculate approximate accuracy (for display during training)
        # Note: With MixUp/CutMix, accuracy is approximate
        _, predicted = torch.max(outputs.data, 1)
        if isinstance(labels, tuple) and len(labels) == 3:
            # MixUp/CutMix returns (y_a, y_b, lam) - use y_a for approximate accuracy
            y_a = labels[0]
            # Ensure y_a is 1D (class indices)
            if y_a.dim() > 1:
                target_labels = torch.argmax(y_a, dim=1).long() if y_a.size(1) > 1 else y_a.squeeze().long()
            else:
                target_labels = y_a.long()
        elif labels.dim() == 2 and labels.size(1) > 1:
            # One-hot encoded
            target_labels = torch.argmax(labels, dim=1).long()
        else:
            # Regular class indices
            target_labels = labels.long() if labels.dtype != torch.long else labels
            if target_labels.dim() > 1:
                target_labels = target_labels.squeeze()
        
        total += target_labels.size(0)
        correct += (predicted == target_labels).sum().item()
        acc = 100 * correct / total if total > 0 else 0.0
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.2f}%'})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    return epoch_loss, accuracy


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, save_path='training_curves.png'):
    """Plot training and validation curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'})
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()


def main():
    # Configuration - Optimized for MPS GPU
    DATA_DIR = 'output_augment'
    BATCH_SIZE = 32  # Reduced to prevent memory issues
    IMG_SIZE = 224
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1
    EARLY_STOPPING_PATIENCE = 20
    NUM_WORKERS = 0  # Set to 0 for MPS to prevent hanging
    
    # Device setup - Apple Silicon GPU (MPS) with optimizations
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Using Apple Silicon GPU (MPS)")
        # Set MPS memory management to prevent hanging
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        # Clear MPS cache
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ Using CUDA GPU")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU available, using CPU")
    
    print(f"Device: {device}\n")
    
    # Load data
    print("Loading dataset...")
    train_loader, val_loader, class_to_idx = get_data_loaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE
    )
    
    num_classes = len(class_to_idx)
    print(f"Number of classes: {num_classes}\n")
    
    # Initialize model - ResNet-101
    print("Initializing ResNet-101 model...")
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    
    # Modify classifier for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")
    
    # Loss function with label smoothing
    # Use standard CrossEntropyLoss - we'll handle label smoothing in Mixup
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler with cosine annealing and warmup
    num_training_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_training_steps - warmup_steps,
        lr_min=1e-6,
        warmup_t=warmup_steps,
        warmup_lr_init=1e-6,
        warmup_prefix=True
    )
    
    # Mixed precision scaler (only for CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # MixUp and CutMix - Reduced probability to prevent memory issues
    # Use mode='batch' which returns tuple format we can handle
    mixup_fn = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=0.5,  # Reduced from 1.0 to 0.5 to save memory
        switch_prob=0.5,
        mode='batch',
        num_classes=num_classes,
        label_smoothing=0.0  # We'll handle smoothing manually
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print("Starting training...\n")
    print("=" * 80)
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            scaler, mixup_fn, epoch, LABEL_SMOOTHING
        )
        
        # Update learning rate
        scheduler.step(epoch * len(train_loader))
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': class_to_idx
            }, 'best_model.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        
        print("=" * 80)
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 80)
    
    # Load best model and do final evaluation
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_loss, final_acc = validate(model, val_loader, criterion, device)
    print(f"\nFinal validation accuracy: {final_acc:.2f}%")
    
    # Detailed classification report
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Plot training curves
    print("\nGenerating training curves...")
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    # Plot confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(all_labels, all_preds, class_names)
    
    print("\n✅ All plots generated successfully!")


if __name__ == '__main__':
    main()

