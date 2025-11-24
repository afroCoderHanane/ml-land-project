"""
Evaluation script for land use classification model.
Computes metrics and generates visualizations.
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from tqdm import tqdm
import pandas as pd

from dataset import get_dataloaders, BINARY_CLASS_NAMES
from model import get_model


def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Tuple of (predictions, true_labels, probabilities)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary of metrics
    """
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    
    # Weighted averages
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'city': {
            'precision': float(precision[0]),
            'recall': float(recall[0]),
            'f1_score': float(f1[0]),
            'support': int(support[0])
        },
        'farmland': {
            'precision': float(precision[1]),
            'recall': float(recall[1]),
            'f1_score': float(f1[1]),
            'support': int(support[1])
        },
        'weighted_avg': {
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1_score': float(f1_weighted)
        }
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['City', 'Farmland'],
        yticklabels=['City', 'Farmland'],
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {save_path}")


def plot_training_curves(history_path, save_path):
    """Plot training and validation curves."""
    try:
        # Load training history
        df = pd.read_csv(history_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss', linewidth=2)
        ax1.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(df['epoch'], df['train_acc'], 'o-', label='Train Accuracy', linewidth=2)
        ax2.plot(df['epoch'], df['val_acc'], 's-', label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved: {save_path}")
    except FileNotFoundError:
        print(f"Warning: Training history file not found: {history_path}")
        print("Skipping training curves plot.")


def plot_sample_predictions(model, test_loader, device, save_path, num_samples=16):
    """Plot sample predictions with images."""
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(test_loader))
    images_display = images[:num_samples]
    labels_true = labels[:num_samples]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images_display.to(device))
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
    
    # Denormalize images for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Create subplot grid
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    
    for idx in range(min(num_samples, rows * cols)):
        ax = axes[idx // cols, idx % cols]
        
        # Denormalize image
        img = images_display[idx].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Get prediction info
        pred_label = preds[idx].item()
        true_label = labels_true[idx].item()
        confidence = probs[idx][pred_label].item()
        
        # Display image
        ax.imshow(img)
        
        # Title with prediction
        color = 'green' if pred_label == true_label else 'red'
        title = f"True: {BINARY_CLASS_NAMES[true_label]}\n"
        title += f"Pred: {BINARY_CLASS_NAMES[pred_label]} ({confidence:.2f})"
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sample predictions saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate land use classification model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, default='./data',
                       help='Root directory for data')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, mps, cpu)')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}\n")
    
    # Load data
    print("="*60)
    print("Loading test data...")
    print("="*60)
    _, _, test_loader = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load model
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    model = get_model(device=device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%\n")
    
    # Evaluate
    print("="*60)
    predictions, true_labels, probabilities = evaluate_model(model, test_loader, device)
    
    # Compute metrics
    print("\n" + "="*60)
    print("Computing metrics...")
    print("="*60)
    metrics = compute_metrics(true_labels, predictions)
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%\n")
    
    print("Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for class_name in ['city', 'farmland']:
        class_metrics = metrics[class_name]
        print(f"{class_name.capitalize():<15} "
              f"{class_metrics['precision']:.4f}      "
              f"{class_metrics['recall']:.4f}      "
              f"{class_metrics['f1_score']:.4f}      "
              f"{class_metrics['support']}")
    
    print("-" * 60)
    print(f"{'Weighted Avg':<15} "
          f"{metrics['weighted_avg']['precision']:.4f}      "
          f"{metrics['weighted_avg']['recall']:.4f}      "
          f"{metrics['weighted_avg']['f1_score']:.4f}")
    print("-" * 60)
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Generate visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # Confusion matrix
    cm_path = os.path.join(args.results_dir, 'confusion_matrix.png')
    plot_confusion_matrix(true_labels, predictions, cm_path)
    
    # Training curves (find most recent history file)
    import glob
    history_files = glob.glob('./results/training_history_*.csv')
    if history_files:
        latest_history = max(history_files, key=os.path.getctime)
        curves_path = os.path.join(args.results_dir, 'training_curves.png')
        plot_training_curves(latest_history, curves_path)
    
    # Sample predictions
    samples_path = os.path.join(args.results_dir, 'sample_predictions.png')
    plot_sample_predictions(model, test_loader, device, samples_path)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)
    print(f"\nResults saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
