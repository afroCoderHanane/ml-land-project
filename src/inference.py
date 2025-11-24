"""
Inference script for predicting land use from satellite images.
"""

import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from model import get_model
from dataset import get_transforms, BINARY_CLASS_NAMES


def predict_image(model, image_path, device):
    """
    Predict land use class for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        device: Device to run inference on
    
    Returns:
        Tuple of (predicted_class, confidence, probabilities)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    # Get transforms (no augmentation for inference)
    transform = get_transforms(is_training=False)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Move to device
    image_tensor = image_tensor.to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
    
    predicted_class = predicted.item()
    confidence_score = confidence.item()
    probs = probabilities.cpu().numpy()[0]
    
    return predicted_class, confidence_score, probs


def visualize_prediction(image_path, predicted_class, confidence, probabilities):
    """
    Visualize the prediction with the original image.
    
    Args:
        image_path: Path to image file
        predicted_class: Predicted class index
        confidence: Confidence score
        probabilities: Probability for each class
    """
    # Load original image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title('Input Satellite Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display prediction
    class_names = ['City', 'Farmland']
    colors = ['#3498db', '#2ecc71']
    
    bars = ax2.barh(class_names, probabilities, color=colors, alpha=0.7)
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Prediction Probabilities', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2,
                f'{prob:.3f}', va='center', fontsize=11)
    
    # Highlight predicted class
    predicted_name = BINARY_CLASS_NAMES[predicted_class]
    ax2.axhline(y=predicted_class, color='red', linestyle='--', linewidth=2, alpha=0.5)
    
    # Overall title
    fig.suptitle(
        f'Prediction: {predicted_name} (Confidence: {confidence:.2%})',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict land use from satellite image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cuda, mps, cpu)')
    parser.add_argument('--no_display', action='store_true',
                       help='Do not display visualization')
    
    args = parser.parse_args()
    
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
    
    # Load model
    print("Loading model...")
    model = get_model(device=device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}\n")
    
    # Predict
    print(f"Predicting land use for: {args.image}")
    predicted_class, confidence, probabilities = predict_image(model, args.image, device)
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"\nPredicted Class: {BINARY_CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence:.2%}")
    print("\nClass Probabilities:")
    print(f"  City:     {probabilities[0]:.4f} ({probabilities[0]*100:.2f}%)")
    print(f"  Farmland: {probabilities[1]:.4f} ({probabilities[1]*100:.2f}%)")
    print("="*60 + "\n")
    
    # Visualize
    if not args.no_display:
        visualize_prediction(args.image, predicted_class, confidence, probabilities)


if __name__ == "__main__":
    main()
