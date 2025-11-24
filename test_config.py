"""Quick test script to verify dataset loading without downloading full dataset"""
import sys
sys.path.append('src')

from dataset import EUROSAT_CLASSES, BINARY_CLASS_NAMES, CLASS_TO_BINARY, CITY_CLASSES, FARMLAND_CLASSES

print("Testing dataset configuration...")
print("\nEuroSAT Classes:", EUROSAT_CLASSES)
print("\nBinary Mapping:")
print(f"  City classes: {CITY_CLASSES}")
print(f"  Farmland classes: {FARMLAND_CLASSES}")
print(f"\nClass mapping dictionary: {CLASS_TO_BINARY}")
print(f"\nBinary class names: {BINARY_CLASS_NAMES}")

print("\n✅ Dataset configuration test passed!")
