"""
Vigil AI - Class-Balanced Sampling for Imbalanced Dataset
==========================================================
Addresses class imbalance by oversampling minority classes during training.
"""

import numpy as np
from collections import Counter
from typing import List, Tuple
import random

class ClassBalancedSampler:
    """Handles class imbalance through strategic oversampling."""
    
    def __init__(self, target_ratio: float = 1.5):
        """
        Args:
            target_ratio: Maximum ratio between majority and minority class samples
                         (1.5 means no class should have more than 1.5x samples of any other)
        """
        self.target_ratio = target_ratio
    
    def balance_dataset(self, dataset: List[Tuple[str, str]], 
                       seed: int = 42) -> List[Tuple[str, str]]:
        """
        Balance dataset by oversampling minority classes.
        
        Args:
            dataset: List of (text, label) tuples
            seed: Random seed for reproducibility
            
        Returns:
            Balanced dataset with oversampled minority classes
        """
        random.seed(seed)
        
        # Count samples per class
        label_counts = Counter([label for _, label in dataset])
        print(f"Original class distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label:22s}: {count:5d}")
        
        # Find target count (based on majority class)
        max_count = max(label_counts.values())
        target_count = int(max_count / self.target_ratio)
        
        balanced_dataset = []
        
        # Process each class
        for label in label_counts:
            class_samples = [(text, lbl) for text, lbl in dataset if lbl == label]
            current_count = len(class_samples)
            
            if current_count >= target_count:
                # Majority class - sample down to target
                sampled = random.sample(class_samples, target_count)
                balanced_dataset.extend(sampled)
                print(f"  {label:22s}: {current_count:5d} -> {target_count:5d} (undersampled)")
            else:
                # Minority class - oversample to target
                multiplier = target_count // current_count
                remainder = target_count % current_count
                
                # Add full copies
                balanced_dataset.extend(class_samples * multiplier)
                
                # Add random remainder
                if remainder > 0:
                    balanced_dataset.extend(random.sample(class_samples, remainder))
                
                print(f"  {label:22s}: {current_count:5d} -> {target_count:5d} (oversampled {multiplier}x)")
        
        # Shuffle final dataset
        random.shuffle(balanced_dataset)
        
        # Verify final distribution
        final_counts = Counter([label for _, label in balanced_dataset])
        print(f"\nBalanced class distribution:")
        for label, count in sorted(final_counts.items()):
            print(f"  {label:22s}: {count:5d}")
        
        print(f"\nDataset size: {len(dataset)} -> {len(balanced_dataset)} ({len(balanced_dataset)/len(dataset):.1f}x)")
        
        return balanced_dataset
    
    def get_class_weights(self, dataset: List[Tuple[str, str]]) -> dict:
        """
        Calculate class weights for loss function balancing.
        
        Args:
            dataset: List of (text, label) tuples
            
        Returns:
            Dictionary mapping class names to weights
        """
        label_counts = Counter([label for _, label in dataset])
        total_samples = len(dataset)
        n_classes = len(label_counts)
        
        # Inverse frequency weighting
        class_weights = {}
        for label, count in label_counts.items():
            weight = total_samples / (n_classes * count)
            class_weights[label] = weight
        
        print("Class weights (inverse frequency):")
        for label, weight in sorted(class_weights.items()):
            print(f"  {label:22s}: {weight:.3f}")
        
        return class_weights

def create_balanced_training_set(dataset: List[Tuple[str, str]], 
                               test_size: float = 0.15,
                               seed: int = 42) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], dict]:
    """
    Create balanced training set with proper train/test split.
    
    Args:
        dataset: Original dataset
        test_size: Fraction for test set
        seed: Random seed
        
    Returns:
        Tuple of (balanced_train_set, test_set, class_weights)
    """
    from sklearn.model_selection import train_test_split
    
    # First split original dataset to avoid data leakage
    texts = [text for text, _ in dataset]
    labels = [label for _, label in dataset]
    
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    
    # Reconstruct training set
    train_set = list(zip(X_train_raw, y_train_raw))
    test_set = list(zip(X_test, y_test))
    
    # Balance only the training set
    sampler = ClassBalancedSampler(target_ratio=1.5)
    balanced_train = sampler.balance_dataset(train_set, seed=seed)
    
    # Calculate class weights from balanced training set
    class_weights = sampler.get_class_weights(balanced_train)
    
    print(f"\nFinal split:")
    print(f"  Train (balanced): {len(balanced_train)} samples")
    print(f"  Test (original):  {len(test_set)} samples")
    
    return balanced_train, test_set, class_weights

if __name__ == "__main__":
    # Example usage
    from train_ml_model import DATASET
    
    print("Testing class-balanced sampling...")
    balanced_train, test_set, class_weights = create_balanced_training_set(DATASET)
    
    print(f"\nClass weights for training:")
    for cls, weight in sorted(class_weights.items()):
        print(f"  {cls}: {weight:.3f}")
