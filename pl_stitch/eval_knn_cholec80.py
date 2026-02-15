#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
k-NN Evaluation for Cholec80 Phase Recognition
Following the paper's methodology:
- Extract features from frozen backbone
- Use k=20 nearest neighbors
- Weighted voting based on cosine similarity
"""

import argparse
import os
import glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import sys

# Add pl_stitch to path
sys.path.append('./pl_stitch')
import models.vision_transformer as vits


# Class mapping for Cholec80
CLASS_MAPPING = {
    'Preparation': 0,
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderRetraction': 4,
    'GallbladderPackaging': 5,
    'CleaningCoagulation': 6
}

CLASS_NAMES = [
    'Preparation',
    'CalotTriangleDissection', 
    'ClippingCutting',
    'GallbladderDissection',
    'GallbladderRetraction',
    'GallbladderPackaging',
    'CleaningCoagulation'
]


def load_model(arch, patch_size, checkpoint_path, device):
    """Load pretrained model with frozen weights"""
    print(f"Loading model: {arch}, patch_size={patch_size}")
    
    # Build model
    if arch == 'vit_base':
        model = vits.vit_base(patch_size=patch_size, return_all_tokens=False)
    elif arch == 'vit_small':
        model = vits.vit_small(patch_size=patch_size, return_all_tokens=False)
    elif arch == 'vit_large':
        model = vits.vit_large(patch_size=patch_size, return_all_tokens=False)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    # Load checkpoint
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        elif 'student' in state_dict:
            state_dict = state_dict['student']
        
        # Remove prefixes
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint with msg: {msg}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def load_labels_from_txt(txt_path):
    """Load phase annotations from txt file"""
    labels = []
    if not os.path.exists(txt_path):
        print(f"Warning: File not found {txt_path}")
        return labels

    with open(txt_path, 'r') as f:
        # 첫 줄 헤더(Frame\tPhase) 건너뛰기
        lines = f.readlines()[1:]
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 탭(\t) 또는 공백으로 분리
            parts = line.split() 
            if len(parts) >= 2:
                phase_name = parts[1].strip()
                # CLASS_MAPPING에 정의된 이름을 인덱스로 변환
                label = CLASS_MAPPING.get(phase_name, -1)
                labels.append(label)
    return labels


def load_video_frames_and_labels(video_dir, label_dir, video_ids):
    """
    Load frames and labels for specified videos
    
    Returns:
        frame_paths: list of (video_id, frame_path)
        labels: list of class indices
    """
    frame_paths = []
    labels = []
    
    for video_id in tqdm(video_ids, desc="Loading video metadata"):
        # Load frames
        video_folder = os.path.join(video_dir, f"{video_id:02d}")
        if not os.path.exists(video_folder):
            print(f"Warning: Video folder not found: {video_folder}")
            continue
        
        frame_files = sorted(glob.glob(os.path.join(video_folder, "*.jpg")))
        
        # Load labels
        label_file = os.path.join(label_dir, f"video{video_id:02d}-phase.txt")
        if not os.path.exists(label_file):
            print(f"Warning: Label file not found: {label_file}")
            continue
        
        video_labels = load_labels_from_txt(label_file)
        
        # Match frames with labels (should be same length)
        if len(frame_files) != len(video_labels):
            print(f"Warning: Frame count ({len(frame_files)}) != Label count ({len(video_labels)}) for video {video_id}")
            min_len = min(len(frame_files), len(video_labels))
            frame_files = frame_files[:min_len]
            video_labels = video_labels[:min_len]
        
        # Add to lists
        for frame_path, label in zip(frame_files, video_labels):
            frame_paths.append((video_id, frame_path))
            labels.append(label)
    
    return frame_paths, labels


def extract_features(model, frame_paths, device, batch_size=32):
    """
    Extract features for all frames using frozen backbone
    
    Returns:
        features: numpy array of shape (N, D)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    features_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(frame_paths), batch_size), desc="Extracting features"):
            batch_paths = frame_paths[i:i+batch_size]
            
            # Load and transform images
            images = []
            for _, frame_path in batch_paths:
                img = Image.open(frame_path).convert('RGB')
                img_tensor = transform(img)
                images.append(img_tensor)
            
            # Stack and move to device
            images = torch.stack(images).to(device)
            
            # Extract features (CLS token)
            feats = model(images)  # [B, D]
            
            features_list.append(feats.cpu().numpy())
    
    features = np.concatenate(features_list, axis=0)
    print(f"Extracted features shape: {features.shape}")
    
    return features


def cosine_similarity_matrix(query_features, train_features):
    """
    Compute cosine similarity between query and train features
    
    Args:
        query_features: (N_test, D)
        train_features: (N_train, D)
    
    Returns:
        similarity: (N_test, N_train)
    """
    # Normalize features
    query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    train_norm = train_features / (np.linalg.norm(train_features, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(query_norm, train_norm.T)
    
    return similarity


def knn_classify(test_features, train_features, train_labels, k=20):
    """
    k-NN classification with weighted voting based on cosine similarity
    
    Args:
        test_features: (N_test, D)
        train_features: (N_train, D)
        train_labels: (N_train,)
        k: number of neighbors
    
    Returns:
        predictions: (N_test,)
    """
    print(f"Computing k-NN predictions with k={k}...")
    
    # Compute similarity matrix
    similarity = cosine_similarity_matrix(test_features, train_features)
    
    predictions = []
    
    for i in tqdm(range(len(test_features)), desc="k-NN classification"):
        # Get k nearest neighbors
        top_k_indices = np.argsort(similarity[i])[-k:][::-1]
        top_k_similarities = similarity[i][top_k_indices]
        top_k_labels = train_labels[top_k_indices]
        
        # Weighted voting
        num_classes = len(CLASS_NAMES)
        votes = np.zeros(num_classes)
        
        for sim, label in zip(top_k_similarities, top_k_labels):
            if 0 <= label < num_classes:
                votes[label] += sim
        
        # Predict class with highest weighted vote
        pred = np.argmax(votes)
        predictions.append(pred)
    
    return np.array(predictions)


def compute_metrics(predictions, labels):
    """Compute accuracy and F1-score"""
    # Filter out invalid labels (-1)
    valid_mask = labels >= 0
    predictions = predictions[valid_mask]
    labels = labels[valid_mask]
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    return accuracy, f1


def main():
    parser = argparse.ArgumentParser(description='k-NN Evaluation for Cholec80')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--arch', type=str, default='vit_base',
                       choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large'],
                       help='Architecture')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size')
    parser.add_argument('--train_dir', type=str,
                       default='/workspace/code/Dataset/cholec80/frames/extract_1fps/training_set',
                       help='Training frames directory')
    parser.add_argument('--test_dir', type=str,
                       default='/workspace/code/Dataset/cholec80/frames/extract_1fps/test_set',
                       help='Test frames directory')
    parser.add_argument('--train_label_dir', type=str,
                       default='/workspace/code/Dataset/cholec80/phase_annotations/training_set_1fps',
                       help='Training labels directory')
    parser.add_argument('--test_label_dir', type=str,
                       default='/workspace/code/Dataset/cholec80/phase_annotations/test_set_1fps',
                       help='Test labels directory')
    parser.add_argument('--k', type=int, default=20,
                       help='Number of nearest neighbors')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for feature extraction')
    parser.add_argument('--output_dir', type=str, default='./knn_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.arch, args.patch_size, args.checkpoint, device)
    
    # Load training data
    print("\n" + "="*60)
    print("LOADING TRAINING DATA")
    print("="*60)
    train_video_ids = list(range(1, 41))  # Videos 01-40
    train_frame_paths, train_labels = load_video_frames_and_labels(
        args.train_dir, args.train_label_dir, train_video_ids
    )
    train_labels = np.array(train_labels)
    print(f"Training: {len(train_frame_paths)} frames loaded")
    
    # Load test data
    print("\n" + "="*60)
    print("LOADING TEST DATA")
    print("="*60)
    test_video_ids = list(range(41, 81))  # Videos 41-80
    test_frame_paths, test_labels = load_video_frames_and_labels(
        args.test_dir, args.test_label_dir, test_video_ids
    )
    test_labels = np.array(test_labels)
    print(f"Test: {len(test_frame_paths)} frames loaded")
    
    # Extract features
    print("\n" + "="*60)
    print("EXTRACTING TRAINING FEATURES")
    print("="*60)
    train_features = extract_features(model, train_frame_paths, device, args.batch_size)
    
    print("\n" + "="*60)
    print("EXTRACTING TEST FEATURES")
    print("="*60)
    test_features = extract_features(model, test_frame_paths, device, args.batch_size)
    
    # k-NN classification
    print("\n" + "="*60)
    print(f"k-NN CLASSIFICATION (k={args.k})")
    print("="*60)
    predictions = knn_classify(test_features, train_features, train_labels, k=args.k)
    
    # Compute metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    accuracy, f1 = compute_metrics(predictions, test_labels)
    
    print(f"k-NN Accuracy: {accuracy*100:.2f}%")
    print(f"Macro F1-Score: {f1*100:.2f}%")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'knn_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Architecture: {args.arch}\n")
        f.write(f"k: {args.k}\n")
        f.write(f"\nResults:\n")
        f.write(f"k-NN Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Macro F1-Score: {f1*100:.2f}%\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Save predictions for analysis
    np.savez(
        os.path.join(args.output_dir, 'predictions.npz'),
        predictions=predictions,
        labels=test_labels,
        test_frame_paths=[fp[1] for fp in test_frame_paths]
    )
    print(f"Predictions saved to: {os.path.join(args.output_dir, 'predictions.npz')}")


if __name__ == '__main__':
    main()