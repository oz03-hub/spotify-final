#!/usr/bin/env python3
"""
Fine-tuning script for Dense Retriever using contrastive learning.

Training uses InfoNCE loss with in-batch negatives and hard negatives.
For each query, we sample one positive and several hard negatives,
then use cross-entropy where the positive should score highest.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from util import get_query_files, load_queries, load_corpus

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    pass

# ============================================================================
# DATASET
# ============================================================================
class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with hard negatives."""
    
    def __init__(
        self,
        training_data,  # List of dicts with pid, positive_track_ids, negative_track_ids
        playlists,      # List of playlist dicts with pid, name, description
        corpus,         # Dict mapping track_id -> track info
        num_negatives=7,
        max_examples_per_epoch=None,
    ):
        self.training_data = training_data
        self.corpus = corpus
        self.num_negatives = num_negatives
        
        # Create pid -> playlist mapping
        self.pid_to_playlist = {str(p['pid']): p for p in playlists}
        
        # Filter training data to only include playlists we have
        self.training_data = [
            t for t in self.training_data 
            if str(t['pid']) in self.pid_to_playlist
        ]
        
        # Limit examples per epoch if specified
        if max_examples_per_epoch and max_examples_per_epoch < len(self.training_data):
            self.max_examples = max_examples_per_epoch
        else:
            self.max_examples = len(self.training_data)
        
        # Sample indices for this epoch
        self.sample_epoch_indices()
        
        print(f"Dataset initialized with {len(self.training_data)} total examples")
        print(f"Using {self.max_examples} examples per epoch")
    
    def sample_epoch_indices(self):
        """Sample indices to use for this epoch."""
        if self.max_examples < len(self.training_data):
            self.epoch_indices = random.sample(
                range(len(self.training_data)), 
                self.max_examples
            )
        else:
            self.epoch_indices = list(range(len(self.training_data)))
            random.shuffle(self.epoch_indices)
    
    def __len__(self):
        return len(self.epoch_indices)
    
    def __getitem__(self, idx):
        # Get actual training example
        real_idx = self.epoch_indices[idx]
        example = self.training_data[real_idx]
        
        pid = str(example['pid'])
        positive_ids = example['positives']
        negative_ids = example['negatives']
        
        # Get playlist query
        playlist = self.pid_to_playlist[pid]
        query = f"{playlist.get('name', '')} {playlist.get('description', '')}".strip()
        
        # Sample one random positive
        positive_id = random.choice(positive_ids)
        positive_track = self.corpus.get(positive_id, {})
        positive_text = self._get_track_text(positive_track)
        
        # Sample random negatives (up to num_negatives)
        num_negs = min(self.num_negatives, len(negative_ids))
        sampled_negative_ids = random.sample(negative_ids, num_negs)
        
        negative_texts = []
        for neg_id in sampled_negative_ids:
            neg_track = self.corpus.get(neg_id, {})
            negative_texts.append(self._get_track_text(neg_track))
        
        return {
            'query': query,
            'positive': positive_text,
            'negatives': negative_texts,
        }
    
    def _get_track_text(self, track):
        """Create text representation for a track."""
        return f"{track.get('track_name', '')} {track.get('artist_name', '')} {track.get('album_name', '')} {track.get('extended_text', '')}".strip()


def collate_fn(batch):
    """Collate function to handle variable number of negatives."""
    queries = [item['query'] for item in batch]
    positives = [item['positive'] for item in batch]
    
    # Flatten all negatives
    all_negatives = []
    neg_counts = []
    for item in batch:
        all_negatives.extend(item['negatives'])
        neg_counts.append(len(item['negatives']))
    
    return {
        'queries': queries,
        'positives': positives,
        'negatives': all_negatives,
        'neg_counts': neg_counts,
    }


# ============================================================================
# MODEL WRAPPER FOR TRAINING
# ============================================================================
class DenseRetrieverTrainer(nn.Module):
    """Wrapper around SentenceTransformer for training."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.device = None
        
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self
        
    def encode_with_grad(self, texts, normalize=True):
        """Encode texts to embeddings with gradient tracking."""
        # Tokenize
        features = self.model.tokenize(texts)
        
        # Move to device
        features = {k: v.to(self.device) for k, v in features.items()}
        
        # Forward pass through the model (this maintains gradients)
        output = self.model(features)
        embeddings = output['sentence_embedding']
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def forward(self, queries, positives, negatives, neg_counts, temperature=0.05):
        """
        Forward pass with contrastive loss.
        
        Args:
            queries: List of query strings
            positives: List of positive document strings
            negatives: List of all negative document strings
            neg_counts: List of number of negatives per query
            temperature: Temperature for softmax scaling
        
        Returns:
            loss: Contrastive loss
            accuracy: Accuracy of selecting positive
        """
        batch_size = len(queries)
        
        # Encode queries with gradients
        query_embeds = self.encode_with_grad(queries)  # [batch_size, dim]
        
        # Encode positives with gradients
        positive_embeds = self.encode_with_grad(positives)  # [batch_size, dim]
        
        # Encode negatives if any
        if negatives:
            negative_embeds = self.encode_with_grad(negatives)  # [total_negatives, dim]
        else:
            negative_embeds = None
        
        # Compute scores for each query
        losses = []
        correct = 0
        
        neg_offset = 0
        for i in range(batch_size):
            # Query embedding
            q_embed = query_embeds[i:i+1]  # [1, dim]
            
            # Positive score (this query's positive)
            pos_score = torch.matmul(q_embed, positive_embeds[i:i+1].T)  # [1, 1]
            
            # In-batch negatives (other queries' positives)
            in_batch_neg_indices = [j for j in range(batch_size) if j != i]
            if in_batch_neg_indices:
                in_batch_neg_scores = torch.matmul(
                    q_embed, 
                    positive_embeds[in_batch_neg_indices].T
                )  # [1, batch_size-1]
            else:
                in_batch_neg_scores = torch.zeros(1, 0, device=self.device)
            
            # Hard negatives for this query
            num_negs = neg_counts[i]
            if num_negs > 0 and negative_embeds is not None:
                hard_neg_embeds = negative_embeds[neg_offset:neg_offset + num_negs]
                hard_neg_scores = torch.matmul(q_embed, hard_neg_embeds.T)  # [1, num_negs]
                neg_offset += num_negs
            else:
                hard_neg_scores = torch.zeros(1, 0, device=self.device)
            
            # Concatenate all scores: [positive, in_batch_negs, hard_negs]
            all_scores = torch.cat([
                pos_score, 
                in_batch_neg_scores, 
                hard_neg_scores
            ], dim=1) / temperature  # [1, 1 + (batch-1) + num_negs]
            
            # Target is always index 0 (the positive)
            target = torch.zeros(1, dtype=torch.long, device=self.device)
            
            # Cross entropy loss
            loss = F.cross_entropy(all_scores, target)
            losses.append(loss)
            
            # Check if positive has highest score
            if all_scores.argmax().item() == 0:
                correct += 1
        
        total_loss = torch.stack(losses).mean()
        accuracy = correct / batch_size
        
        return total_loss, accuracy
    
    def save_pretrained(self, save_path):
        """Save the fine-tuned model."""
        self.model.save(str(save_path))
        print(f"Model saved to {save_path}")


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train(
    model,
    train_loader,
    optimizer,
    scheduler,
    device,
    epoch,
    temperature=0.05,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        loss, acc = model(
            batch['queries'],
            batch['positives'],
            batch['negatives'],
            batch['neg_counts'],
            temperature=temperature,
        )
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        total_acc += acc
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'acc': f'{total_acc/num_batches:.4f}',
        })
    
    return total_loss / num_batches, total_acc / num_batches


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Dense Retriever")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, default="dataset/dpr_train.jsonl",
                        help="Path to training data JSONL file")
    parser.add_argument("--corpus", type=str, default="dataset/track_corpus.json",
                        help="Path to track corpus JSON file")
    parser.add_argument("--train_queries_dir", type=str, default="dataset/train",
                        help="Directory containing training query files")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                        help="Pretrained model name")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_retriever",
                        help="Output directory for fine-tuned model")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--num_negatives", type=int, default=20,
                        help="Number of hard negatives per query")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Temperature for contrastive loss")
    parser.add_argument("--max_examples_per_epoch", type=int, default=40000,
                        help="Maximum number of training examples per epoch (None for all)")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load corpus
    print(f"Loading corpus from {args.corpus}...")
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} tracks")
    
    # Load training data
    print(f"Loading training data from {args.train_data}...")
    training_data = []
    with open(args.train_data, 'r') as f:
        for line in f:
            if line.strip():
                training_data.append(json.loads(line.strip()))
    print(f"Loaded {len(training_data)} training examples")
    
    # Load playlists
    print(f"Loading playlists from {args.train_queries_dir}...")
    train_query_files = get_query_files(args.train_queries_dir)
    all_training_playlists = []
    for query_file in train_query_files:
        playlists = load_queries(query_file)
        all_training_playlists.extend(playlists)
    print(f"Loaded {len(all_training_playlists)} playlists")
    
    # Create dataset
    dataset = ContrastiveDataset(
        training_data=training_data,
        playlists=all_training_playlists,
        corpus=corpus,
        num_negatives=args.num_negatives,
        max_examples_per_epoch=args.max_examples_per_epoch,
    )
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Shuffling handled in dataset
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    print(f"Loading model: {args.model_name}")
    model = DenseRetrieverTrainer(args.model_name)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    # Scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    print(f"Total training steps: {total_steps}")
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=args.warmup_ratio,
        anneal_strategy='linear',
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training")
    print("="*50)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Resample training examples for this epoch
        dataset.sample_epoch_indices()
        
        # Train
        train_loss, train_acc = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            temperature=args.temperature,
        )
        
        print(f"\nEpoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = Path(args.output_dir) / "best"
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_path)
    
    # Save final model
    final_path = Path(args.output_dir) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    
    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best model saved to: {Path(args.output_dir) / 'best'}")
    print(f"Final model saved to: {final_path}")
    print("="*50)


if __name__ == "__main__":
    main()
