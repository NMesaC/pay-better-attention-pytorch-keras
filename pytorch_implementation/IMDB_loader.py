import re
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import Dict, List, Tuple

class IMDBDataset(Dataset):
    def __init__(self, csv_path: str, vocab_size: int = 10000, max_length: int = 200):
        self.data = pd.read_csv(csv_path)
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        self.preprocess_data()
        self.build_vocabulary()
        self.tokenize_reviews()

    def preprocess_data(self):
        self.data['review'] = self.data['review'].apply(self.clean_text)
        self.data['sentiment'] = self.data['sentiment'].map({"positive": 1, "negative": 0})

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"<br\s*/?>", " ", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.strip()

    def build_vocabulary(self):
        word_freq = Counter()
        for review in self.data['review']:
            word_freq.update(review.split())

        special_tokens = ['<PAD>', '<SOS>']
        common_words = [word for word, _ in word_freq.most_common(self.vocab_size - len(special_tokens))]
        self.vocab = {word: idx for idx, word in enumerate(special_tokens + common_words)}

    def tokenize_reviews(self):
        self.tokenized_reviews = []
        for review in self.data['review']:
            tokens = [self.vocab['<SOS>']]
            tokens.extend([self.vocab.get(word, self.vocab['<PAD>']) for word in review.split()[:self.max_length-1]])
            self.tokenized_reviews.append(torch.tensor(tokens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.tokenized_reviews[idx], self.data['sentiment'].iloc[idx]

def collate_imdb(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reviews, sentiments = zip(*batch)
    padded_reviews = pad_sequence(reviews, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(review) for review in reviews])
    sentiments = torch.tensor(sentiments, dtype=torch.float32)
    return padded_reviews, lengths, sentiments

def get_dataloader(csv_path: str, vocab_size: int, max_length: int, batch_size: int, val_split : float) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    dataset = IMDBDataset(csv_path, vocab_size, max_length)
    train_size = 40000
    val_size   = (int)(train_size * val_split)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_imdb)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_imdb)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=collate_imdb)
    return train_loader, val_loader, test_loader
