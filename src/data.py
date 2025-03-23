import torch
from torch.utils.data import Dataset, DataLoader
from .config import config


def tokenize_text(text):
    """Tokenize the input text into words."""
    return text.lower().split()


class StoryDataset(Dataset):
    """Dataset class for story text data."""

    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx : idx + self.seq_length]),
            torch.tensor(self.data[idx + 1 : idx + self.seq_length + 1]),
        )


def prepare_dataset(story):
    """Prepare the dataset for training."""
    tokenized_story = tokenize_text(story)
    vocab = sorted(set(tokenized_story))
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for i, word in enumerate(vocab)}

    if config.UNK_TOKEN not in vocab:
        vocab.append(config.UNK_TOKEN)
        word2idx[config.UNK_TOKEN] = len(word2idx)
        idx2word[len(idx2word)] = config.UNK_TOKEN

    data = [word2idx[word] for word in tokenized_story]
    dataset = StoryDataset(data, config.SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    return dataloader, vocab, word2idx, idx2word
