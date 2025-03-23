import torch
import torch.nn as nn
from .config import config


class TextGenerationModel(nn.Module):
    """Text Generation Model using various RNN architectures."""

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, num_layers, model_type="LSTM"
    ):
        super(TextGenerationModel, self).__init__()
        self.model_type = model_type

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        elif model_type == "BiLSTM":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                bidirectional=True,
            )
        else:
            raise ValueError("Unknown model type")

        if model_type == "BiLSTM":
            self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        else:
            self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        if self.model_type == "LSTM" or self.model_type == "BiLSTM":
            return (
                torch.zeros(config.NUM_LAYERS, batch_size, config.HIDDEN_DIM),
                torch.zeros(config.NUM_LAYERS, batch_size, config.HIDDEN_DIM),
            )
        else:
            return torch.zeros(config.NUM_LAYERS, batch_size, config.HIDDEN_DIM)


def get_model(vocab_size, model_type="LSTM"):
    """Create and return an instance of the TextGenerationModel."""
    return TextGenerationModel(
        vocab_size,
        config.EMBEDDING_DIM,
        config.HIDDEN_DIM,
        config.NUM_LAYERS,
        model_type=model_type,
    )
