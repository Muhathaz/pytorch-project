import torch
from .config import config
from .data import tokenize_text


def generate_text(model, start_text, length, word2idx, idx2word):
    """Generate new text using the trained model."""
    model.eval()
    tokens = tokenize_text(start_text)

    input_seq = torch.tensor(
        [word2idx.get(word, word2idx[config.UNK_TOKEN]) for word in tokens],
        dtype=torch.long,
    ).unsqueeze(0)

    hidden = model.init_hidden(1)

    generated_text = start_text
    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        next_word_idx = output.argmax(dim=2)[:, -1].item()
        next_word = idx2word[next_word_idx]

        generated_text += " " + next_word
        input_seq = torch.cat([input_seq, torch.tensor([[next_word_idx]])], dim=1)[
            :, -config.SEQ_LENGTH :
        ]

    return generated_text


def run_inference(model, start_text, length, word2idx, idx2word):
    """Run inference using the trained model."""
    generated_story = generate_text(model, start_text, length, word2idx, idx2word)
    print("Generated Story:\n", generated_story)
