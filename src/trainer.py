import torch
import torch.nn as nn
import torch.optim as optim
from .config import config


def train_model(model, dataloader, epochs, lr, vocab_size):
    """Train the text generation model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)

            hidden = (
                tuple(h.detach() for h in hidden)
                if isinstance(hidden, tuple)
                else hidden.detach()
            )

            optimizer.zero_grad()

            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}"
                )

    print("Training complete.")
