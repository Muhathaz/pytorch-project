from .data import prepare_dataset
from .model import get_model
from .trainer import train_model
from .inference import run_inference
from .config import config
import torch


def train_and_save_model(dataloader, vocab_size, model_type):
    """Train and save the model."""
    model = get_model(vocab_size, model_type)
    train_model(model, dataloader, config.EPOCHS, config.LEARNING_RATE, vocab_size)

    torch.save(model.state_dict(), f"text_generation_{model_type}.pth")
    print(f"Model saved to text_generation_{model_type}.pth")

    return model


def load_model(vocab_size, model_type):
    """Load a previously trained model."""
    model = get_model(vocab_size, model_type)
    try:
        model.load_state_dict(torch.load(f"text_generation_{model_type}.pth"))
        print(f"Model loaded from text_generation_{model_type}.pth")
    except FileNotFoundError:
        print(
            f"Model file text_generation_{model_type}.pth not found. Please train the model first."
        )
        return None
    return model


if __name__ == "__main__":
    # Example story
    story = """
    Once upon a time, in a land far away, there was a peaceful village surrounded by mountains. 
    The villagers lived in harmony with nature. They grew crops, raised animals, and lived a simple but happy life.
    One day, a young girl named Lily discovered a mysterious cave hidden in the forest. She was curious and decided to explore.
    Inside the cave, she found glowing crystals and strange markings on the walls.
    As she ventured deeper, she realized she was not alone.
    """

    # Prepare dataset
    dataloader, vocab, word2idx, idx2word = prepare_dataset(story)

    # Set model type and train
    model_type = "LSTM"
    model = train_and_save_model(dataloader, len(vocab), model_type)

    # Run inference
    start_text = "once upon a time"
    run_inference(model, start_text, length=50, word2idx=word2idx, idx2word=idx2word)
