import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data import prepare_dataset
from src.model import get_model
from src.trainer import train_model
from src.inference import run_inference
from src.config import config

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
    model = get_model(vocab_size=len(vocab), model_type=model_type)
    train_model(model, dataloader, config.EPOCHS, config.LEARNING_RATE, len(vocab))

    # Run inference
    start_text = "once upon a time"
    run_inference(model, start_text, length=50, word2idx=word2idx, idx2word=idx2word)
