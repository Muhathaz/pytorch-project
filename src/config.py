class Config:
    """Configuration class for text generation model."""

    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    SEQ_LENGTH = 10
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 0.001
    UNK_TOKEN = "<UNK>"


config = Config()
