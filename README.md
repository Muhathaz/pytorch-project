# Text Generation with PyTorch

A Text generation model using PyTorch and LSTM/RNN architectures.

## Features

- Multiple RNN architectures support (LSTM, GRU, BiLSTM, RNN)
- Configurable model parameters
- Text generation with custom prompts
- Easy to train on custom datasets

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd text-generation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the example script:
```bash
python run.py
```

## Configuration

Model parameters can be configured in `src/config.py`:

- EMBEDDING_DIM: Dimension of word embeddings
- HIDDEN_DIM: Hidden layer dimension
- NUM_LAYERS: Number of RNN layers
- SEQ_LENGTH: Input sequence length
- BATCH_SIZE: Training batch size
- EPOCHS: Number of training epochs
- LEARNING_RATE: Model learning rate

## Project Structure

```
text-generation/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── trainer.py
│   └── inference.py
├── requirements.txt
├── setup.py
└── run.py
```

## License

MIT License 