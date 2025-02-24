import torch

# General settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 123

# Model hyperparameters
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
PAD_IDX = 0
NUM_LAYERS = 2

# Training settings
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
NUM_EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
WEIGHT_DECAY = 0

# File paths
DATASET_DIR = "data/combined_emotion.csv"
CHECKPOINT_PATH = "models/emotion_classifier_biderectional.pth.tar"
VOCAB_PATH = "vocab.pkl"

# Emotion labels
INDEX_TO_EMOTION = {0: "fear", 1: "sad", 2: "love", 3: "joy", 4: "surprise", 5: "anger"}
