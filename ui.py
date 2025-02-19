import pickle

import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import torch

from model import EmotionClassifier
from utils import load_checkpoint
from vocabulary import Vocabulary

st.set_page_config(page_title="Emotion Classification", page_icon=":smiley:")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "emotion_classifier.pth.tar"
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
PAD_IDX = 0
NUM_LAYERS = 2

index_to_emotion = {0: "fear", 1: "sad", 2: "love", 3: "joy", 4: "suprise", 5: "anger"}


@st.cache_resource
def load_model_and_vocab():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    model = EmotionClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=6,
        pad_idx=PAD_IDX,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    load_checkpoint(torch.load(CHECKPOINT_PATH), model)

    model.eval()

    return model, vocab


model, vocab = load_model_and_vocab()

st.title("Emotion Classification")
user_input = st.text_input("Enter text for analysis")

if user_input:
    tokens = vocab.encode(user_input)
    inputs = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(inputs, [len(tokens)])
        prediction = torch.argmax(output, dim=1).item()
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    emotion = index_to_emotion.get(prediction, "Unknown")
    st.write(f"Predicted emotion: **{emotion}**")

    emotions = list(index_to_emotion.values())
    fig = px.bar(
        x=emotions,
        y=probabilities,
        labels={"x": "Emotion", "y": "Probability"},
        title="Emotion Prediction Probabilities",
        color=emotions,
    )

    st.plotly_chart(fig)
