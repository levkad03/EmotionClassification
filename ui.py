import pickle

import plotly.express as px
import streamlit as st
import torch

from config import (
    CHECKPOINT_PATH,
    DEVICE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    INDEX_TO_EMOTION,
    NUM_LAYERS,
    PAD_IDX,
)
from model import EmotionClassifier
from utils import load_checkpoint

st.set_page_config(page_title="Emotion Classification", page_icon=":smiley:")


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

    emotion = INDEX_TO_EMOTION.get(prediction, "Unknown")
    st.write(f"Predicted emotion: **{emotion}**")

    emotions = list(INDEX_TO_EMOTION.values())
    fig = px.bar(
        x=emotions,
        y=probabilities,
        labels={"x": "Emotion", "y": "Probability"},
        title="Emotion Prediction Probabilities",
        color=emotions,
    )

    st.plotly_chart(fig)
