# -*- coding: utf-8 -*-
"""Dataaugmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HDG3Nge3iwfSeAB1bmV9uUD5iol308p2
"""

!pip install PyTorch
!pip install TensorFlow

import pandas as pd
from random import choice, seed
from transformers import pipeline, CamembertTokenizer, CamembertForMaskedLM
import torch

!pip install sentencepiece

# Set a random seed for reproducibility
seed(42)

# Load the CamemBERT tokenizer and model
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertForMaskedLM.from_pretrained('camembert-base')

# Check and use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Initialize the fill-mask pipeline with the GPU device
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0)  # device=0 for the first GPU

# Function to replace a random word in a sentence with a mask token
def mask_random_word(sentence):
    tokens = sentence.split()
    mask_position = choice(range(len(tokens)))
    tokens[mask_position] = tokenizer.mask_token
    return " ".join(tokens), tokens[mask_position]

# Function to get predictions for the masked sentence
def get_predictions(masked_sentence, original_word):
    predictions = fill_mask(masked_sentence)
    # Filter out the original word and return the rest
    synonyms = [pred['token_str'] for pred in predictions if pred['token_str'] != original_word]
    return synonyms

!git clone https://github.com/melvin2504/Detecting-the-difficulty-level-of-French-texts/ project_repo

# Load the dataset
df = pd.read_csv("/content/project_repo/data/training_data.csv")

# Augment the dataset
augmented_sentences = []
augmented_labels = []

for _, row in df.iterrows():
    sentence, difficulty = row['sentence'], row['difficulty']
    masked_sentence, original_word = mask_random_word(sentence)
    predictions = get_predictions(masked_sentence, original_word)

    for synonym in predictions:
        new_sentence = masked_sentence.replace(tokenizer.mask_token, synonym)
        augmented_sentences.append(new_sentence)
        augmented_labels.append(difficulty)

# Create a DataFrame with the augmented data
augmented_df = pd.DataFrame({
    'sentence': augmented_sentences,
    'difficulty': augmented_labels
})

# Combine with the original dataset
augmented_df = pd.concat([df, augmented_df]).reset_index(drop=True)

# Save the augmented dataset
augmented_df.to_csv('augmented_training_data_2.csv', index=False)