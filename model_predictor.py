import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import joblib
import torch.nn.functional as F
class FrenchSentenceDifficultyPredictor:
    def __init__(self, model_path, max_token_len=512):
        # Initialize the tokenizer
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-base-ccnet')

        # Initialize the model
        self.model = CamembertForSequenceClassification.from_pretrained('camembert/camembert-base-ccnet', num_labels=6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Load the model's state dictionary
        self.load_model(model_path)


        # Set other attributes
        self.max_token_len = max_token_len

        # Define the labels directly
        self.labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

    def load_model(self, model_path):
        # Load the state dictionary into the model, ensuring it is loaded in CPU mode
        model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(model_state_dict)
        # Make sure to move the model to the correct device after loading the weights
        self.model.to(self.device)

    def encode_label(self, label):
        # Mapping from label to encoded value
        mapping = {'A1': 0, 'A2': 1, 'B1': 2, 'B2': 3, 'C1': 4, 'C2': 5}
        return mapping.get(label, -1)  # Returns -1 for unknown labels

    def decode_label(self, encoded_label):
        # Mapping from encoded value to label
        mapping = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}
        return mapping.get(encoded_label, 'Unknown')  # Returns 'Unknown' for unknown encoded values


    def predict_difficulty(self, sentence):
        # Ensure model is in evaluation mode
        self.model.eval()

        # Tokenize the input sentence
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move tensors to the correct device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Perform the prediction
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # Apply softmax to logits to get probabilities
            probabilities = F.softmax(logits, dim=1)
            # Get the predicted label index with the highest probability
            predicted_label_idx = logits.argmax(dim=1).cpu().numpy()[0]
            # Get the scores for each category
            scores = probabilities[0].cpu().numpy()

        # Decode the predicted label
        predicted_label = self.decode_label(predicted_label_idx)

        # Create a dictionary of labels and their corresponding scores
        label_scores = {label: score for label, score in zip(self.labels, scores)}

        # Return the predicted label and the scores dictionary
        return predicted_label, label_scores

