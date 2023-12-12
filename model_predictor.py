import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import joblib
import torch.nn.functional as F
class FrenchSentenceDifficultyPredictor:
    def __init__(self, model_path, label_encoder_path, max_token_len=512):
        # Initialize the tokenizer
        self.tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-base-ccnet')

        # Initialize the model
        self.model = CamembertForSequenceClassification.from_pretrained('camembert/camembert-base-ccnet', num_labels=6)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Load the model's state dictionary
        self.load_model(model_path)

        # Load the label encoder
        self.label_encoder = joblib.load(label_encoder_path)

        # Set other attributes
        self.max_token_len = max_token_len

    def load_model(self, model_path):
        # Load the state dictionary into the model
        model_state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(model_state_dict)



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
        predicted_label = self.label_encoder.inverse_transform([predicted_label_idx])[0]

        # Create a dictionary of labels and their corresponding scores
        labels = self.label_encoder.classes_
        label_scores = {label: score for label, score in zip(labels, scores)}

        # Return the predicted label and the scores dictionary
        return predicted_label, label_scores

