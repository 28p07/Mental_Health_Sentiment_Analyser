import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.batch_norm(hidden)
        output = self.fc(self.dropout(hidden))
        return output


class ModelEvaluator:
    def __init__(self):
        pass

    def initiate_model_evalution(self, test_df, vocab_size):
        try:
            input = "embeddings"
            target = "status"
            x_test = test_df[input].values
            y_test = test_df[target].values

            x_test = torch.tensor(list(x_test), dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)

            class CustomDataset(Dataset):
                def __init__(self, features, labels):
                    self.features = features
                    self.labels = labels

                def __len__(self):
                    return len(self.features)

                def __getitem__(self, idx):
                    return self.features[idx], self.labels[idx]

            test_dataset = CustomDataset(x_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=128, drop_last=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # === Rebuild the model architecture ===
            model = LSTMModel(vocab_size=vocab_size, embedding_dim=256, hidden_dim=64, output_dim=7).to(device)

            #model_path = os.path.join(os.getcwd(), "artifacts", "model.pth")
            # === Load the weights ===
            model.load_state_dict(torch.load("D:/Projects/mental_health_sentiment/artifacts/model.pth",map_location=device,weights_only=True))
            model.eval()

            criterion = nn.CrossEntropyLoss()

            def evaluate_model(model, dataloader, criterion, class_names=None):
                model.eval()
                all_preds = []
                all_labels = []
                total_loss = 0

                with torch.no_grad():
                    for batch_features, batch_labels in dataloader:
                        batch_features = batch_features.long().to(device)
                        batch_labels = batch_labels.long().to(device)

                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_labels)
                        total_loss += loss.item()

                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(batch_labels.cpu().numpy())

                avg_loss = total_loss / len(dataloader)
                accuracy = (np.array(all_preds) == np.array(all_labels)).mean()

                print(f"\nEvaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
                print("\nConfusion Matrix:")
                print(confusion_matrix(all_labels, all_preds))

                return {
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'report': classification_report(all_labels, all_preds, output_dict=True)
                }

            class_names = ["Class_0", "Class_1", "Class_2", "Class_3", "Class_4", "Class_5", "Class_6"]

            results = evaluate_model(model, test_loader, criterion, class_names)
            return results

        except Exception as e:
            raise CustomException(e, sys)