import os
import sys
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pth")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    

    def initiate_model_trainer(self,train_df,vocab_size):

        try:
            input = "embeddings"
            target = "status"
            x_train = np.array(train_df[input].tolist())
            y_train = np.array(train_df[target].tolist())

            x_train = torch.tensor(x_train, dtype=torch.long)
            y_train = torch.tensor(y_train, dtype=torch.long)

            class CustomDataset(Dataset):
                def __init__(self,features,labels):
                    self.features = features
                    self.labels = labels

                def __len__(self):
                    return len(self.features)

                def __getitem__(self,idx):
                    return self.features[idx],self.labels[idx]
            
            train_dataset = CustomDataset(x_train,y_train)
            train_loader = DataLoader(train_dataset,batch_size=128,drop_last=True)
        
            class LSTMModel(nn.Module):
                def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
                    super().__init__()

                    # Embedding Layer
                    self.embedding = nn.Embedding(vocab_size, embedding_dim)

                    # LSTM Layer (Stacked + Bidirectional)
                    self.lstm = nn.LSTM(
                        input_size=embedding_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        batch_first=True,
                        bidirectional=True,
                        dropout=dropout
                    )

                    # Batch Normalization Layer
                    self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)  # Normalize across features

                    # Fully Connected Layer
                    self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional

                    # Dropout Layer for Regularization
                    self.dropout = nn.Dropout(dropout)

                def forward(self, x):
                    x = self.embedding(x)  # Convert words to embeddings
                    lstm_out, (hidden, _) = self.lstm(x)  # Run through LSTM

                    # Get the last hidden state from both directions
                    hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # Concatenate last forward & backward hidden states

                    # Apply Batch Normalization
                    hidden = self.batch_norm(hidden)

                    # Apply Dropout before passing to the FC layer
                    output = self.fc(self.dropout(hidden))

                    return output
            
            learning_rate = 0.01
            epochs = 30
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = LSTMModel(vocab_size=vocab_size,embedding_dim=256,hidden_dim=64,output_dim=7).to(device)

            ## loss
            criterion = nn.CrossEntropyLoss()

            ## optimizer
            optimizer = optim.Adam(model.parameters(),lr=learning_rate)

            logging.info("Model training started")

            for i in range(epochs):

                total_epoch_loss = 0

                for batch_features,batch_labels in train_loader:
                    batch_features = batch_features.long().to(device)
                    batch_labels = batch_labels.long().to(device)

                    ## forward pass
                    y_preds = model(batch_features)

                    ## loss calculation
                    loss = criterion(y_preds,batch_labels)

                    ## clearing grads
                    optimizer.zero_grad()

                    ## backward pass
                    loss.backward()

                    ## update weights
                    optimizer.step()

                    total_epoch_loss += loss.item()

                mean_loss = total_epoch_loss/len(train_loader)

                print(f" Epoch : {i+1}  Average Loss : {mean_loss}")
            
            logging.info("Model training completed")

            torch.save(model.state_dict(),self.model_trainer_config.trained_model_file_path)

        except Exception as e:
            raise CustomException(e,sys)



    