import os
import sys
from dataclasses import dataclass

import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTranformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def clean(self,txt):
        if isinstance(txt,str):
            txt  = re.sub(r'[^\x00-\x7F]+', '', txt)
            txt = re.sub(r"http[s]?://\S+|\[.*?\]\(.*?\)|@\w+", "", txt)
            txt = re.sub(r"[^a-zA-Z\s]","",txt)
            txt = re.sub(r"\s+"," ",txt).strip()
            return txt
        return txt


    def create_word_index(self,train_data,test_data):

        main_column = "statement"

        data = pd.concat([train_data,test_data],axis=0)
        data = data.reset_index(drop=True)

        all_words = []

        for i in range(data[main_column].shape[0]):
            all_words.extend(data[main_column][i])
        
        word_count = Counter(all_words)

        word_to_index = {"<PAD>": 0, "<UNK>": 1}  # Padding and Unknown token
        word_to_index.update({word: idx+2 for idx, (word, _) in enumerate(word_count.most_common())})

        return word_to_index
    
    def encode_sentences(self,sentence,word_to_index):

        indices = [word_to_index.get(word,word_to_index['<UNK>']) for word in sentence]

        if len(indices)<256: ##max length 256
            indices += [word_to_index['<PAD>']]*(256-len(indices))

        return indices[:256]


    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")
            logging.info("Starting tranformation process")


            main_column = "statement"
            target_column = "status"
            column_to_be_removed = "Unnamed: 0"

            
            train_df = train_df.drop(columns=[column_to_be_removed],axis=1)
            test_df = test_df.drop(columns=[column_to_be_removed],axis=1)

            train_df = train_df.dropna().reset_index(drop=True)
            train_df = train_df.drop_duplicates().reset_index(drop=True)
            logging.info("Removed null values")

            test_df = test_df.dropna().reset_index(drop=True)
            test_df = test_df.drop_duplicates().reset_index(drop=True)
            logging.info("Removed duplicates values")

            for df in [train_df, test_df]:
                df[main_column] = df[main_column].str.lower()
                df[main_column] = df[main_column].apply(lambda x: self.clean(str(x)))
                df[main_column] = df[main_column].apply(lambda x: (word_tokenize(str(x))))
            logging.info("Cleaned text and did tokenization")
            
            word_to_index = self.create_word_index(train_df,test_df)
            vocab_size = len(word_to_index)
            logging.info("Created word to index")

            for df in [train_df,test_df]:
                df['embeddings'] = df[main_column].apply(lambda x:self.encode_sentences(x,word_to_index))
                df[target_column] = df[target_column].map({'Normal':int(0),'Depression':int(1),'Suicidal':int(2),'Anxiety':int(3),'Bipolar':int(4),'Stress':int(5),"Personality disorder":int(6)})
            
            logging.info("Embeddings created")

            return train_df,test_df,vocab_size


        except Exception as e:
            raise CustomException(e,sys)


