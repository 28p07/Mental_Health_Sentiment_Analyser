import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_tansformation import DataTranformation
from src.components.model_trainer import ModelTrainer

logging.info("Training pipeline started")
try:
    data_ingestion = DataIngestion()
    train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTranformation()
    train_df,test_df,vocab_size = data_transformation.initiate_data_transformation(train_path=train_data_path,test_path=test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_df=train_df,vocab_size=vocab_size)
except Exception as e:
    raise CustomException(e,sys)
logging.info("Training pipeline ended")