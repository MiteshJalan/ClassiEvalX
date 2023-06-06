#for data injection even from mongo dvb etc can be used
import os
import sys
from src.exception import CustomException
from src.logger import logging  #import logging and exception 
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

# from src.components.model_trainer_math import ModelTrainerConfig_math
from src.components.model_trainer import ModelTrainer

from src.components.model_trainer_reading import ModelTrainerConfig_reading
from src.components.model_trainer_reading import ModelTrainer_reading

from src.components.model_trainer_writing import ModelTrainerConfig_writing
from src.components.model_trainer_writing import ModelTrainer_writing

@dataclass #declarator to use this DataIngestionConfig instead of __init__ we use this directly
class DataIngestionConfig:   #take inputs in the file
    #all 3 below are inputs to dataIngComponent and now it knows where to save them. at this point we dont have artifacts folder and csv for raw,test train
    train_data_path: str=os.path.join('artifacts',"train.csv") #path to ingestion even for output
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")


class DataIngestion:
    def __init__(self):#as soon as its called the path will be saved for all train test raw.
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self): #read data from database
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\stud.csv') #read csv file for data ingestion 
            logging.info('Read the dataset as dataframe') #create log file to see if process completed
            #create the dirctory artifacts at this point
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #exist_ok means if floder is present use same folder.
            #also above st. will help to make train folder.
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # save to dirctor. the raw data 
            logging.info("Train test split initiated") #important to see status of file
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) 

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)#save data to same dir.

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)#save data to same dir.

            logging.info("Inmgestion of the data iss completed") 

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    logging.info("Done DataInjection in main")
    data_transformation_math=DataTransformation("preprocessor_math.pkl")
    data_transformation_reading=DataTransformation("preprocessor_reading.pkl")
    data_transformation_writing=DataTransformation("preprocessor_writing.pkl")

    train_arr_math,test_arr_math,_=data_transformation_math.initiate_data_transformation(train_data,test_data)
    train_arr_reading,test_arr_reading,_=data_transformation_reading.initiate_data_transformation(train_data,test_data)
    train_arr_writing,test_arr_writing,_=data_transformation_writing.initiate_data_transformation(train_data,test_data)
    logging.info("Done Data transformation in main")
    modeltrainer_math=ModelTrainer("model_math.pkl")
    modeltrainer_writing=ModelTrainer("model_writing.pkl")
    modeltrainer_reading=ModelTrainer("model_reading.pkl")
    # modeltrainer_reading=ModelTrainer_reading()
    # modeltrainer_writing=ModelTrainer_writing()
    # print(train_arr_math,test_arr_math,train_arr_reading,test_arr_reading,train_arr_writing,test_arr_writing)

    print(modeltrainer_math.initiate_model_trainer(train_arr_math,test_arr_math))
    print(modeltrainer_writing.initiate_model_trainer(train_arr_writing,test_arr_writing))
    print(modeltrainer_reading.initiate_model_trainer(train_arr_reading,test_arr_reading))