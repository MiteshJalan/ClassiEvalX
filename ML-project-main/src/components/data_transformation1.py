#for handeling data missing or not na etc., categorical numerical fearures handeling..
import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer #from notebook for pipeline > 
from sklearn.impute import SimpleImputer # for missing values
from sklearn.pipeline import Pipeline # 
from sklearn.preprocessing import OneHotEncoder,StandardScaler # encoding 

from src.exception import CustomException #do exception handeling here too 
from src.logger import logging 
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path_math=os.path.join('artifacts',"preprocessor_math.pkl") #give i/p and o/p pipeline pkle file for 
    preprocessor_obj_file_path_writing=os.path.join('artifacts',"preprocessor_writing.pkl")
    preprocessor_obj_file_path_reading=os.path.join('artifacts',"preprocessor_reading.pkl")
class DataTransformation: 
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()#will preprocessor_obj_file_path variable. 

    def get_data_transformer_object(self,col1,col2):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            self.col1=col1
            self.col2=col2
            numerical_columns = [self.col1, self.col2]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj_math=self.get_data_transformer_object("writing_score","reading_score")#has all pipelines
            preprocessing_obj_writing=self.get_data_transformer_object("reading_score","math_score")
            preprocessing_obj_reading=self.get_data_transformer_object("writing_score","math_score")
            logging.info("Done Preprocessing for all 3")

            target_column_name_math="math_score"
            target_column_name_reading="reading_score"
            target_column_name_writing="writing_score"
            numerical_columns_math = ["writing_score", "reading_score"]
            numerical_columns_reading=["writing_score","math_score"]
            numerical_columns_writing=["reading_score","math_score"]

            input_feature_train_df_math=train_df.drop(columns=[target_column_name_math],axis=1)
            input_feature_train_df_reading=train_df.drop(columns=[target_column_name_reading],axis=1)
            input_feature_train_df_writing=train_df.drop(columns=[target_column_name_writing],axis=1)
            
            target_feature_train_df_math=train_df[target_column_name_math]
            target_feature_train_df_reading=train_df[target_column_name_reading]
            target_feature_train_df_writing=train_df[target_column_name_writing]



            input_feature_test_df_math=test_df.drop(columns=[target_column_name_math],axis=1)
            input_feature_test_df_reading=test_df.drop(columns=[target_column_name_reading],axis=1)
            input_feature_test_df_writing=test_df.drop(columns=[target_column_name_writing],axis=1)

            
            target_feature_test_df_math=test_df[target_column_name_math]
            target_feature_test_df_writing=test_df[target_column_name_writing]
            target_feature_test_df_reading=test_df[target_column_name_reading]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe. 121"
            )

            input_feature_train_arr_math=preprocessing_obj_math.fit_transform(input_feature_train_df_math)#fit transform train df
            input_feature_train_arr_reading=preprocessing_obj_reading.fit_transform(input_feature_train_df_reading)
            input_feature_train_arr_writing=preprocessing_obj_writing.fit_transform(input_feature_train_df_writing)
            
            input_feature_test_arr_math=preprocessing_obj_math.transform(input_feature_test_df_math)#transform test df
            input_feature_test_arr_reading=preprocessing_obj_reading.transform(input_feature_test_df_reading)
            input_feature_test_arr_writing=preprocessing_obj_writing.transform(input_feature_test_df_writing)
            
            train_arr_math = np.c_[
                input_feature_train_arr_math, np.array(target_feature_train_df_math)
            ]
            train_arr_reading = np.c_[
                input_feature_train_arr_reading, np.array(target_feature_train_df_reading)
            ]
            train_arr_writing = np.c_[
                input_feature_train_arr_writing, np.array(target_feature_train_df_writing)
            ]

            test_arr_math = np.c_[input_feature_test_arr_math, np.array(target_feature_test_df_math)]
            test_arr_reading = np.c_[input_feature_test_arr_reading, np.array(target_feature_test_df_math)]
            test_arr_writing = np.c_[input_feature_test_arr_writing, np.array(target_feature_test_df_math)]

            logging.info(f"Saved preprocessing object.")

            save_object( #imported from util.py

                file_path=self.data_transformation_config.preprocessor_obj_file_path_math,
                obj=preprocessing_obj_math

            )
            save_object( #imported from util.py

                file_path=self.data_transformation_config.preprocessor_obj_file_path_reading,
                obj=preprocessing_obj_reading

            )
            save_object( #imported from util.py

                file_path=self.data_transformation_config.preprocessor_obj_file_path_writing,
                obj=preprocessing_obj_writing

            )

            return (
                train_arr_math,
                test_arr_math,
                train_arr_reading,
                test_arr_reading,
                train_arr_writing,
                test_arr_writing,
                self.data_transformation_config.preprocessor_obj_file_path_math,
                self.data_transformation_config.preprocessor_obj_file_path_writing,
                self.data_transformation_config.preprocessor_obj_file_path_reading
            )
        except Exception as e:
            raise CustomException(e,sys)
