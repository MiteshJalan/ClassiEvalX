import sys
from src.logger import logging  #for logging and saving file as exception take place in logging 

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() ##first 2 are useless last is important as it gives location and all of excption
    file_name=exc_tb.tb_frame.f_code.co_filename #from documentation custom exception handeling in python.
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
     file_name,exc_tb.tb_lineno,str(error))
    return error_message

    

class CustomException(Exception): 
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message) #as we inherrit init funtion from exception class 
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    def __str__(self): ###will be self to print it.
        return self.error_message
    


        