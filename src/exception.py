import sys
import logging
import logger

def error_details(error, details : sys):
    _, _, exc_tb = details.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = f'Error occured in python script \nName: {filename}\nLine Number: {exc_tb.tb_lineno}\nDetails:{str(error)}'
    
    return error_message

class Custom_Exp(Exception):
    def __init__(self, error_message, details):
        super().__init__(error_message)
        self.error_message = error_details(error_message, details = details)
        
    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1/0
    
    except Exception as e:
        logger.logging.warning("Warning has been made")
        raise Custom_Exp(e, sys)