
from datetime import datetime


def create_file_name():
    return datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
