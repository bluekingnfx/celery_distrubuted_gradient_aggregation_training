
import csv
from pathlib import Path
from uuid import uuid4

from helpers.produce_date_str import create_file_name

from helpers.global_config import Config

create_field_names = lambda metrics: [k for k in metrics]

def write_to_csv(
    metrics:dict['str',float | str],
    evaluation_metrics = False
):
    parent_path = Path(__file__).resolve().parent.parent
    metrics_folder = parent_path / Config.METRICS_FOLDER / Config.TESTING_FOLDER if evaluation_metrics else parent_path / Config.METRICS_FOLDER / Config.TESTING_FOLDER
    
    if metrics_folder.exists() is False:
        metrics_folder.mkdir(parents=True, exist_ok=True)
    
    file_name = metrics[Config.FILE_NAME_CONVENTION]
    
    if evaluation_metrics:
        file_path = metrics_folder / f"{file_name}.csv"
    else:
        file_path = metrics_folder / f"{file_name}.csv"
        
    epoch_id = str(uuid4())
    epoch_id = f"id_{epoch_id}"

    timestamp = create_file_name()
    
    metrics = {
        "epoch_id": epoch_id,
        Config.TIMESTAMP: timestamp, # type: ignore
        **metrics
    }
    
    del metrics[Config.FILE_NAME_CONVENTION]
    
    with open(file_path, mode='a+', newline='') as csv_file:
        fieldnames = create_field_names(metrics)
        writer = csv.DictWriter(f=csv_file, fieldnames=fieldnames)
        
        if csv_file.tell() == 0:
            writer.writeheader()
        
        writer.writerow(metrics)
    
            
