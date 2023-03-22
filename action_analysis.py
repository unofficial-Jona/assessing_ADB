from zipfile import ZipFile
import xmltodict
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

anno_path = '../../pvc-meteor/downloads/Frame XML Annotations/'
labels = {'LaneChanging': {'True'}, 'LaneChanging(m)': {'True'}, 'OverTaking': {'True'}, 'Cutting': {'True'}, 'OverSpeeding': {'True'},'RuleBreak':{'TrafficLight', 'WrongLane', 'WrongTurn'}, 'WrongLane': {'WrongLane'}, 'WrongTurn':{'WrongTurn'}, 'TrafficLight':{'TrafficLight'}}
label_mapping = {'OverTaking': 0, 'OverSpeeding': 1, 'LaneChanging(m)': 2, 'LaneChanging': 2, 'TrafficLight': 3, 'WrongLane': 4, 'WrongTurn': 5, 'Cutting': 6}


df = pd.DataFrame(columns=label_mapping.keys())


for file_name in tqdm(os.listdir(anno_path)):
    if '.zip' not in file_name:
        continue
    zip_file = ZipFile(os.path.join(anno_path, file_name))
    
    # iterate through frames
    for zip_content_name in zip_file.namelist():
        if '.xml' not in zip_content_name:
            continue
        try: 
            xml_file = xmltodict.parse(zip_file.read(zip_content_name))['annotation']
        except:
            print(f'{file_name} was catched by try_except block')
            continue

        if 'object' not in xml_file:
            continue
        
        if not isinstance(xml_file['object'], list):
            xml_file['object'] = [xml_file['object']]

        for obj in xml_file['object']:
            # iteratively build index of df
            if obj['name'] not in df.index:
                df.loc[obj['name'],:] = np.zeros(len(label_mapping.keys()))
            
            for attr in obj['attributes']['attribute']:
                if 'GPSData' in attr or attr['name'] not in labels.keys():
                    continue

                if attr['name'] in labels.keys() and attr['value'] in labels[attr['name']]:
                    if attr['name'] == 'RuleBreak':
                        df.loc[obj['name'], attr['value']] += 1 
                        continue
                    df.loc[obj['name'], attr['name']] += 1

df.to_csv('actions_by_actors.csv')