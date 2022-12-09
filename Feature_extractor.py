import numpy as np
from glob import glob
from zipfile import ZipFile
import xmltodict
import os
import pickle
from datetime import datetime
import json
# import cv2

from tqdm import tqdm
from torchvision.io import read_video
import torch

from sklearn.model_selection import train_test_split

from multiprocessing import cpu_count, Pool, log_to_stderr
# TODO: disable log_to_stderr()
log_to_stderr()

# imports for RGB feature extraction
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names

# imports for optical flow 
from torchvision.utils import flow_to_image
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import torchvision.transforms.functional as TF

# write class to represent a single video 
# --> in this class the frame reading process should take place, so it can be nicely dsitributed.


class FeatureExtractor():
    def __init__(self, FPS, resize, rgb_model, flow_model, **kwargs):
        assert FPS in [1, 2, 3, 5, 10, 15, 30], 'for now input musst be perfect divisor of 30 ([1, 2, 3, 5, 10, 15, 30])'        
                
        self.frame_interval = int(30/FPS) # 30 is the framerate of the METEOR Data, 
        self.resize = resize
        self.rgb_model = rgb_model
        self.flow_model = flow_model
        
        if self.flow_model.__class__.__name__ == 'RAFT':
            self.flow_weights = Raft_Large_Weights.DEFAULT
            self.transforms = self.flow_weights.transforms()

        self.vid_dir = '/workspace/pvc-meteor/Raw_Videos'
        self.anno_zip_dir = '/workspace/pvc-meteor/downloads/Video XML Annotations'
        self.output_dir = '/workspace/pvc-meteor/dataset_aggresive'
        

        if 'vid_dir' in kwargs.keys():
            self.vid_dir = kwargs['vid_dir']
        if 'anno_zip_dir' in kwargs.keys():
            self.anno_zip_dir = kwargs['anno_zip_dir']
        if 'output_dir' in kwargs.keys():
            self.output_dir = kwargs['output_dir']

        self.labels = {'RuleBreak': {'WrongLane', 'WrongTurn', 'TrafficLight'}, 'LaneChanging': {'True'}, 'LaneChanging(m)': {'True'}, 'OverTaking': {'True'}, 'Yield': {'True'}, 'Cutting': {'True'}, 'ZigzagMovement': {'True'}, 'OverSpeeding':{'True'}}
        self.labels_mapping = {'RuleBreak': 0, 'LaneChanging': 1, 'LaneChanging(m)': 1, 'OverTaking': 2, 'Yield': 3, 'Cutting': 4, 'ZigzagMovement':5, 'OverSpeeding':6}


    def video_processor(self, video):
        
        frames, _, _ = read_video(video, pts_unit='sec', output_format='TCHW')

        rgb_frames = torch.stack([frames[i] for i in range(frames.shape[0]) if i % self.frame_interval == 0])
        rgb_frames = frames
        rgb_frames = TF.resize(rgb_frames, size = self.resize)

        # need to access dict object by node name


        rgb_features = self.rgb_model(rgb_frames[1:]/255)['flatten']
        flow_stack1, flow_stack2 = self.transforms(rgb_frames[:-1], rgb_frames[1:])
        flow_features = self.flow_model(flow_stack1, flow_stack2)
        flow_features = flow_to_image(flow_features[-1])
        flow_features = self.rgb_model(flow_features/255)['flatten']

        # cut first frame from rgb stack so flow_features.shape[0] == rgb_features.shape[0]
        # rgb_features = rgb_features[1:]
        
        assert flow_features.shape == rgb_features.shape, 'shapes of rgb and flow features do not match'

        return rgb_features.detach().cpu().resolve_conj().resolve_neg().numpy(), flow_features.detach().cpu().resolve_conj().resolve_neg().numpy()


    def annotation_processor(self, video_name): # '/workspace/pvc-meteor/downloads/Video XML Annotations'
        zip_name = video_name[:-4] + '.zip'
        # check if name exist

        if zip_name not in os.listdir(self.anno_zip_dir):
            return None
        
        else:
            # load zip object
            zip_file = ZipFile(os.path.join(self.anno_zip_dir, zip_name))
            nr_frames = len([i for i in zip_file.namelist() if '.xml' in i])
            nr_frames = np.floor(nr_frames / self.frame_interval).astype(int)
            template = np.zeros((nr_frames, 7))
            
            append_folder = 'Annotations/' in zip_file.namelist()
            # iterate through key-frames
            for i_temp, i_frame in enumerate([i * self.frame_interval for i in range(nr_frames)]):
                frame_name = 'frame_{0:06d}.xml'.format(i_frame)
                frame_name = 'Annotations/' + frame_name if append_folder else f'{zip_name[:-4]}/Annotations/' + frame_name
                
                xml_file = xmltodict.parse(zip_file.read(frame_name))['annotation']
                
                if 'object' not in xml_file:
                    # behave as if no file found
                    return None 
                    
                
                if not isinstance(xml_file['object'], list):
                    xml_file['object']= [xml_file['object']]
                
                for obj in xml_file['object']:
                    if obj['name'] != 'EgoVehicle':
                        continue
                    for attr in obj['attributes']['attribute']:
                        if 'GPSData' in attr:
                            continue
                        if attr['name'] in self.labels:
                            if attr['value'] in self.labels[attr['name']]:
                                c_idx = self.labels_mapping[attr['name']]

                                template[i_temp, c_idx] = 1
        return template[1:]

    def file_to_features(self, file_path):
        video_file_name = file_path[-29:]
        zip_file_name = video_file_name[:-4] + '.zip'
        
        annotation = self.annotation_processor(zip_file_name)

        # skip video if no annotations are available
        if not isinstance(annotation, np.ndarray):
            return None, None

        else:
            rgb_features, flow_features = self.video_processor(os.path.join(self.vid_dir, video_file_name))
        
        assert rgb_features.shape == flow_features.shape, 'rgb and flow features have different shape'
        assert annotation.shape[0] == rgb_features.shape[0], 'annotations and image features have different length'

        # create dictionary and write to pickle file
        return {
            video_file_name:
                {
                    'rgb': rgb_features,
                    'flow': flow_features
                }
            }, {video_file_name: annotation}
        
    def create_json(self, output_dict, train_size=0.8, test_size=0.2):
        assert train_size + test_size == 1, 'relative set sizes do not add up to one'

        vid_names = list(output_dict['annotations'].keys())
        train, test = train_test_split(vid_names, test_size = test_size)

        # prepare json
        json_dict = {
            'class_index' : list(self.labels.keys()),
            'train_session_set' : train,
            'test_session_set' : test,
        }
        return json_dict

    def dir_to_features(self, save=True, use_mp=True):
        # keep start time to differentiate between pickled files
        start_time = datetime.now()
        start_time = start_time.strftime("%d-%m-%Y-%H-%M")

        general_informaion = {'fps': self.frame_interval, 'rgb_extractor': self.rgb_model.__class__.__name__, 'flow_extractor':self.flow_model.__class__.__name__, 'extraction_time':start_time}

        output_dict = {'meta': general_informaion, 'features':dict(), 'annotations':dict()}

        if use_mp:
            # pool = Pool(processes=cpu_count())

            path_iterator = glob(self.vid_dir + '/*.MP4')

            with Pool(processes = 2) as p:
                result_iterator = p.map(self.file_to_features, path_iterator)
            
                p.close()
                p.join()
                
            for result in result_iterator:
                vid_features, vid_annot = result
                if vid_features == None:
                    continue
                output_dict['features'].update(vid_features)
                output_dict['annotations'].update(vid_annot)


        else:
            for file_path in tqdm(glob(self.vid_dir + '/*.MP4')):
                vid_features, vid_annot = self.file_to_features(file_path)
                                
                if vid_features == None:
                    continue

                output_dict['features'].update(vid_features)
                output_dict['annotations'].update(vid_annot)
                

        if save:
            # create the file name using the start_time and extracted_features variables
            pickle_name = 'extracted_features_{}.pkl'.format(start_time)
            json_name = 'data_info_new_{}.json'.format(start_time)

            # open the file in write mode
            with open(self.output_dir + pickle_name, 'wb') as file:
            # use pickle to serialize the dictionary and write it to the file
                pickle.dump(output_dict, file)
            
            json_dict = self.create_json(output_dict)
            with open(self.output_dir + json_name, 'wb') as file:
                json.dump(json_dict, file)                

        return output_dict



if __name__ == '__main__':
    prototype = FeatureExtractor(
        FPS=15, #METEOR Dataset has 30 FPS --> only taking every 30th frame means using only 1 frame every second. 
        resize=(448,448), 
        rgb_model = create_feature_extractor(resnet50(weights=ResNet50_Weights.DEFAULT, progress=False), return_nodes={'flatten':'flatten'}), 
        flow_model =  raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False),
        vid_dir = '../data sample/videos/',
        output_dir = '../data sample/output/',
        anno_zip_dir = '../data sample/annotations/frame'
        )

    # prototype.video_processor('/workspace/persistent/data sample/videos/REC_1970_01_01_07_40_16_F.MP4')
    # prototype.annotation_processor2('REC_1970_01_01_07_40_16_F.zip', '/workspace/persistent/data sample/annotations/frame')
    prototype.dir_to_features()

