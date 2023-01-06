# %%
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.utils import flow_to_image
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from glob import glob
from zipfile import ZipFile
import xmltodict
import os
import pickle
from datetime import datetime
import warnings
import gc
import argparse

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from torchvision.io import read_video
import torch

from multiprocessing import cpu_count, Pool, log_to_stderr, Manager

# %%
class FeatureExtractor():
    def __init__(self, FPS, **kwargs):
        assert FPS in [1, 2, 3, 5, 10, 15, 30], 'for now input musst be perfect divisor of 30 ([1, 2, 3, 5, 10, 15, 30])'

        # 30 is the framerate of the METEOR Data,
        self.frame_interval = int(30/FPS)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.resize = kwargs.get('resize', (448,448))
    
        self.batch_size = kwargs.get('batch_size', 32)

        self.error_file = dict()

        if 'flow_model' in kwargs.keys():
            warnings.warn('Using a different flow model may require disabling backprop', category=UserWarning, stacklevel=2)
            self.flow_model = kwargs['flow_model']
            flow_weights = kwargs['flow_weights']
            self.transforms = flow_weights.transforms()

        else:
            self.flow_model = raft_large(
                weights=Raft_Large_Weights.DEFAULT, progress=False).to(self.device)
            for param in self.flow_model.parameters():
                param.requires_grad = False
            flow_weights = Raft_Large_Weights.DEFAULT
            self.transforms = flow_weights.transforms()

        if 'rgb_model' in kwargs.keys():
            self.rgb_model = kwargs['rgb_model']
            warnings.warn('Using a different rgb model may require disabling backprop', category=UserWarning, stacklevel=2)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=False)
            for param in model.parameters():
                param.requires_grad = False
            self.rgb_model = create_feature_extractor(
                model, return_nodes={'flatten': 'flatten'}).to(self.device)

        if 'vid_dir' in kwargs.keys():
            self.vid_dir = kwargs['vid_dir']
        else:
            self.vid_dir = '/workspace/pvc-meteor/Raw_Videos/'
        if 'anno_zip_dir' in kwargs.keys():
            self.anno_zip_dir = kwargs['anno_zip_dir']
        else:
            self.anno_zip_dir = '/workspace/pvc-meteor/downloads/Frame XML Annotations/'
        if 'output_dir' in kwargs.keys():
            self.output_dir = kwargs['output_dir']
        else:
            self.output_dir = '/workspace/pvc-meteor/features/'

        if 'labels' in kwargs.keys():
            self.labels = kwargs['labels']
        else:
            self.labels = {'RuleBreak': {'WrongLane', 'WrongTurn', 'TrafficLight'}, 'LaneChanging': {'True'}, 'LaneChanging(m)': {
                'True'}, 'OverTaking': {'True'}, 'Yield': {'True'}, 'Cutting': {'True'}, 'ZigzagMovement': {'True'}, 'OverSpeeding': {'True'}}
        if 'labels_mapping' in kwargs.keys():
            self.labels_mapping = kwargs['labels_mapping']
        else:
            self.labels_mapping = {'RuleBreak': 0, 'LaneChanging': 1,
                                   'LaneChanging(m)': 1, 'OverTaking': 2, 'Yield': 3, 'Cutting': 4, 'ZigzagMovement': 5, 'OverSpeeding': 6}

    def video_processor(self, video):
        frames, _, _ = read_video(video, pts_unit='sec', output_format='TCHW')

        rgb_frames = torch.stack([frames[i] for i in range(
            frames.shape[0]) if i % self.frame_interval == 0])
        rgb_frames = TF.resize(rgb_frames, size=self.resize)
        # divide by 255 to map pixel values to interval [0,1]
        rgb_frames = (rgb_frames/255).to(self.device)

        # extract rgb_features
        rgb_chunks = torch.split(rgb_frames, split_size_or_sections=self.batch_size, dim=0)
        rgb_features = []

        for chunk in rgb_chunks:
            features = self.rgb_model(chunk)['flatten']
            rgb_features.append(features)

        # gc.collect(features, rgb_chunks, chunk)
        rgb_features = torch.cat(rgb_features, dim=0).to('cpu')
        rgb_features = rgb_features[1:]

        # extract flow_features
        flow_stack1, flow_stack2 = self.transforms(rgb_frames[:-1], rgb_frames[1:])
        flow_stack1 = torch.split(flow_stack1, split_size_or_sections=self.batch_size, dim=0)
        flow_stack2 = torch.split(flow_stack2, split_size_or_sections=self.batch_size, dim=0)

        flow_features = []

        for stack1, stack2 in zip(flow_stack1, flow_stack2):
            features = self.flow_model(stack1, stack2)
            # append last item from extracted flow_features, as it corresponds to last iteration (most accurate one)
            features = flow_to_image(features[-1])
            features = self.rgb_model(features/255)['flatten']
            flow_features.append(features)

        # gc.collect(features, stack1, stack2, flow_stack1, flow_stack2)
        flow_features = torch.cat(flow_features, dim=0).to('cpu')

        # assert flow_features.shape == rgb_features.shape, f'shapes of rgb ({rgb_features.shape}) and flow ({flow_features.shape}) features do not match'

        return rgb_features.detach().cpu().resolve_conj().resolve_neg().numpy(), flow_features.detach().cpu().resolve_conj().resolve_neg().numpy()

    # '/workspace/pvc-meteor/downloads/Video XML Annotations'
    def annotation_processor(self, video_name):
        zip_name = video_name[:-4] + '.zip'
        # check if name exist

        if zip_name not in os.listdir(self.anno_zip_dir):
            self.error_file[video_name] = 'zip file not found'
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
                frame_name = 'Annotations/' + \
                    frame_name if append_folder else f'{zip_name[:-4]}/Annotations/' + frame_name

                xml_file = xmltodict.parse(zip_file.read(frame_name))[
                    'annotation']

                if 'object' not in xml_file:
                    # behave as if no file found
                    return None

                if not isinstance(xml_file['object'], list):
                    xml_file['object'] = [xml_file['object']]

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
            rgb_features, flow_features = self.video_processor(
                os.path.join(self.vid_dir, video_file_name))

        if rgb_features.shape != flow_features.shape:
            print(f'rgb_shape={rgb_features.shape}, flow_shape={flow_features.shape}')
            self.error_file[video_file_name] = 'feature shape mismatch'
            return None, None
        if annotation.shape[0] != rgb_features.shape[0]:
            self.error_file[video_file_name] = 'annotation shape mismatch'
            return None, None

        # create dictionary and write to pickle file
        return {
            video_file_name:
                {
                    'rgb': rgb_features,
                    'flow': flow_features
                }
        }, {video_file_name: annotation}

    def dir_to_features(self, save=True, use_mp=False):
        if use_mp: 
            warnings.warn('using multiprocessing with cuda is currently not supported', category=UserWarning, stacklevel=2)
            
        # keep start time to differentiate between pickled files
        start_time = datetime.now()
        start_time = start_time.strftime("%d-%m-%Y-%H-%M")

        general_informaion = {
            'fps': self.frame_interval, 
            'rgb_extractor': self.rgb_model.__class__.__name__,
            'flow_extractor': self.flow_model.__class__.__name__, 
            'extraction_time': start_time
            }

        output_dict = {'meta': general_informaion,
                       'features': dict(), 
                       'annotations': dict()
                       }

        if use_mp:
            # pool = Pool(processes=cpu_count())

            path_iterator = glob(self.vid_dir + '/*.MP4')

            with Pool(processes=2) as p:
                result_iterator = tqdm(p.imap_unordered(self.file_to_features, path_iterator), total=len(path_iterator), desc="Extracting features")

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
                try:
                    vid_features, vid_annot = self.file_to_features(file_path)

                    if vid_features == None:
                        continue

                    output_dict['features'].update(vid_features)
                    output_dict['annotations'].update(vid_annot)


                    if save:
    
                        file_name = f'file_features_{file_path[-29:-4]}.pkl'
                        vid_save = {'meta': general_informaion,
                            'features': vid_features, 
                            'annotations': vid_annot
                            }

                        with open(self.output_dir + file_name, 'wb') as file:
                            pickle.dump(vid_save, file)

                except:
                    self.error_file[file_path[-29:-4]] = 'catched by try-except block'
                    continue

        output_dict['error_file'] = self.error_file
        
        if save:
            # create the file name using the start_time and extracted_features variables
            file_name = f'extracted_features_{start_time}.pkl'

            # open the file in write mode
            with open(self.output_dir + file_name, 'wb') as file:
                # use pickle to serialize the dictionary and write it to the file
                pickle.dump(output_dict, file)
                
        return output_dict


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--FPS', type=int)
    config = parser.parse_args()


    extractor = FeatureExtractor(config.FPS)
    extractor.dir_to_features()