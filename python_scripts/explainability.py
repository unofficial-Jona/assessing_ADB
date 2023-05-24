import sys
sys.path.append("/workspace/persistent/thesis/OadTR")

import os

from glob import glob
from zipfile import ZipFile
import xmltodict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.utils import flow_to_image

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

from pdb import set_trace

# TODO: fix model input
from custom_utils import get_args, load_model_and_checkpoint

def get_bbx_coordinates(zip_path):
    """
    return list of tensors.
    one tensor for each agent in frame ('track' in xml_file). 
    Each tensor is initialized as (nr_frames, 4). 
    Each entry at dimension 1 represents a coordinate of a bounding box. (xtl, ytl, xbr, ybr)
    """
    out_dict = dict()

    zip_file = ZipFile(zip_path)

    xml_file = xmltodict.parse(zip_file.read('annotations.xml'))['annotations']

    # get index from stop_frame
    last_frame = int(xml_file['meta']['task']['stop_frame'])

    if not isinstance(xml_file['track'], list):
        xml_file['track'] = [xml_file['track']]

    for track in xml_file['track']:
        actor_id = track['@id']
        actor_name = track['@label']
        if actor_name == 'EgoVehicle':
            continue
        track_tensor = torch.zeros((last_frame + 1, 4))
        for box in track['box']:
            frame_index = box['@frame']
            xtl = box['@xtl']
            ytl = box['@ytl']
            xbr = box['@xbr']
            ybr = box['@ybr']

            for i, coordinate in zip(range(4), [xtl, ytl, xbr, ybr]):
                track_tensor[int(frame_index), i] = float(coordinate)

        out_dict.update({actor_id:{'name':actor_name, 'bbx':track_tensor}})
    return out_dict

def coordinates_to_masks(frames_shape, coordinates):
    """
    utility function to convert tensor of shape (t, 4) to mask
    frames_shape: touple of ints, representing TCHW format
    
    returned masks are 0 in bounding box and 1 everywhere else (positive mask)
    """
    mask = torch.ones(frames_shape)

    for t, (xtl, ytl, xbr, ybr) in enumerate(coordinates):
        # Convert bounding box coordinates to integers
        xtl, ytl, xbr, ybr = map(int, [xtl, ytl, xbr, ybr])

        # Set the area inside the bounding box to 0
        mask[t, :, ytl:ybr, xtl:xbr] = 0

    return mask

def check_mask_for_bbx(mask):
    """
    utility function to check whether there is an actual bounding box within the mask.
    can be used to determine if mask can be omited.
    Check if mask is all ones or all zeros. In both cases return True
    """
    mask = mask.to(bool)
    return torch.all(mask) or torch.all(~mask)

def apply_mask(frames, mask, pos=True):
    """
    utility function to apply mask tensor to frames tensor.
    """
    assert frames.shape == mask.shape, 'frames and masks have different shapes'

    # invert mask if pos=False
    if pos == False:
        mask = ~mask.to(bool)

    # apply mask
    frames = frames * mask
    return frames

def reduce_framerate(frames, target_fps=15, source_fps = 30):
    assert source_fps % target_fps == 0, "Target FPS must be a divisor of the source FPS"

    frame_skip = int(source_fps / target_fps)
    reduced_fps_video = frames[::frame_skip, :, :, :]
    return reduced_fps_video

def get_agent_frames(video_name, frame_level_features=True, start_frame=0, length=64, **kwargs):
    if frame_level_features:
        length += 1
    
    # define paths to original videos
    ANNO_DIR = kwargs.get('anno_dir', '/workspace/pvc-meteor/downloads/Video XML Annotations/')
    VID_DIR = kwargs.get('vid_dir', '/workspace/pvc-meteor/Raw_Videos/')
    positive_mask = kwargs.get('pos', True)
    target_fps = kwargs.get('FPS', 15)
    
    zip_loc = os.path.join(ANNO_DIR, video_name[:-4] + '.zip')
    video_loc = os.path.join(VID_DIR, video_name)
    
    # load coordinates
    # {actor_id:{'name':actor_name, 'bbx':track_tensor}}
    coordinates = get_bbx_coordinates(zip_loc)
    
    # load frames
    frames, _, _ = read_video(video_loc, pts_unit='sec', output_format='TCHW')
    org_frames_shape = frames.shape
    # apply FPS change to video
    frames = reduce_framerate(frames, target_fps = target_fps)[start_frame:start_frame + length]
    
    # prepare dictionary to handle agent_id, name and masked video
    agent_dict = dict()
    # save original frames for comparison
    coordinates.update({'-1':{'name':'orig', 'masked_frames': frames}})
    # agent_dict.update({'-1':{'name':'orig', 'masked_frames': frames}})
    # for agent in video: 
    for k, v in coordinates.items():        
        if not 'bbx' in v.keys():
            continue
        coordinate_tensor = v['bbx']
        # generate mask
        mask = coordinates_to_masks(org_frames_shape, coordinate_tensor)
        # apply FPS change
        mask = reduce_framerate(mask, target_fps = target_fps)
        # cut mask (add + 1 so flow can be calculated)
        mask = mask[start_frame: start_frame + length]

        # if mask is all 1s, there is no bbx --> 
        if check_mask_for_bbx(mask):
            continue
        
        agent_frames = apply_mask(frames, mask, pos=positive_mask)

        coordinates[k].update({'masked_frames':agent_frames})
        # agent_dict.update({k:{'name':v['name'], 'masked_frames':agent_frames}})

    return coordinates

def get_conv_features(agent_dict, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    rgb_extractor = create_feature_extractor(resnet50(weights = ResNet50_Weights.DEFAULT, progress=False), return_nodes={'flatten': 'flatten'}).eval().to(device)
    flow_extractor = raft_large(weights = Raft_Large_Weights.DEFAULT, progress=False).eval().to(device)

    # disable gradients to reduce memory footprint
    for model in [rgb_extractor, flow_extractor]:
        for param in model.parameters():
            param.requires_grad = False

    feature_dict = dict()

    for k, v in agent_dict.items():
        if 'masked_frames' not in v.keys():
            continue
        assert v['masked_frames'].shape[0] == 64 + 1, 'wrong number of frames'
        frames = (v['masked_frames']/255).to(device)

        flow_stack_1 = frames[:-1]
        flow_stack_2 = frames[1:]

        rgb_transforms = ResNet50_Weights.DEFAULT.transforms()
        flow_transforms = Raft_Large_Weights.DEFAULT.transforms()

        rgb_features = rgb_extractor(rgb_transforms(flow_stack_2))['flatten']
        rgb_features = rgb_features.detach().cpu().numpy()

        flow_estimate = flow_extractor(rgb_transforms(flow_stack_1), rgb_transforms(flow_stack_2))

        flow_img = flow_to_image(flow_estimate[-1])

        flow_features = rgb_extractor(rgb_transforms(flow_img))['flatten']
        flow_features = flow_features.detach().cpu().numpy()


        agent_dict[k].update({'rgb_features':rgb_features, 'flow_features':flow_features})
        # feature_dict.update({k: {'name':v['name'], 'rgb_features':rgb_features, 'flow_features':flow_features}})
    
    return agent_dict


def get_predictions_from_model(feature_dict, model, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)
    for k,v in  feature_dict.items():
        rgb_features = v['rgb_features'].to(device)
        flow_features = v['flow_features'].to(device)
        out = model(rgb_features, flow_features)
        feature_dict[k].update({'classification':out})
    return feature_dict

if __name__ == '__main__':
    # define files for testing purposes
    TEST_NAME = 'REC_2020_09_08_04_51_57_F.MP4'
    MODEL_DIR = 'OadTR/experiments/final/features_conv_15_new.pkl/'
    
    agent_dict = get_agent_frames(TEST_NAME)
    out_features = get_conv_features(agent_dict)

    args = get_args(MODEL_DIR)
    model = load_model_and_checkpoint(args, MODEL_DIR)

    out_dict = get_predictions_from_model(out_features, model)
    print('HEUREKA!')
