#%%
import os

from glob import glob
from zipfile import ZipFile
import xmltodict
import numpy as np
import pandas as pd
import torch.nn as nn
from torchvision.io import read_video
from torchvision.utils import flow_to_image

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

# define paths to original videos
ANNO_DIR = '/workspace/pvc-meteor/downloads/Video XML Annotations/'
VID_DIR = '/workspace/pvc-meteor/Raw_Videos/'

# define files for testing purposes
TEST_NAME = 'REC_1970_01_01_07_40_16_F.MP4'

TEST_VID = os.path.join(VID_DIR, TEST_NAME)
TEST_ANNO = os.path.join(ANNO_DIR, TEST_NAME[:-4] + '.zip')

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
    last_frame = xml_file['meta']['task']['stop_frame']

    if not isinstance(xml_file['track'], list):
        xml_file['track'] = [xml_file['track']]

    for track in xml_file['track']:
        actor_id = track['@id']
        actor_name = track['@label']
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
    mask = np.ones_like(frames_shape)

    for t, (xtl, ytl, xbr, ybr) in enumerate(coordinates):
        # Convert bounding box coordinates to integers
        xtl, ytl, xbr, ybr = map(int, [xtl, ytl, xbr, ybr])

        # Set the area inside the bounding box to 0
        mask[t, :, ytl:ybr, xtl:xbr] = 0

    return mask

def check_mask_for_bbx(mask):
    """
    utility function to check whether there is an actual bounding box within the mask.
    can be used to determine if mask can be omited
    """
    return ~torch.all(mask)

def apply_mask(frames, mask, pos=True):
    """
    utility function to apply mask tensor to frames tensor.
    """
    assert frames.shape == mask.shape, 'frames and masks have different shapes'

    # invert mask if pos=False
    if pos == False:
        mask = ~mask

    # apply mask
    frames = frames * mask
    return frames

def reduce_framerate(frames, target_fps=15, source_fps = 30):
    assert source_fps % target_fps == 0, "Target FPS must be a divisor of the source FPS"

    frame_skip = int(source_fps / target_fps)
    reduced_fps_video = frames[:, ::frame_skip, :, :]

    return reduced_fps_video

def get_agent_frames(video, frame_level_features=True, start_frame=0, length=64, **kwargs):
    if frame_level_features:
        length += 1
    
    positive_mask = kwargs.get('pos', True)
    target_fps = kwargs.get('FPS', 15)

    print('prepare coordinates')
    # load coordinates
    zip_name = os.path.join(ANNO_DIR, video[:-4] + '.zip')
    # {actor_id:{'name':actor_name, 'bbx':track_tensor}}
    coordinates = get_bbx_coordinates(zip_name)
    
    print('read frames')
    # load frames
    frames, _, _ = read_video(video, pts_unit='sec', output_format='TCHW')
    org_frames_shape = frames.shape
    # apply FPS change to video
    frames = reduce_framerate(frames, target_fps = target_fps)[start_frame:start_frame + length]

    print('apply masks')
    # prepare dictionary to handle agent_id, name and masked video
    agent_dict = dict()
    # save original frames for comparison
    agent_dict.update({'0':{'name':'orig', 'masked_frames': frames}})
    # for agent in video: 
    for k, v in coordinates.items():
        coordinate_tensor = v['bbx']

        # generate mask
        mask = coordinates_to_masks(org_frames_shape, coordinate_tensor)
        # apply FPS change
        mask = reduce_framerate(mask, target_fps = target_fps)
        # cut mask (add + 1 so flow can be calculated)
        mask = mask[start_frame: start_frame + length]

        # check if masks represents agent in selected time --> omit if not
        if check_mask_for_bbx(mask):
            continue
        
        agent_frames = apply_mask(frames, mask, pos=positive_mask)

        agent_dict.update({k + 1:{'name':v['name'], 'masked_frames':agent_frames}})

    return agent_dict

def get_conv_features(agent_dict, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    rgb_extractor = create_feature_extractor(resnet50(weights = ResNet50_Weights.DEFAULT, progress=False), return_nodes={'flatten': 'flatten'})).eval().to(device)
    flow_extractor = raft_large(weights = Raft_Large_Weights.DEFAULT, progress=False).eval().to(device)

    assert k['0']['masked_frames'].shape[0] == 64 + 1, 'wrong number of frames'
    for k, v in agent_dict.items():
        frames = v['masked_frames']to(device)

        flow_stack_1 = frames[:-1]
        flow_stack_2 = frames[1:]

        rgb_features = rgb_extractor(flow_stack_2)
        rgb_features = rgb_features.detach().cpu().numpy()

        flow_estimate = flow_extractor(flow_stack_1, flow_stack_2)

        flow_img = flow_to_image(flow_estimate)

        flow_features = rgb_extractor(flow_img)
        flow_features = flow_features.detach().cpu().numpy()

        feature_dict.update({k: {v['name'], 'rgb_features':rgb_features, 'flow_features':flow_features}})

    return feature_dict

if __name__ == '__main__':
    agent_dict = get_agent_frames()
    get_conv_features(agent_dict)


"""
def get_predictions_from_model(features, model):
    model = model.to('cuda')
    out_list = list()
    for track in features:
        rgb_track, flow_track = track
        rgb_track, flow_track = rgb_track.to('cuda'), flow_track.to('cuda')
        out = model(rgb_track, flow_track)
"""