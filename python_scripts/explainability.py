import sys
sys.path.append("/workspace/persistent/thesis/OadTR")

import os
import xmltodict
import numpy as np
import pandas as pd
import pickle
import json

from glob import glob
from zipfile import ZipFile
from pdb import set_trace
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.utils import flow_to_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines


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
        if actor_name not in ['Car', 'MotorBike', 'Scooter']:
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

def reduce_framerate(frames, start_frame, length, target_fps=15, source_fps = 30):
    assert source_fps % target_fps == 0, "Target FPS must be a divisor of the source FPS"

    frame_skip = int(source_fps / target_fps)
    # create np.array that handles the index to keep
    keep_idx = np.arange(frames.shape[0])[:start_frame + 1][::-frame_skip][::-1][-length:]
    
    reduced_fps_video = frames[keep_idx]
    return reduced_fps_video


def get_agent_frames(video_name, frame_level_features=True, start_frame=65, length=64, **kwargs):
    if frame_level_features:
        length += 1
    
    # define paths to original videos
    ANNO_DIR = kwargs.get('anno_dir', '/workspace/pvc-meteor/downloads/Video XML Annotations/')
    VID_DIR = kwargs.get('vid_dir', '/workspace/pvc-meteor/Raw_Videos/')
    positive_mask = kwargs.get('pos_mask', True)
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
    frames = reduce_framerate(frames, start_frame, length, target_fps = target_fps)
    # prepare dictionary to handle agent_id, name and masked video


    # save original frames for comparison
    coordinates.update({'-1':{'name':'orig', 'masked_frames': frames}})
    # for agent in video: 
    for k, v in coordinates.items():        
        if not 'bbx' in v.keys():
            continue
        coordinate_tensor = v['bbx']
        # reduce FPS of coordinate tensor and cut to correct times
        coordinate_tensor = reduce_framerate(coordinate_tensor, start_frame, length, target_fps = target_fps)

        # save reduced bbx tensor instead of the old one
        v.update({'bbx': coordinate_tensor})

        # generate mask
        mask = coordinates_to_masks(frames.shape, coordinate_tensor)

        # if mask is all 1s, there is no bbx --> 
        if check_mask_for_bbx(mask):
            continue
        
        agent_frames = apply_mask(frames, mask, pos=positive_mask)

        coordinates[k].update({'masked_frames':agent_frames})
    return coordinates

def add_conv_features_to_dict(dic, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    rgb_extractor = create_feature_extractor(resnet50(weights = ResNet50_Weights.DEFAULT, progress=False), return_nodes={'flatten': 'flatten'}).eval().to(device)
    flow_extractor = raft_large(weights = Raft_Large_Weights.DEFAULT, progress=False).eval().to(device)

    # disable gradients to reduce memory footprint
    for model in [rgb_extractor, flow_extractor]:
        for param in model.parameters():
            param.requires_grad = False

    feature_dict = dict()

    for k, v in dic.items():
        if 'masked_frames' not in v.keys():
            continue
        assert v['masked_frames'].shape[0] == 64 + 1, f"got {v['masked_frames'].shape[0]} frames instead of 65"
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


        dic[k].update({'rgb_features':rgb_features, 'flow_features':flow_features})
        # feature_dict.update({k: {'name':v['name'], 'rgb_features':rgb_features, 'flow_features':flow_features}})
    
    return dic

def add_model_predictions_to_dict(feature_dict, model, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)
    for k,v in  feature_dict.items():
        if 'masked_frames' not in v.keys():
            continue
        rgb_features = torch.from_numpy(v['rgb_features']).to(device)
        flow_features = torch.from_numpy(v['flow_features']).to(device)
        out = model(rgb_features.unsqueeze(0), flow_features.unsqueeze(0))
        # use first element of tuple output (represents final classification score)
        out = out[0]
        # transform values into range [0,1]
        out = torch.sigmoid(out)
        out = out.detach().cpu().numpy()
        feature_dict[k].update({'prediction':out})
    return feature_dict

def add_prediction_differences_to_dict(prediction_dict):
    orig_predict = prediction_dict['-1']['prediction']
    for k, v in prediction_dict.items():
        # only agents that are in the scene
        if 'prediction' not in v.keys():
            continue
        
        agent_predict = v['prediction']
        pred_dif = orig_predict - agent_predict
        # pred_dif = torch.sigmoid(torch.from_numpy(pred_dif)).numpy()
        prediction_dict[k].update({'prediction_dif':pred_dif})
    return prediction_dict

def get_true_labels(vid_name, frame, **kwargs):
    # load zip_file
    # iterate through tracks
    # if @name not in actors: skip
    # for box in track: if box@frame != frame: skip
    # return_dict with bbx + binary array --> indicate if action is present.
    MAPPING = {'OverTaking':0, 'LaneChanging':1, 'LaneChanging(m)':1, 'RuleBreak':2, 'Cutting':3}
    ATTRIBUTES = ['LaneChanging', 'LaneChanging(m)', 'OverTaking', 'Cutting', 'RuleBreak']
    VALUES = ['True', 'true', 'WrongLane']

    ANNO_DIR = kwargs.get('anno_dir', '/workspace/pvc-meteor/downloads/Video XML Annotations/')
    zip_name = vid_name if vid_name.endswith('.zip') else vid_name[:-4] + '.zip'
    zip_path = os.path.join(ANNO_DIR, zip_name)
    zip_file = ZipFile(zip_path)
    xml_file = xmltodict.parse(zip_file.read('annotations.xml'))['annotations']

    if not isinstance(xml_file['track'], list):
        xml_file['track'] = [xml_file['track']]
    
    out_dict = dict()

    for track in xml_file['track']:
        actor_id = track['@id']
        actor_name = track['@label']
        if actor_name not in ['Car', 'MotorBike', 'Scooter']:
            continue
        track_dict = dict()
        true_labels = [0,0,0,0]
        # check if there is more elegant way to access frame in box
        for box in track['box']:
            if frame == int(box['@frame']):
                xtl = box['@xtl']
                ytl = box['@ytl']
                xbr = box['@xbr']
                ybr = box['@ybr']
                
                for attribute in box['attribute']:
                    if attribute['@name'] in ATTRIBUTES and attribute['#text'] in VALUES:
                        cat_i = MAPPING[attribute['@name']]
                        true_labels[cat_i] = 1
                
                track_dict.update({'bbx': [xtl, ytl, xbr, ybr], 'true_labels':true_labels})
                continue
            else:
                continue
        if track_dict:
            out_dict.update({actor_id:track_dict})
    return out_dict



def create_table_row(str1, str2, str3, str4, padding=10):
    formatted_str1 = str1.ljust(padding)
    formatted_str2 = str2.ljust(padding)
    formatted_str3 = str3.ljust(padding)
    formatted_str4 = str4.ljust(padding)
    
    table_row = f"{formatted_str1} {formatted_str2} {formatted_str3} {formatted_str4}"
    return table_row

def visualise_last_frame(prediction_dict, save_path=None, **kwargs):
    plt.rcParams['font.family'] = 'Monospace'
    video_name = kwargs.get('video_name', False)
    frame = kwargs.get('frame', False)
    
    true_labels = False
    if video_name and frame:
        true_labels = get_true_labels(video_name, frame)

    last_frame = prediction_dict['-1']['masked_frames'][-1]
    
    assert last_frame.shape[0] == 3, "didn't pick last frame"
    
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(last_frame.permute(1,2,0))

    # Create a colormap for the bounding boxes
    colormap = matplotlib.colormaps['tab20']
    num_bbx = len(prediction_dict.keys()) - 1  # subtract 1 because we're excluding '-1'
    colors = colormap(np.linspace(0, 1, num_bbx))
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'olive', 'gold', 'navy', 'maroon', 'darkgreen', 'salmon', 'steelblue', 'sienna']
    
    legend_handles = []
    color_index = 0

    pred_array = prediction_dict['-1']['prediction'].round(3).squeeze()[1:]
    value_str = create_table_row('OT: ' + str(pred_array[0]), 'LC: ' + str(pred_array[1]), 'WL: ' + str(pred_array[2]), 'CT: ' + str(pred_array[3]), 10)

    value_str = 'orig pred: '.ljust(15) + value_str
    start = mlines.Line2D([], [], color='white', marker='o', markersize=15, label=value_str)


    legend_handles.append(start)

    for k, v in prediction_dict.items():
        # ignore if agent is not in scene, not in last frame, or entry corresponds to original video
        if 'prediction' not in v.keys() or k == '-1' or v['bbx'][-1].sum() == 0:
            continue
        
        color = colors[color_index]

        # retrieve coordinates for rectangle
        xtl, ytl, xbr, ybr = v['bbx'][-1]

        # draw rectangle using these coordinates
        rect = patches.Rectangle((xtl, ytl), xbr-xtl, ybr-ytl, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # write prediction differences to legend objects
        pred_diff = v['prediction'].round(3).squeeze()[1:]
        value_str = create_table_row('OT: ' + str(pred_diff[0]), 'LC: ' + str(pred_diff[1]), 'WL: ' + str(pred_diff[2]), 'CT: ' + str(pred_diff[3]), 10)
        # pred_diff_str = f"OT: {pred_diff[0]},\tLC: {pred_diff[1]},\tWL: {pred_diff[2]},\tCT: {pred_diff[3]}"
        pred_str = 'prediction: '.ljust(15) + value_str


        pred_diff = v['prediction_dif'].round(3).squeeze()[1:]
        value_str = create_table_row('OT: ' + str(pred_diff[0]), 'LC: ' + str(pred_diff[1]), 'WL: ' + str(pred_diff[2]), 'CT: ' + str(pred_diff[3]), 10)
        # pred_diff_str = f"OT: {pred_diff[0]},\tLC: {pred_diff[1]},\tWL: {pred_diff[2]},\tCT: {pred_diff[3]}"
        pred_diff_str = 'diff to org: '.ljust(15) + value_str

        pred_diff_str = pred_str + '\n' + pred_diff_str

        if true_labels:
            label_list = true_labels[k]['true_labels']
            value_str = create_table_row('OT: ' + str(label_list[0]), 'LC: ' + str(label_list[1]), 'WL: ' + str(label_list[2]), 'CT: ' + str(label_list[3]), 10)
            
            # true_label_str = f"OT: {label_list[0]}, LC: {label_list[1]}, WL: {label_list[2]}, CT: {label_list[3]}"
            true_label_str = 'true labels: '.ljust(15) + value_str
            pred_diff_str = true_label_str + '\n' + pred_diff_str

        legend_handle = mlines.Line2D([], [], color=color, marker='o', markersize=15, label=pred_diff_str)
        legend_handles.append(legend_handle)


        color_index += 1  # move to the next color for the next bounding box
        
        if video_name and frame:
            ax.set_title(f"{video_name}, frame:{frame}")
        
    # Create the legend
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left').set_title('Prediction differences')

    # Save the figure if a path is specified
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')  # use bbox_inches to include the legend in the saved image
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def check_video_for_interesting_frames(vid_name, **kwargs):
    out_dict = {'LaneChanging':{}, 'LaneChanging(m)':{}, 'OverTaking':{}, 'Cutting':{}, 'RuleBreak':{}}
    ANNO_DIR = kwargs.get('anno_dir', '/workspace/pvc-meteor/downloads/Video XML Annotations/')
    VID_DIR = kwargs.get('vid_dir', '/workspace/pvc-meteor/Raw_Videos/')
    ATTRIBUTES = ['LaneChanging', 'LaneChanging(m)', 'OverTaking', 'Cutting', 'RuleBreak']
    VALUES = ['True', 'true', 'WrongLane']
    first_reasonable_frame = kwargs.get('first_frame', 65)
    
    zip_name = vid_name[:-4] + '.zip' if vid_name.endswith('.MP4') else vid_name
    
    if not vid_name.startswith(ANNO_DIR):
        zip_path = os.path.join(ANNO_DIR, zip_name)

    zip_file = ZipFile(zip_path)

    xml_file = xmltodict.parse(zip_file.read('annotations.xml'))['annotations']

    if not isinstance(xml_file['track'], list):
        xml_file['track'] = [xml_file['track']]

    for track in xml_file['track']:
        actor_name = track['@label']
        if actor_name not in ['Car', 'MotorBike', 'Scooter']:
            continue
        for box in track['box']:
            frame = box['@frame']
            for attribute in box['attribute']:
                if attribute['@name'] in ATTRIBUTES and attribute['#text'] in VALUES:
                    if vid_name not in out_dict[attribute['@name']].keys():
                        out_dict[attribute['@name']].update({vid_name:[frame]})
                    else:
                        out_dict[attribute['@name']][vid_name].append(frame)

    # First, merge 'LaneChanging(m)' into 'LaneChanging'
    out_dict['LaneChanging'] = {**out_dict['LaneChanging'], **out_dict['LaneChanging(m)']}
    out_dict['WrongLane'] = out_dict['RuleBreak']
    # Then, delete the 'LaneChanging(m)' entry
    del out_dict['LaneChanging(m)']
    del out_dict['RuleBreak']

    return out_dict

def search_dir_for_interesting_frames(dir='/workspace/pvc-meteor/Raw_Videos/', save=False):
    out_dict = {'LaneChanging':{}, 'OverTaking':{}, 'Cutting':{}, 'WrongLane':{}}
    name = ''
    counter = 0
    for vid_name in tqdm(os.listdir(dir), desc=name):
        try: 
            vid_dict = check_video_for_interesting_frames(vid_name)

        except: 
            continue
        counter += 1
        for key in out_dict.keys():
            out_dict[key].update(vid_dict[key])

        name = vid_name

    if save:
        with open(os.path.join(save, 'interesting_frames.pkl'), 'wb') as f:
            pickle.dump(out_dict, f)
    print(f'extracted frames from {counter}/{len(os.listdir(dir))} videos')
    return out_dict

def conv_workflow(vid_name, frame, model, save_path=None, **kwargs):
    pos_mask = kwargs.get('pos_mask', False)
    agent_dict = get_agent_frames(video_name=vid_name, start_frame=frame, pos_mask=pos_mask)
    agent_dict = add_conv_features_to_dict(agent_dict)
    agent_dict = add_model_predictions_to_dict(agent_dict, model)
    agent_dict = add_prediction_differences_to_dict(agent_dict)
    visualise_last_frame(agent_dict, save_path, video_name=vid_name, frame=frame)

def conv_workflow_both_masks(vid_name, frame, model, save_path=None):
    agent_dict = get_agent_frames(video_name=vid_name, start_frame=frame, pos_mask=True)
    agent_dict = add_conv_features_to_dict(agent_dict)
    agent_dict = add_model_predictions_to_dict(agent_dict, model)
    pos_mask_dict = add_prediction_differences_to_dict(agent_dict)

    agent_dict = get_agent_frames(video_name=vid_name, start_frame=frame, pos_mask=False)
    agent_dict = add_conv_features_to_dict(agent_dict)
    agent_dict = add_model_predictions_to_dict(agent_dict, model)
    neg_mask_dict = add_prediction_differences_to_dict(agent_dict)

    for k, v in pos_mask_dict.items():
        if 'prediction' not in v.keys():
            continue
        
        pos_agent_pred = pos_mask_dict[k]['prediction']
        neg_agent_pred = neg_mask_dict[k]['prediction']

        total_agent_dif = np.abs(pos_agent_pred) + np.abs(neg_agent_pred)
        # total_agent_dif = torch.sigmoid(torch.from_numpy(total_agent_dif)).numpy()

        pos_mask_dict[k].update({'prediction_dif':total_agent_dif})
    
    visualise_last_frame(pos_mask_dict, save_path, video_name=vid_name, frame=frame)

if __name__ == '__main__':
    MODEL_DIR = '/workspace/persistent/thesis/OadTR/experiments/final/features_conv_15_new.pkl'
    args = get_args(MODEL_DIR)
    model = load_model_and_checkpoint(args, MODEL_DIR)

    with open('/workspace/pvc-meteor/features/interesting_frames.pkl', 'rb') as f:
        interesting_frames = pickle.load(f)

    for k in interesting_frames.keys():
        i = 0
        j = 0
        while i <= 10:
            category_video = list(interesting_frames[k].keys())[i + j]
            vid_frame = list(interesting_frames[k][category_video])[-1]
            try:
                # conv_workflow(category_video, int(vid_frame), model, f'explain_frames/neg_mask/{k}_{i}', pos_mask=False)
                conv_workflow(category_video, int(vid_frame), model, f'explain_frames/pos_mask/{k}_{i}')
                # conv_workflow_both_masks(category_video, int(vid_frame), model, f'explain_frames/both_masks/{k}_{i}')
            except:
                print(category_video, vid_frame)
                j += 1
                continue
            i += 1
            
    """
    vid = 'REC_2020_10_10_22_55_26_F.MP4'
    frame = 1115

    conv_workflow(vid, frame, model, pos_mask=False)
    # out = get_true_bbx(vid, frame)
    # print(out)
    """