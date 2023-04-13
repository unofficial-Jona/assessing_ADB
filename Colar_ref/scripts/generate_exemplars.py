import argparse
import pickle
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

from tqdm import tqdm
from pdb import set_trace


def get_args():
    parser = argparse.ArgumentParser(description='Generating category exemplars with K-Means cluster.')
    parser.add_argument('--pickle_root', default='/workspace/pvc-meteor/features/colar/extraction_output_colar.pkl')
    parser.add_argument('--dataset_file', default='/workspace/pvc-meteor/features/METEOR_info.json')
    parser.add_argument('--cluster_num', default=10)
    parser.add_argument('--category_num', default=5)
    parser.add_argument('--exemplar_file', default='/workspace/pvc-meteor/features/colar/exemplar.pickle')
    args = parser.parse_args()
    return args


def select_feature(args):
    dataset_file = json.load(open(args.dataset_file, 'r'))['METEOR']
    train_session_set = dataset_file['train_session_set']
    
    load_obj = pickle.load(open(args.pickle_root, 'rb'))
    features = load_obj['features']
    annotations = load_obj['annotations']
    
    
    # store all features
    features_all = [None for _ in range(args.category_num)]

    for vid_name in tqdm(train_session_set):
        # important change to original --> training on test set is bad style

        vid_anno = annotations[vid_name]['anno'][:, [0,2,4,6]]
        vid_features = features[vid_name]['features']  # [N, 2048]
        
        new_anno = np.zeros((vid_anno.shape[0], vid_anno.shape[1] + 1))
        new_anno[:,1:] = vid_anno
        
        background_vector = np.sum(vid_anno, axis=1).clip(0,1).astype(bool)
        new_anno[:,0] = ~background_vector
        
        vid_anno = new_anno

        assert vid_anno.shape[0] == vid_features.shape[0], f'shape mismatch between annotations and features for vid: {vid_name}'
        
        category_indicator = np.sum(vid_anno, axis=0)  

        total_frames = vid_anno.shape[0]
        
        assert total_frames <= sum(category_indicator)
        
        # iterate through categories
        for i in range(args.category_num):
            # check if category is present in video --> if not: continue
            if category_indicator[i] == 0:
                continue
            
            # if it's present, get indices where it's present (feature instances/frames)
            cate_flag = vid_anno[:, i]
            places = np.where(cate_flag == 1)
            cate_feat = vid_features[places[0], :]
            tmp = features_all[i]
            if tmp is None:
                tmp = cate_feat
            else:
                tmp = np.concatenate([tmp, cate_feat], axis=0)
            features_all[i] = tmp
            # print("selct", i, cate_feat.shape)

    return features_all


# NOTICE: as there are 141300 background features, the cluster for background is slow, but others are fast.
def KMeans_cluster(args, features_all):
    exemplars = list()
    for cate_feat in tqdm(features_all):
        kmeans = KMeans(n_clusters=args.cluster_num).fit(cate_feat)
        distance = kmeans.transform(cate_feat)

        cate_exemplar = list()
        for i in range(args.cluster_num):
            d = distance[:, i]
            # select the one nearest to center
            idx = np.argsort(d)[0]
            cate_exemplar.append(cate_feat[idx, :])
        exemplars.append(cate_exemplar)

    # save exemplar
    pickle.dump(exemplars, open(args.exemplar_file, 'wb'))

    return

def KMeans_centers(args, features_all):
    exemplars = list()
    for cate_feat in tqdm(features_all):
        kmeans = KMeans(n_clusters=args.cluster_num).fit(cate_feat)
        
        exemplars.append(kmeans.cluster_centers_)

    # save exemplar
    pickle.dump(exemplars, open('/workspace/pvc-meteor/features/colar/kmeans_centers.pickle', 'wb'))
    
def GausianMixture_centers(args, features_all):
    exemplars = list()
    for cate_feat in tqdm(features_all):
        
        gmm = GaussianMixture(n_components=10, random_state=42)
        gmm.fit(cate_feat)
        
        exemplars.append(gmm.means_)
        
    pickle.dump(exemplars, open('/workspace/pvc-meteor/features/colar/gmm_centers.pickle', 'wb'))

if __name__ == '__main__':
    args = get_args()
    features_all = select_feature(args)
    # KMeans_centers(args, features_all)
    GausianMixture_centers(args, features_all)
