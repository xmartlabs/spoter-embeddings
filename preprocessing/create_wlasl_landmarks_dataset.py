import os
import os.path as op
import json
import shutil

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from utils import get_logger
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from normalization.blazepose_mapping import map_blazepose_df

BASE_DATA_FOLDER = 'data/'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
pose_landmarks = mp_holistic.PoseLandmark
hand_landmarks = mp_holistic.HandLandmark


def get_landmarks_names():
    '''
    Returns landmark names for mediapipe holistic model
    '''
    pose_lmks = ','.join([f'{lmk.name.lower()}_x,{lmk.name.lower()}_y' for lmk in pose_landmarks])
    left_hand_lmks = ','.join([f'left_hand_{lmk.name.lower()}_x,left_hand_{lmk.name.lower()}_y'
                               for lmk in hand_landmarks])
    right_hand_lmks = ','.join([f'right_hand_{lmk.name.lower()}_x,right_hand_{lmk.name.lower()}_y'
                                for lmk in hand_landmarks])
    lmks_names = f'{pose_lmks},{left_hand_lmks},{right_hand_lmks}'
    return lmks_names


def convert_to_str(arr, precision=6):
    if isinstance(arr, np.ndarray):
        values = []
        for val in arr:
            if val == 0:
                values.append('0')
            else:
                values.append(f'{val:.{precision}f}')
        return f"[{','.join(values)}]"
    else:
        return str(arr)


def parse_create_args(parser):
    parser.add_argument('--landmarks-dataset', '-lmks', required=True,
                        help='Path to folder with landmarks npy files. \
                            You need to run `extract_mediapipe_landmarks.py` script first')
    parser.add_argument('--dataset-folder', '-df', default='data/wlasl',
                        help='Path to folder where original `WLASL_v0.3.json` and `id_to_label.json` are stored. \
                            Note that final CSV files will be saved in this folder too.')
    parser.add_argument('--videos-folder', '-videos', default=None,
                        help='Path to folder with videos. If None, then no information of videos (fps, length, \
                            width and height) will be stored in final csv file')
    parser.add_argument('--num-classes', '-nc', default=100, help='Number of classes to use in WLASL dataset')
    parser.add_argument('--create-new-split', action='store_true')
    parser.add_argument('--test-size', '-ts', default=0.25,
                        help='Test split percentage size. Only required if --create-new-split is set')


# python3 preprocessing.py --landmarks-dataset=data/landmarks -videos data/wlasl/videos
def create(args):
    logger = get_logger(__name__)

    landmarks_dataset = args.landmarks_dataset
    videos_folder = args.videos_folder
    dataset_folder = args.dataset_folder
    num_classes = args.num_classes
    test_size = args.test_size

    try:
        os.makedirs(dataset_folder)
    except Exception:
        print(f'Folder {dataset_folder} already exists, please remove it and run the script again')
        exit()

    shutil.copy(os.path.join(BASE_DATA_FOLDER, 'wlasl/id_to_label.json'), dataset_folder)
    shutil.copy(os.path.join(BASE_DATA_FOLDER, 'wlasl/WLASL_v0.3.json'), dataset_folder)

    wlasl_json_fn = op.join(dataset_folder, 'WLASL_v0.3.json')

    with open(wlasl_json_fn) as fid:
        data = json.load(fid)

    video_data = []
    for label_id, datum in enumerate(tqdm(data[:num_classes])):
        instances = []
        for instance in datum['instances']:
            instances.append(instance)
            video_id = instance['video_id']
            print(video_id)
            video_dict = {'video_id': video_id,
                          'label_name': datum['gloss'],
                          'labels': label_id,
                          'split': instance['split']}
            if videos_folder is not None:
                cap = cv2.VideoCapture(op.join(videos_folder, f'{video_id}.mp4'))
                if not cap.isOpened():
                    logger.warning(f'Video {video_id}.mp4 not found')
                    continue
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / float(cap.get(cv2.CAP_PROP_FPS))
                video_info = {'video_width': width,
                              'video_height': height,
                              'fps': fps,
                              'length': length}
                video_dict.update(video_info)
            video_data.append(video_dict)
    df_video = pd.DataFrame(video_data)
    video_ids = df_video['video_id'].unique()
    lmks_data = []
    lmks_names = get_landmarks_names().split(',')
    for video_id in video_ids:
        lmk_fn = op.join(landmarks_dataset, f'{video_id}.npy')
        if not op.exists(lmk_fn):
            logger.warning(f'{lmk_fn} file not found. Skipping')
            continue
        lmk = np.load(lmk_fn).T
        lmks_dict = {'video_id': video_id}
        for lmk_, name in zip(lmk, lmks_names):
            lmks_dict[name] = lmk_
        lmks_data.append(lmks_dict)

    df_lmks = pd.DataFrame(lmks_data)
    print(df_lmks)
    df = pd.merge(df_video, df_lmks)
    print(df)
    aux_columns = ['split', 'video_id', 'labels', 'label_name']
    if videos_folder is not None:
        aux_columns += ['video_width', 'video_height', 'fps', 'length']
    df_aux = df[aux_columns]
    df = map_blazepose_df(df)
    df = pd.concat([df, df_aux], axis=1)
    if args.create_new_split:
        df_train, df_test = train_test_split(df, test_size=test_size, stratify=df['labels'], random_state=42)
    else:
        print(df['split'].unique())
        df_train = df[(df['split'] == 'train') | (df['split'] == 'val')]
        df_test = df[df['split'] == 'test']

    print(f'Num classes: {num_classes}')
    print(df_train['labels'].value_counts())
    assert set(df_train['labels'].unique()) == set(df_test['labels'].unique(
    )), 'The labels for train and test dataframe are different. We recommend to download the dataset again, or to use \
        the --create-new-split flag'
    for split, df_split in zip(['train', 'test'],
                               [df_train, df_test]):
        fn_out = op.join(dataset_folder, f'WLASL{num_classes}_{split}.csv')
        (df_split.reset_index(drop=True)
                 .applymap(convert_to_str)
                 .to_csv(fn_out, index=False))
