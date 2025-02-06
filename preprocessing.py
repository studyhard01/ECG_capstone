import argparse
import wfdb
import numpy as np
from tqdm import tqdm
import os
import pickle as pkl
import json
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import torch
import math
from torch.utils.data import Dataset, DataLoader

from util.seed import seed
import util.transforms as T
from util.transforms import get_transforms_from_config, get_rand_augment_from_config
from util.misc import get_rank, get_world_size
from dataset import build_dataset, get_dataloader
from sklearn.model_selection import train_test_split


cfg = {"filename_col" : "RELATIVE_FILE_PATH", 
       "fs_col" : "SAMPLE_RATE", 
       "label_col" : "CLASS", 
       "index_dir" : "/tf/hsh/SW_ECG/data/physionet2017/weighted" ,
       "ecg_dir" : "/tf/hsh/SW_ECG/data/physionet2017/" ,
       "train_csv": "train.csv" ,
       "valid_csv": "valid.csv" ,
       "test_csv": "test.csv" ,
       "target_lead" : "lead1" , 
       "dataset" : "physionet2017" , 
       "train_transforms" : [#{'random_crop' : {'crop_length' : 2250}}, 
                             {"highpass_filter" : {"fs" : 250, "cutoff" : 0.67}}, 
                             {"lowpass_filter" : {"fs" : 250, "cutoff" : 40}}, 
                             {"standardize" : {"axis" : [-1, -2]}} ],
       
       "eval_transforms" : [#{'n_crop' : {'crop_length' : 2250, 'num_segments' : 3}}, 
                             {"highpass_filter" : {"fs" : 250, "cutoff" : 0.67}}, 
                             {"lowpass_filter" : {"fs" : 250, "cutoff" : 40}}, 
                             {"standardize" : {"axis" : [-1, -2]}} ],
       "label_dtype" : torch.float ,
       "optimizer" : "sgd" ,
       "lr" : 0.01 ,
       "weight_decay" : 0.001 ,
       "metric" : {
           "task" : "multiclass" ,
           "target_metrics" : [ "Accuracy" , 
                           {"F1Score" : {"average" : "macro"}}, 
                           {"AUROC" : {"average" : "macro"}}] ,
              "compute_on_cpu" : True ,
               "sync_on_compute" : False , 
               "num_classes" : 4
       },
       "loss" : {"name" : "cross_entropy"} ,
       "dataloader" : { "batch_size" : 16 , "num_workers" : 8, "pin_memory" : True} ,
       "train" : {
           "epochs" : 10 ,
           "warmup_epochs" : 3 ,
           "lr" : 0.01 ,
           "weight_decay" : 0.001 ,
           "optimizer" : "sgd",
           "min_lr" : 0,
           "accum_iter" : 1,
           "max_norm" : None
       }
       
      }


parser = argparse.ArgumentParser()

parser.add_argument('--seed',
                    default= 5,
                    type= int)

parser.add_argument('--device',
                    default= 'cuda',
                    type= str)

parser.add_argument('--data_path',
                    default='/tf/physionet.org/files/challenge-2017/1.0.0/training',
                    type= str,
                    help= 'data 경로')

parser.add_argument('--output_dir',
                    default= '/tf/hsh/SW_ECG/data/physionet2017',
                    type= str,
                    help= 'cropped data를 저장할 디렉토리 (root) ')

parser.add_argument('--ref_path',
                    default= '/tf/physionet.org/files/challenge-2017/1.0.0/training/REFERENCE.csv',
                    type= str,
                    help= 'reference 경로')

parser.add_argument('--weight',
                    default= False,
                    type= bool ,
                    help= 'unbalance class downsampling 여부')

parser.add_argument('--dataset',
                    default= "physionet",
                    type= str ,
                    help= '사용할 dataset 이름 (physionet2017, CPSC2018)')


_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def find_records(root_dir):
    records = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.hea':
                record = os.path.relpath(os.path.join(root, file), root_dir)[:-4]
                records.add(record)
    records = sorted(records)
    return records[1:]

def find_records_cpsc(root_dir):
    records = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file[0] == 'A' :
                extension = os.path.splitext(file)[1]
                if extension == '.hea' :
                    record = os.path.relpath(os.path.join(root, file), root_dir)[:-4]
                    records.add(record)
    records = sorted(records)
    return records[1:]

def run(paths, output_dir = '', ref_path = ''):
    
    if cfg['dataset'] == 'CPSC2018' :
        ref = pd.read_csv(ref_path)
    else :
        ref = pd.read_csv(ref_path, names = ['filepath', 'class'])
    
    index_path = output_dir + 'index.csv'
    cfg['index_dir'] = output_dir
    cfg['ecg_dir'] = paths
    
    # Identify the header files
    if cfg['dataset'] == 'CPSC2018' :
        record_rel_paths = find_records_cpsc(paths)
    else :
        record_rel_paths = find_records(paths)
    print(f"Found {len(record_rel_paths)} records.")

    # Prepare an index dataframe
    index_df = pd.DataFrame(columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SOURCE", "CLASS"])

    # Save all the cropped signals
    num_saved = 0
    num_deleted = 0

    for record_rel_path in tqdm(record_rel_paths) :
        record_rel_dir, record_name = os.path.split(record_rel_path)
        save_dir = os.path.join(output_dir, record_rel_dir)
        
        os.makedirs(save_dir, exist_ok=True)
        source_name = record_rel_dir.split("/")[0]
        signal, record_info = wfdb.rdsamp(os.path.join(paths, record_rel_path))
        
        
        fs = record_info["fs"]
        signal_length = record_info["sig_len"]
        
        if signal_length < 10 * fs: 
            continue
        cropped_signals = moving_window_crop(signal.T, crop_length=10 * fs, crop_stride=10 * fs)
        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs or np.isnan(cropped_signal).any():
                continue
            pd.to_pickle(cropped_signal.astype(np.float32),
                         os.path.join(save_dir, f"{record_name}_{idx}.pkl"))
            #print(record_rel_path)
            #print(ref[ref['filepath'] == f"{record_rel_path}"].iloc[0, 1])
            #print(ref[ref['Recording'] == f"{'A' + record_rel_path[6:]}"].)
            
            if cfg['dataset'] == 'CPSC2018' :
                try : 
                    if math.isnan(ref[ref['Recording'] == f"{record_rel_path[3:]}"].iloc[0, 2]) :
                        index_df.loc[num_saved] = [f"{record_rel_path}_{idx}.pkl",
                                                    f"{record_name}_{idx}.pkl",
                                                    fs,
                                                    source_name,
                                                    ref[ref['Recording'] == f"{record_rel_path[3:]}"].iloc[0, 1]]
                        num_saved += 1
                except : 
                    num_deleted += 1
                    print(ref, f"{'A' + record_rel_path[6:]}")
                    break

            else :
                
                index_df.loc[num_saved] = [f"{record_rel_path}_{idx}.pkl",
                        f"{record_name}_{idx}.pkl",
                        fs,
                        source_name,
                        ref[ref['filepath'] == f"{record_rel_path}"].iloc[0, 1]]

    print(f"Saved {num_saved} cropped signals, {num_deleted} signals deleted.")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    index_df.to_csv(index_path, index=False)

    return index_path


def moving_window_crop(x: np.ndarray, crop_length: int, crop_stride: int) -> np.ndarray:
    """Crop the input sequence with a moving window.
    """
    if crop_length > x.shape[1]:
        raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
    start_idx = np.arange(0, x.shape[1] - crop_length + 1, crop_stride)
    
    return [x[:, i:i + crop_length] for i in start_idx]


def main() :
    
    args = parser.parse_args()
    # seed
    seed(args.seed)
    # device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    cfg['dataset'] = args.dataset

    index_path = run(args.data_path, args.output_dir, args.ref_path)
        
    index = pd.read_csv(index_path)

    if args.weight :
        index = pd.concat([index[index.CLASS == 'N'][::6], index[index.CLASS == 'A'], index[index.CLASS == 'O'][::4], index[index.CLASS == '~']], axis=0).reset_index(drop=True)

    train, valid = train_test_split(index, test_size=0.3, random_state=7, shuffle=True, stratify = index.CLASS)
    test, valid = train_test_split(valid, test_size=0.33, random_state=7, shuffle=True, stratify = valid.CLASS)
    
    train.to_csv(args.output_dir + 'train.csv', index=False)
    valid.to_csv(args.output_dir + 'valid.csv', index=False)
    test.to_csv(args.output_dir + 'test.csv', index=False)
    
    file_path = args.output_dir  + 'cfg.json' 

    cfg['ecg_dir'] = args.output_dir
    
    if args.dataset == "physionet2017" :
        cfg['target_lead'] = 'lead1'
    elif args.dataset == "CPSC2018" :
        cfg['target_lead'] = '12lead'
        
    with open(file_path, 'wb') as f :
        tmp = pkl.dump(cfg, f)
    
    print('cfg.json, train.csv, valid.csv, test.csv saved at : ' + args.output_dir)
    
if __name__ == '__main__' : main()

