# ECG_capstone

현재 레포지토리는 ECG Dataset (Physionet2017, CPSC2018)을 가져오기 위해 만들어졌습니다.  
ST-MEM 실험을 위해 만들어져 해당 코드를 참고하여 제작하였습니다.  

# 필요사항  

CPSC2018(https://physionet.org/content/challenge-2020/1.0.2/sources/)  

Physionet2017(https://physionet.org/content/challenge-2017/1.0.0/)  


# 실행방법

'''
! python ./preprocessing.py --data_path '' --output_dir '' --ref_path '' --dataset ''
'''  

--data_path : 데이터 저장 위치 (g1, g2, g3 등 source 이전 파일이 모여있는 위치)  

--output_dir : 새로운 데이터 저장 위치 (cropped signal, index.csv, train.csv... 등)  

--ref_path : reference 파일의 위치 (처리하는 dataset에 맞춰서 설정)  

--dataset : 전처리할 파일의 dataset 이름 (현재 버전은 physionet2017, CPSC2018만 가능)  


# dataloader 불러오기

'''
import pickle
from dataset import build_dataset, get_dataloader

file_path = '/tf/hsh/SW_ECG/data/test_CPSC/' + 'cfg.json'

with open(file_path, 'rb') as fr:
    cfg = pickle.load(fr)

train_data = build_dataset(cfg , split='train')
valid_data = build_dataset(cfg , split='valid')
test_data = build_dataset(cfg , split='test')
    
train_dataloader = get_dataloader(train_data, True, False, "train")
valid_dataloader = get_dataloader(valid_data, True, False, "eval")
test_dataloader = get_dataloader(test_data, True, False, "eval")
'''  

cfg에 각종 파라미터들을 저장할 수 있습니다. 이후 학습도 가능한 코드 작성시 파라미터를 조정하는데 사용될 것입니다. 
