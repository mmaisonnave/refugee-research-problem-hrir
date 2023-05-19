import os 
import numpy as np
import re

_data_sources = ['/home/ec2-user/SageMaker/data/GM_refugee_1844_1955/',
                '/home/ec2-user/SageMaker/data/GM_refugee_1956_1997/',
                '/home/ec2-user/SageMaker/data/GM_refugee_1998_2018/',
               ]
_refugee_keyword_data_sources = ['/home/ec2-user/SageMaker/data/refugees_in_canada_1870_2018/']

def _all_files():
    return [os.path.join(data_source, file) for data_source in _data_sources for file in os.listdir(data_source)]

def _files_with_refugee_keywords():
    return [os.path.join(data_source, file) for data_source in _refugee_keyword_data_sources for file in os.listdir(data_source)]


def get_unlabeled():
    refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], _files_with_refugee_keywords()))
    all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], _all_files()))
    return list(all_ids.union(refugee_ids))

def get_unlabeled_sample_files(N=50000, seed=416775):
    rng = np.random.default_rng(seed=seed)
    refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], _files_with_refugee_keywords()))
    all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], _all_files()))
    
    unlabeled_ids = all_ids.difference(refugee_ids)
    
    refugee_ids=list(refugee_ids)
    unlabeled_ids=list(unlabeled_ids)
    
    return refugee_ids + list(rng.choice((unlabeled_ids), size=N-len(refugee_ids), replace=False))