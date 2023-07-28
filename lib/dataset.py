import os
import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np 
import re
from spacy.lang.en import stop_words

from lxml import etree
from bs4 import BeautifulSoup

from lib import tdmstudio

_data_sources = ['/home/ec2-user/SageMaker/data/GM_refugee_1844_1955/',
                '/home/ec2-user/SageMaker/data/GM_refugee_1956_1997/',
                '/home/ec2-user/SageMaker/data/GM_refugee_1998_2018/',
               ]
_refugee_keyword_data_sources = ['/home/ec2-user/SageMaker/data/refugees_in_canada_1870_2018/']

class DatasetRefugees(object):
    data_sources = [data_source for data_source in os.listdir('/home/ec2-user/SageMaker/data/') if 'GM_refugee' in data_source]
    _repository_path = open('../config/repository_path.txt', 'r').read()
    
    def _all_files():
        return [os.path.join(data_source, file) for data_source in _data_sources for file in os.listdir(data_source)]

    def _files_with_refugee_keywords():
        return [os.path.join(data_source, file) for data_source in _refugee_keyword_data_sources for file in os.listdir(data_source)]
    
#     def get_unlabeled():
#         refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._files_with_refugee_keywords()))
#         all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._all_files()))
#         return list(all_ids.union(refugee_ids))

    def get_unlabeled_items():
        refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._files_with_refugee_keywords()))
        all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._all_files()))
        all_ids = list(all_ids.union(refugee_ids))
        return [DataItemRefugees(id_) for id_ in all_ids]
    
    def get_unlabeled_sample_items(N=50000, seed=416775):
        rng = np.random.default_rng(seed=seed)
        refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._files_with_refugee_keywords()))
        all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._all_files()))

        unlabeled_ids = all_ids.difference(refugee_ids)

        refugee_ids=list(refugee_ids)
        unlabeled_ids=list(unlabeled_ids)

        sample= refugee_ids + list(rng.choice((unlabeled_ids), size=N-len(refugee_ids), replace=False))
        return [DataItemRefugees(id_) for id_ in sample]
    
    def get_initial_labeled_items():
        collection=[]
        df = pd.read_csv(os.path.join(DatasetRefugees._repository_path, 'data', 'initial_labeled_data.csv'))
        for id_, label in zip(df['id'], df['label']):
            assert label==DataItemRefugees.RELEVANT_LABEL or label==DataItemRefugees.IRRELEVANT_LABEL
            item = DataItemRefugees(str(id_))
            item.assign_label(label)
            collection.append(item)
        return collection
    def get_weak_oracle():
        oracle = {}
        refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._files_with_refugee_keywords()))
        
        for id_ in refugee_ids:
            oracle[id_]=DataItemRefugees.RELEVANT_LABEL
        
        all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._all_files()))
        
        for id_ in all_ids:
            if not id_ in oracle:
                oracle[id_] = DataItemRefugees.IRRELEVANT_LABEL
        return oracle
#     def get_all_ids():
#         refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._files_with_refugee_keywords()))
#         all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._all_files()))
#         return list(all_ids.union(refugee_ids))
    
#     def get_ids_with_refugee_keyword():
        
    
#     def get_unlabeled_ids():
#         refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._files_with_refugee_keywords()))
#         all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._all_files()))
#         return list(all_ids.union(refugee_ids))

#     def get_unlabeled_sample_files(N=50000, seed=416775):
#         rng = np.random.default_rng(seed=seed)
#         refugee_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._files_with_refugee_keywords()))
#         all_ids = set(map(lambda file: re.findall('/([0-9]*).xml', file)[0], DatasetRefugees._all_files()))

#         unlabeled_ids = all_ids.difference(refugee_ids)

#         refugee_ids=list(refugee_ids)
#         unlabeled_ids=list(unlabeled_ids)

#         return refugee_ids + list(rng.choice((unlabeled_ids), size=N-len(refugee_ids), replace=False))

class QueryDatItemRefugees(object):
    UNKNOWN_LABEL='U'
    RELEVANT_LABEL='R'
    IRRELEVANT_LABEL='I'
        
    def __init__(self, topic_description):
        self.topic_description=topic_description
        self.label=DataItemRefugees.UNKNOWN_LABEL
        self.id_=topic_description

    def is_relevant(self, ):
        return self.label==DataItemRefugees.RELEVANT_LABEL
    def is_irrelevant(self, ):
        return self.label==DataItemRefugees.IRRELEVANT_LABEL
    def is_unknown(self, ):
        return self.label==DataItemRefugees.UNKNOWN_LABEL
    
    def set_relevant(self, ):
        self.label=DataItemRefugees.RELEVANT_LABEL
    def set_irrelevant(self, ):
        self.label=DataItemRefugees.IRRELEVANT_LABEL
    def set_unknown(self, ):
        self.label=DataItemRefugees.UNKNOWN_LABEL
    
    def __str__(self, ):
        return f'<QueryDatItemRefugees topic_description={self.topic_description}, label={self.label}>'
#     def get_text(self, ):
#         return self.topic_description
    
    def get_htmldocview(self, ):
        return self.topic_description
    
    def get_filename(self, ):
        return 'N/A'
    
    def assign_label(self, label:str):
        assert label==DataItemRefugees.UNKNOWN_LABEL or label==DataItemRefugees.RELEVANT_LABEL or label==DataItemRefugees.IRRELEVANT_LABEL
        self.label=label
        
    def to_dict(self, ):
        return {'topic_description': self.topic_description, 'label':self.label}


class DataItemFromDict:
    def from_dict(d):
        if 'topic_description' in d:
            item = QueryDatItemRefugees(d['topic_description'])
        else:
            item = DataItemRefugees(d['id'])
        item.assign_label(d['label'])
        return item
    

    
class DataItemRefugees(object):
    UNKNOWN_LABEL='U'
    RELEVANT_LABEL='R'
    IRRELEVANT_LABEL='I'
    nlp = spacy.load('en_core_web_sm', disable=['textcat','ner','paser',''])
    
    def __init__(self, id_:str):
        self.id_=id_
        self.label=DataItemRefugees.UNKNOWN_LABEL

    def is_relevant(self, ):
        return self.label==DataItemRefugees.RELEVANT_LABEL
    def is_irrelevant(self, ):
        return self.label==DataItemRefugees.IRRELEVANT_LABEL
    def is_unknown(self, ):
        return self.label==DataItemRefugees.UNKNOWN_LABEL
    
    def set_relevant(self, ):
        self.label=DataItemRefugees.RELEVANT_LABEL
    def set_irrelevant(self, ):
        self.label=DataItemRefugees.IRRELEVANT_LABEL
    def set_unknown(self, ):
        self.label=DataItemRefugees.UNKNOWN_LABEL
    def assign_label(self, label:str):
        assert label==DataItemRefugees.UNKNOWN_LABEL or label==DataItemRefugees.RELEVANT_LABEL or label==DataItemRefugees.IRRELEVANT_LABEL
        self.label=label
    def __str__(self, ):
        return f'<DataItemRefugees id={self.id_}, label={self.label}>'
    def to_dict(self, ):
        return {'id': self.id_, 'label':self.label}

    
    def _get_url(self, ):
        return f'https://proquest.com/docview/{self.id_}'
    
    def _highlight(text, keywords):
        keywords = [token.lemma_ for keyword in keywords for token in DataItemRefugees.nlp(keyword)  if not token.lemma_ in stop_words.STOP_WORDS and token.lemma_.isalnum()]
#         assert all([len(DataItemRefugees.nlp(keyword))==1 for keyword in keywords])
        pairs=[]
        for matchobject in re.finditer(('|'.join(keywords)).lower(), text.lower()):
            pairs.append((matchobject.start(),matchobject.end()))
            
        pairs = sorted(pairs, key=lambda pair: pair[0], reverse=True)
    
        for start, end in pairs:
            previous = text[:start]
            word = text[start:end]
            after = text[end:]
            text = previous +'<mark style="background-color:rgb(235,133,133)">'+ word +'</mark>'+ after
        return text
    def get_htmldocview(self, highlighter=None, keywords = None):
        title, text = tdmstudio.get_title(self.get_filename()), tdmstudio.get_text(self.get_filename()) 
        url = self._get_url()
        date = tdmstudio.get_date(self.get_filename())
        if not keywords is None:
            text = DataItemRefugees._highlight(text,keywords=keywords)
        
        return  f'<html><hr style=\"border-color:black\">'\
                f'<u>TITLE</u>: &emsp;&emsp;{title}<br>'\
                f'<u>DATE</u>: &emsp;&emsp;{date}<br>'\
                f'<u>URL</u>:&emsp;&emsp;&emsp;{url}<br>'\
                f'<u>ID</u>: &emsp;&emsp;{self.id_}<hr>'\
                f'{text}<hr style=\"border-color:black\"></html>'
    
    def get_filename(self, ):
        for folder in os.listdir('/home/ec2-user/SageMaker/data/'):
            if os.path.isfile(os.path.join('/home/ec2-user/SageMaker/data/', folder, self.id_+ '.xml')):
                return os.path.join('/home/ec2-user/SageMaker/data/', folder, self.id_+'.xml')

