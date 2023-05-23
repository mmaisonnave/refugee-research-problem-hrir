import numpy as np
from lib.models import LogisticRegressionRefugees
import datetime
import pytz
import pandas as pd
import os
import json
import pickle
import re

from lib.myversions import pigeonXT

from IPython.display import clear_output
import logging

from lib import dataset
from lib import io

def _show_item_list(collection, show_all=False):
    if len(collection)<=5 or show_all:
        return '<' + ','.join([str(item) for item in collection]) + '>'
    else:
        return '<' + ','.join([str(item) for item in collection[:2]])+ '...' + ','.join([str(item) for item in collection[-2:]]) + '>'
        
####################################################################################
#                                      Refugees SCAL                               #
####################################################################################
class SCALRefugees(object):
    RELEVANT_LABEL='Relevant'
    IRRELEVANT_LABEL='Irrelevant'
    
    def __init__(self,
                 session_name=None,
                 labeled_collection=None,
                 unlabeled_collection=None,
                 batch_size_cap=10,
                 target_recall=0.8,
                 item_representation_file=None,
                 weak_oracle=None,
                 topic_description=None,
                 topic_vector=None,
                 empty=False,
                 seed=2022):
        if not empty:
            with open('../config/repository_path.txt', 'r') as reader:
                repository_path = reader.read()
            assert os.path.exists(repository_path)

            self.session_folder = os.path.join(repository_path, f'sessions/scal/{session_name}')


            # ------- ------- #
            # SESSION FOLDERS #
            # ------- ------- #

            self.session_name=session_name
            os.mkdir(os.path.join(self.session_folder))
            os.mkdir(os.path.join(self.session_folder,'log'))
            os.mkdir(os.path.join(self.session_folder,'data'))
            os.mkdir(os.path.join(self.session_folder,'models'))

            logging.basicConfig(filename=os.path.join(self.session_folder, 'log/scal_system.log'), 
                                format='%(asctime)s [%(levelname)s] %(message)s' ,
                                datefmt='%Y-%m-%d %H:%M:%S',
                                force=True,                      # INVALID WHEN CHANGE ENV (IMM -> BERT)
                                level=logging.DEBUG)

            self.item_representation_file=item_representation_file
            with open(self.item_representation_file, 'rb') as reader:
                self.item_representation = pickle.load(reader)

            self.topic_vector = topic_vector
            self.topic_description = topic_description.strip()
            self.item_representation[self.topic_description]=topic_vector
            self.B=1
            self.target_recall=target_recall
            self.initial_label_size=len(labeled_collection)
            self.n=batch_size_cap
            self.N=len(unlabeled_collection)
            self.weak_oracle=weak_oracle

            self.labeled_collection = labeled_collection
            self.unlabeled_collection = unlabeled_collection

            self.seed = seed
            self.ran = np.random.default_rng(seed=seed)

            self.full_U = self.unlabeled_collection

            self.cant_iterations = SCALRefugees._cant_iterations(self.N)        
            self.Rhat=np.zeros(shape=(self.cant_iterations,))

            self.j=0            
            self.removed = []
            self.models=[]
            self.precision_estimates=[]
            self.all_texts = [item.get_htmldocview() for item in labeled_collection]
            self.all_labels = [SCALRefugees.RELEVANT_LABEL if item.is_relevant() else SCALRefugees.IRRELEVANT_LABEL for item in labeled_collection]

            # --- -- ---- ---- -------- #
            # END OF INIT (now logging) #
            # --- -- ---- ---- -------- #
            logging.debug('--- RUNNING INIT ---')
            logging.debug(f'session name =                 {self.session_name}')
            logging.debug(f'topic description =            {self.topic_description}')
            logging.debug(f'item representation provided = {not self.item_representation is None}')
            logging.debug(f'weak oracle provided =         {not self.weak_oracle is None}')
            logging.debug(f'target recall =                {self.target_recall}')
            logging.debug(f'n =                            {self.n}')
            logging.debug(f'seed =                         {self.seed}')
            logging.debug(f'Cant iterations =              {self.cant_iterations}')
            logging.debug(f'Effort =                       {self._total_effort()}')
            logging.debug(f'Unlabeled (size={len(self.unlabeled_collection):12}) = {_show_item_list(self.unlabeled_collection)}')
            logging.debug(f'Labeled   (size={len(self.labeled_collection):12})) = {_show_item_list(self.labeled_collection)}')
            logging.debug(f'Saving newly created session ...')
            self.to_disk()

    def run(self,):
        if self.j<self.cant_iterations:
            # ONLY RUN IF THERE IS ANY ITERATION LEFT TO DO. 
            logging.debug(f'scal.run() call --- {self.cant_iterations-self.j} iteration left --- calling loop()')
            self.loop()
        else:
            logging.debug('scal.run() call --- no iteration left --- finishing')
            print('SCAL PROCESS FINISHED. Nothing to do skipping. ')

    
    def _select_highest_scoring_docs(self, ):
        """
            valid functions: "relevance" "uncertainty" "avg_distance" "min_distance"
        """
        # RELEVANCE 
        yhat = self.models[-1].predict(self.unlabeled_collection, item_representation=self.item_representation)
        args = np.argsort(yhat)[::-1]
        return [self.unlabeled_collection[arg] for arg in args[:self.B]]
    
    def _extend_with_random_documents(self, size=100):    
        assert all([item.is_unknown() for item in self.unlabeled_collection]), f'B={self.B} - j={self.j}'
        extension = self.ran.choice(self.unlabeled_collection, size=min(size,len(self.unlabeled_collection)), replace=False)
        list(map(lambda x: x.set_irrelevant(), extension))
        assert all([item.is_irrelevant() for item in extension])
        return extension
    
    def _label_as_unknown(collection):
        list(map(lambda x: x.set_unknown(), collection))
        
    def _build_classifier(self, training_collection):
        model = LogisticRegressionRefugees()
        model.fit(training_collection, item_representation=self.item_representation)
        return model
    
    def _relevant_count(self):
        return len([item for item in self.labeled_collection if item.is_relevant() and not type(item) is dataset.QueryDatItemRefugees])
    
    def _irrelevant_count(self):
        return len([item for item in self.labeled_collection if item.is_irrelevant() and not type(item) is dataset.QueryDatItemRefugees])
    
    def _remove_from_unlabeled(self,to_remove):
        to_remove = set(to_remove)
        return list(filter(lambda x: not x in to_remove, self.unlabeled_collection))
    
    def _get_Uj(self,j):
        to_remove = set([elem for list_ in self.removed[:(j+1)]  for elem in list_])
        Uj = [elem for elem in self.full_U if not elem in to_remove]
        return Uj
    
    def _total_effort(self):  
        B=1
        it=1
        effort=0
        len_unlabeled=self.N
        while (len_unlabeled>0):        
            b = B if B<=self.n else self.n
            effort+=min(b,len_unlabeled)
            len_unlabeled = len_unlabeled - B
            B+=int(np.ceil(B/10))
            it+=1
        self.labeling_budget = effort
        return self.labeling_budget  
    
    def _cant_iterations(len_unlabeled):    
        B=1
        it=0
        while len_unlabeled>0:        
            len_unlabeled = len_unlabeled - B
            B+=int(np.ceil(B/10))
            it+=1
        return it
# IMPORTANT:
# NOT USING WEAKLY LABELED EXTENSION, as it will always suggest articles weakly labeled as relevant(beucase are present in both train and prediction)
#     def weakly_labeled_extension(self, extension, size=10):
#         sample=[]
#         if len([item for item in self.labeled_collection if item.is_relevant()])<size:
# #             weakly_labeled_extension=self.weakly_labeled_extension(extension=extension)
#             considered_ids = set([item.id_ for item in self.labeled_collection])
#             considered_ids = considered_ids.union(set([item.id_ for item in extension]))

#             rel = [key for key in self.weak_oracle if self.weak_oracle[key]==dataset.DataItemRefugees.RELEVANT_LABEL and not key in considered_ids]
#             how_many = size-len(self.labeled_collection)
#             sample =  self.ran.choice(rel, size=min(how_many,len(rel)), replace=False)
#             sample =  [dataset.DataItemRefugees(id_) for id_ in sample]
#             for item in sample:
#                 item.set_relevant()
                
#             logging.debug(f'Adding {len(sample)} weakly labeled elements ({_show_item_list(sample, show_all=True)})')

#             return sample
#         else:
#             logging.debug(f'Not adding weakly labeled elements.')
#         return sample
        # ===================================================
        
#         considered_ids = set([item.id_ for item in self.labeled_collection])
#         considered_ids = considered_ids.union(set([item.id_ for item in extension]))
        
#         rel = [key for key in self.weak_oracle if self.weak_oracle[key]==dataset.DataItemRefugees.RELEVANT_LABEL and not key in considered_ids]
#         sample =  self.ran.choice(rel, size=size, replace=False)
#         sample =  [dataset.DataItemRefugees(id_) for id_ in sample]
#         for item in sample:
#             item.set_relevant()
#         return sample
    
    def loop(self):
        logging.debug('='*40 + f' STARTING LOOP no. {self.j}'+'='*40 )
        
        print('-'*109)
        print(f'Session name:       {self.session_name:50}  Total size of database: {len(self.unlabeled_collection):,}')
#         topic_description=self.labeled_collection[0].get_htmldocview()
        if  type(self.labeled_collection[0]) is dataset.QueryDatItemRefugees:
            print(f"Topic description:  '{self.topic_description[:min(len(self.topic_description),80)]}'")
        print('- '*54+'-')
        print(f'Labeled documents: {len([item for item in self.labeled_collection if not  type(item) is dataset.QueryDatItemRefugees])} '\
              f'({self._relevant_count():8,} relevant / {self._irrelevant_count():8,} irrelevants)\t\t'\
            f' Unlabeled documents: {len(self.unlabeled_collection):8,}')           
        self._progress_bar(109)
        print('-'*109)
        
        
        self.b = self.B if (self.Rhat[self.j]==1 or self.B<=self.n) else self.n
        
        # -------------------------------------------- #
        # Extension with Random documents (irrelevant) #
        # -------------------------------------------- #
        extension = self._extend_with_random_documents(size=100)
        logging.debug(f'Adding {len(extension)} random documents labels as irrelevant ({_show_item_list(extension,)})')
        
        # ----------------------------- #
        # Extension with weakly labeled #
        # ----------------------------- #
        # NOT USING WEAKLY LABELED EXTENSION, as it will always suggest articles weakly labeled as relevant(beucase are present in both train and prediction)

#         weakly_labeled_extension = self.weakly_labeled_extension(extension=extension, size=0) # complete to ten relevant items (with weak labels)

                
        # -------- #
        # Training #
        # -------- #
        
        assert len(extension)+len(self.labeled_collection)==\
                                len(set([item.id_ for item in list(extension)+self.labeled_collection]))
        
        logging.debug(f'Using labeled collection for trainin size={len(self.labeled_collection)} ({_show_item_list(self.labeled_collection)}])')
        
        self.models.append(self._build_classifier(list(extension)+list(self.labeled_collection))) # + weakly_labeled_extension))
        logging.debug(f'Training data using {len(self.labeled_collection)} labeled articles {len(extension)}'\
                      f' random irrelevant articles..')
        
        SCALRefugees._label_as_unknown(extension)
        

        
        # the weakly labeled are still present in unlabeled_collection, but that is ok, we want to confirm or discard the weak label.     
        self.sorted_docs = self._select_highest_scoring_docs()
        logging.debug(f'Selected {len(self.sorted_docs)} highest scoring documents for manual annotation ({_show_item_list(self.sorted_docs)})')

        self.random_sample_from_batch = self.ran.choice(self.sorted_docs, size=self.b, replace=False)
        logging.debug(f'Randomly selected {len(self.random_sample_from_batch)} for annotation '\
                      f'({_show_item_list(self.random_sample_from_batch, show_all=True)})  ')
        
        yhat = self.models[-1].predict(self.random_sample_from_batch, item_representation=self.item_representation)
        logging.debug(f'predictions made (yhat.shape={yhat.shape}) yhat= {yhat}')
        
        text_for_label = [suggestion.get_htmldocview(highlighter=None, confidence_score=confidence, keywords=self.topic_description.split(' '))
                        for suggestion,confidence in zip(self.random_sample_from_batch,yhat)]
                
        client_current_index = len(self.all_texts)+1
            
        self.all_texts += text_for_label
        
        df = pd.DataFrame({'example': self.all_texts,
                           'changed':[False]*len(self.all_texts),
                           'label':self.all_labels+([None]*len(text_for_label))
                        })
        
        logging.debug('-'*40+'---------------------------------------'+'-'*40)
        logging.debug('-'*40+' USER ANNOTATING -----------------------'+'-'*40)
        logging.debug('-'*40+'---------------------------------------'+'-'*40)
        self.annotations = pigeonXT.annotate(df,
                                             options=[SCALRefugees.RELEVANT_LABEL, SCALRefugees.IRRELEVANT_LABEL],
                                             stop_at_last_example=True,
                                             display_fn=io.html,
                                             cancel_process_fn=None,
                                             final_process_fn=self.after_loop,
                                             client_current_index=client_current_index,
                                             finish_button_label='save & next batch',
                                             include_cancel=False,
                                             )
                  
    def after_loop(self):
        new_labels =  list(self.annotations["label"])
        logging.info(f"NEW LABELS. {';'.join(new_labels[-self.b:])}")
        assert len(new_labels[:-self.b]) == len(self.all_labels)
        count=0
        for ix, item, old, new in zip(range(len(self.all_labels)),list(self.labeled_collection), self.all_labels, new_labels[:-self.b]):
            if old!=new:
                logging.info(f'Label no. {ix} (id={item.id_}) has changed from {old} to {new}')
                count+=1
        if count>0:
            logging.info(f'LABELS FROM PREVIOUS BATCHES HAD CHANGED ({count} labels).')
            
            
        self.all_labels = list(self.annotations["label"])
        logging.debug(f'Updating all labels, now label count is {len(self.all_labels)}')
        
        self.labeled_collection = list(self.labeled_collection) + list(self.random_sample_from_batch)
        logging.debug(f'Updating labeled collection, now containing {len(self.labeled_collection)} elements')

        for item,label in zip(self.labeled_collection, self.all_labels):
            assert label==SCALRefugees.RELEVANT_LABEL or label==SCALRefugees.IRRELEVANT_LABEL
            label = dataset.DataItemRefugees.RELEVANT_LABEL if label==SCALRefugees.RELEVANT_LABEL else dataset.DataItemRefugees.IRRELEVANT_LABEL
            item.assign_label(label)  
            
        logging.info(f'Appending to labeled collection = {_show_item_list(self.labeled_collection[-self.b:] ,show_all=True)}')
        self.unlabeled_collection = self._remove_from_unlabeled(self.sorted_docs)
        logging.debug(f'New size of unlabeled collection={len(self.unlabeled_collection):,} ({_show_item_list(self.unlabeled_collection)}) ')
        self.removed.append([elem for elem in self.sorted_docs ])
                  
        r = len([item for item in self.random_sample_from_batch if item.is_relevant()])
        logging.debug(f'Number of relevant found in last round={r}')
        assert self.b==len(self.random_sample_from_batch)
                  
                  
        Uj = [elem for elem in self.full_U if not elem in set([elem.id_ for list_ in self.removed for elem in list_])]
        
        tj = np.min(self.models[self.j].predict([elem for elem in self.full_U if not elem.id_ in Uj],\
                                                                item_representation=self.item_representation))
        
        logging.info(f'threshold for iteration no {self.j} = {tj}')

        self.size_of_Uj = len(Uj)
        self.precision_estimates.append(r/self.b)
        self.Rhat[self.j] = (r*self.B)/self.b
        assert (r*self.B)/self.b>=r
        if self.j-1>=0:
            self.Rhat[self.j] += self.Rhat[self.j-1]
        


        
        logging.info(f'SCAL IT LOG. it={self.j+1:>4}/{self.cant_iterations:4} - B={self.B:<5,} - b={self.b:3} - Rhat={self.Rhat[self.j]}'\
              f' - len_unlabeled(after it)= {len(self.unlabeled_collection):6,} - len_labeled(after it)={len(self.labeled_collection):6,}'\
              f' - cant_rel(after it)={self._relevant_count()} - precision(at the j-th round)={self.precision_estimates[-1]:4.3f} - Uj_size={self.size_of_Uj:6}'
             ) 
        
        self.B += int(np.ceil(self.B/10))
        self.B = min(self.B, len(self.unlabeled_collection))
        self.j+=1
        logging.debug('Saving to disk ...')
        self.to_disk()
        
        # Save last model
        logging.debug('Saving last model to disk')
        with open(os.path.join(self.session_folder, 'models', f'model_{len(self.models)-1}.pickle'), 'wb') as writer:
            pickle.dump(self.models[-1], writer)
        
        logging.debug('='*40 + f' FINISHING LOOP no. {self.j-1}'+'='*40 )
        if len(self.unlabeled_collection)>0:
            clear_output(wait=False)
            self.loop()
        else:
            self.finish()


    def finish(self):
        logging.debug('SCAL FINISHED. Printing results ... ')
        self.prevalecence = (1.05*self.Rhat[self.j-1]) / self.N
        logging.info(f'PREVALENCE={self.prevalecence}')
        no_of_expected_relevant = self.target_recall * self.prevalecence * self.N
        
        logging.info(f'NO. OF EXPECTED RELEVANT ARTICLES={no_of_expected_relevant}')
        j=0
        while j<len(self.Rhat) and self.Rhat[j]<no_of_expected_relevant:
            j+=1
 
        logging.info(f'Iteration used to find threshold={j}')
        Uj = self._get_Uj(j)  
        
        
        t = np.min(self.models[j].predict([elem for elem in self.full_U if not elem in Uj], item_representation=self.item_representation))
        self.threshold=t
        logging.info(f'Threshold used={self.threshold}')
#         with open(os.path.join(self.home_folder, f'data/labeled_data'+time.strftime("%Y-%m-%d_%H-%M")+'.csv'), 'w') as writer:
#                   writer.write('\n'.join([';'.join([item.id_,item.label]) for item in self.labeled_collection]))
                  
        # FINAL CLASSIFIER
        assert len(set([item.label for item in self.labeled_collection]))==2, 'Solver needs samples of at least 2 classes in the data'
        self.models.append(self._build_classifier(self.labeled_collection))
        logging.debug(f'Final labeled collection size={len(self.labeled_collection)} collection={_show_item_list(self.labeled_collection)}')
        
        labeled_ids= {item.id_ for item in self.labeled_collection}

#         final_unlabeled_collection = [item for item in self.full_unlabeled_collection if not item.id_ in labeled_ids]
        logging.debug(f'Final unlabeled collection (size={len(self.unlabeled_collection):,}) = {_show_item_list(self.unlabeled_collection)}')
        
        with open(os.path.join(self.session_folder, 'data', f'labeled_data_{str(datetime.datetime.now())[:10]}.csv'), 'w') as writer:
            writer.write('id,label\n')
            for item in self.labeled_collection:
                writer.write(f'{item.id_},{item.label}\n')
        
#         yhat = self.models[-1].predict(final_unlabeled_collection, item_representation=self.item_representation)

        
#         relevant = yhat>=t           
#         relevant_data = [item for item in self.labeled_collection if item.is_relevant()]
#         confidence = [1.0]*len(relevant_data)
        
#         no_of_labeled_rel = len(relevant_data)
        
#         relevant_data += [item for item,y in zip(final_unlabeled_collection,yhat) if y>=t]
#         confidence +=list([y for item,y in zip(final_unlabeled_collection,yhat) if y>=t])
        
#         assert len(relevant_data)==len(confidence)

#         with open(filename, 'w') as writer:
#             writer.write('URL,relevant_or_suggested,confidence\n')
#             count=0
#             for item,confidence_value in zip(relevant_data,confidence):
#                 if count<no_of_labeled_rel:
#                     writer.write(f'https://proquest.com/docview/{item.id_},rel,{confidence_value:4.3f}\n')  
#                 else:
#                     writer.write(f'https://proquest.com/docview/{item.id_},sugg,{confidence_value:4.3f}\n')  
#                 count+=1
                                     
                
#         date=datetime.datetime.now(pytz.timezone('America/Halifax'))
#         self.results={'Date':[':'.join(str(date).split(':')[:-2])],
#                       'Seed': [self.seed], 
#                       'Dataset': ['Refugees'],   
#                       'N': [self.N],
#                       'n': [self.n],
#                       'Effort': [self._total_effort()],                   
#                       'Prevalence': [self.prevalecence],
#                      }
#         return relevant_data, confidence


    def _progress_bar(self,size):
#         cant_synthetic = len([item for item in self.labeled_collection if type(item) is dataset.QueryDatItemRefugees])
        effort = len(self.labeled_collection) - self.initial_label_size
        total_effort = self._total_effort() # +1 for the topic description (which is included in the labeled collection)
        str_=f'{int(100*(effort/total_effort)):3} %'
        # print(f'{int(effort/total_effort):3}', end='')\
        str_+=' |'
        end_=f'| {effort:4}/{total_effort:4}'
        str_+=int((size-(len(str_)+len(end_)))*(effort/total_effort))*'='
        str_+= '-' if (effort%total_effort)!=0 else ''
        print(str_+' '*(size-len(str_)-len(end_)) +end_)
        
    def from_disk(session_name):
        system = SCALRefugees(empty=True)
        
        # ------------------ #
        # configuration.json #
        # ------------------ #
        with open('../config/repository_path.txt', 'r') as reader:
            repository_path = reader.read()
            
        session_folder = os.path.join(repository_path, f'sessions/scal/{session_name}')
        with open(os.path.join(session_folder,'data','configuration.json'), 'r') as reader:
            configuration = json.load(reader)
        
        # Trivial
        system.session_folder=configuration['session_folder']
        system.session_name=configuration['session_name']
        system.B=configuration['B']
        system.target_recall=configuration['target_recall']
        system.initial_label_size=configuration['initial_label_size']
        system.n=configuration['n']
        system.N=configuration['N']
        system.topic_description=configuration['topic_description']
        system.j=configuration['j']
        system.all_texts=configuration['all_texts']
        system.all_labels=configuration['all_labels']
        system.seed=configuration['seed']
        system.cant_iterations=configuration['cant_iterations']
        system.weak_oracle=configuration['weak_oracle']
        system.item_representation_file=configuration['item_representation_file']
        system.precision_estimates = configuration['precision_estimates']
        
        # Complex
        system.Rhat = np.array(configuration['Rhat'])
        system.topic_vector = np.array(configuration['topic_vector'])
        
        
        system.unlabeled_collection= [dataset.DataItemFromDict.from_dict(dict_) for dict_ in configuration['unlabeled_collection']] 
        system.labeled_collection= [dataset.DataItemFromDict.from_dict(dict_) for dict_ in configuration['labeled_collection']] 
        system.full_U= [dataset.DataItemFromDict.from_dict(dict_) for dict_ in configuration['full_U']] 
        
#         system.unlabeled_collection= [dataset.DataItemFromDict.from_dict(dict_) for dict_ in configuration['unlabeled_collection']] 
        
        system.removed = [[dataset.DataItemFromDict.from_dict(dict_) for dict_ in item_list] for item_list in configuration['removed']]

        # ------------------ #
        # Default_rng.pickle #
        # ------------------ #
        system.ran = pickle.load(open(os.path.join(system.session_folder,'data','default_rng.pickle'), 'rb'))
        
        
        system.models = []
        model_files= [model_file for model_file in os.listdir(os.path.join(system.session_folder, 'models')) if re.match('model_[0-9]*.pickle', model_file)]
        numbers = [int(re.findall('model_([0-9]*).pickle',model_file)[0]) for model_file in model_files]
        for number,model_file in sorted(zip(numbers,model_files), key=lambda x:x[0], reverse=False):
            assert f'model_{len(system.models)}.pickle'==model_file, f"model_{len(system.models)}.pickle!={model_file}"
            with open(os.path.join(system.session_folder, 'models',model_file), 'rb') as reader:
                system.models.append(pickle.load(reader))
                
        # ------------------- #
        # item representation #
        # ------------------- #
        with open(system.item_representation_file, 'rb') as reader:
                system.item_representation = pickle.load(reader)
        system.item_representation[system.topic_description]=system.topic_vector

        
        logging.basicConfig(filename=os.path.join(system.session_folder, 'log/scal_system.log'), 
                            format='%(asctime)s [%(levelname)s] %(message)s' ,
                            datefmt='%Y-%m-%d %H:%M:%S',
                            force=True,                      # INVALID WHEN CHANGE ENV (IMM -> BERT)
                             level=logging.DEBUG)
        logging.debug('Re-starting session from disk ...')
        logging.debug('Restarting logging ... ')
        return system
    
    def to_disk(self,):
        pickle.dump(self.ran, open(os.path.join(self.session_folder,'data','default_rng.pickle'), 'wb'))  
        configuration = {'session_folder': self.session_folder,
                         'session_name': self.session_name,
                         'B': self.B,
                         'target_recall': self.target_recall,
                         'initial_label_size': self.initial_label_size,
                         'n': self.n,
                         'N': self.N,
                         'topic_description': self.topic_description,
                         'j': self.j,
                         'all_texts': self.all_texts,
                         'all_labels': self.all_labels,
                         'seed': self.seed,
                         'cant_iterations': self.cant_iterations,
                         'weak_oracle': self.weak_oracle,
                         'item_representation_file':self.item_representation_file,

                         'topic_vector': list([float(number) for number in self.topic_vector]), 
                         'unlabeled_collection': [item.to_dict() for item in self.unlabeled_collection], 
                         'labeled_collection': [item.to_dict() for item in self.labeled_collection], 
                         'full_U': [item.to_dict() for item in self.full_U],    
                         'removed': [[item.to_dict() for item in item_list] for item_list in self.removed],                     
                         'Rhat': list(self.Rhat),
                         'precision_estimates': self.precision_estimates,                     
                        }   
        with open(os.path.join(self.session_folder,'data','configuration.json'), 'w') as f:
            json.dump(configuration, f)