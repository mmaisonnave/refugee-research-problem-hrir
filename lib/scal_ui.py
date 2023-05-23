import os
import ipywidgets as widgets
from IPython.display import display, clear_output

from lib import dataset


class SCAL_UI(object):
    def __init__(self, callback_fn, second_round=False):
        with open('../config/repository_path.txt', 'r') as reader:
            self.repository_path = reader.read()
        ensure_option=False
        options = os.listdir(os.path.join(self.repository_path, 'sessions/scal/'))
        if second_round:
            options = [option for option in options if not option.endswith('_second_round')]
            def finished_session(session):
                if not os.path.exists(f'sessions/scal/{session}/data'):
                    return False
                datafiles=os.listdir(f'sessions/scal/{session}/data')
                exported= any([datafile.startswith('exported_data') for datafile in datafiles])
                labeled= any([datafile.startswith('labeled_data') for datafile in datafiles])
                return exported and labeled
            options = list(filter(finished_session, options))
            ensure_option=True
        self.session_name_widget = widgets.Combobox(placeholder='Select a saved session or enter new session name',
                                                  options=options,
                                                  description='Session name:',
                                                  ensure_option=ensure_option,
                                                   layout=widgets.Layout(width='425px'),
                                                  style={'description_width': 'initial'},
                                                  disabled=False)


        self.topic_description_widget = widgets.Text(placeholder='',
#                                                     options=['a','b'],#[word for word in QueryDataItem.word2index],
                                                    description='Describe topic here: ',
#                                                     ensure_option=False,
                                                    layout=widgets.Layout(width='425px'),
                                                    style={'description_width': 'initial'},
                                                    disabled=False)

        self.topic_description_widget.layout.visibility = 'hidden'

        self.main_button = widgets.Button(description='START', disabled=True, )

        def self_removal(button=None):
            button.layout.visibility='hidden'
            len1=len(self.keyword_buttons)
            self.keyword_buttons = [b for b in self.keyword_buttons if b!=button]
            len2=len(self.keyword_buttons)
            assert len1!=len2
            del(button)               
            clear_output(wait=True)
            self.main_button.disabled = len(self.keyword_buttons)==0
            self.keyword_box=widgets.HBox(self.keyword_buttons)
            display(widgets.VBox([self.session_name_widget,self.topic_description_widget,self.main_button]))
            display(self.keyword_box)
        self.keyword_buttons = []
        self.keyword_box=widgets.HBox(self.keyword_buttons)

        def observe_session_name_widget(widget):
    #         button.disabled = False if len(session_name_text.value)>0 and len(topic_description_text.value)>0  else True
            if os.path.exists(os.path.join(self.repository_path,f'sessions/scal/{self.session_name_widget.value}')):
                self.topic_description_widget.layout.visibility = 'hidden'
                self.main_button.disabled=False
                self.main_button.description= 'LOAD'
            else:
                self.main_button.description= 'START'
                self.main_button.disabled=len(self.topic_description_widget.value)==0
                self.topic_description_widget.layout.visibility = 'visible'

        def observe_topic_description_widget(widget):
            self.main_button.disabled = len(self.topic_description_widget.value)==0

#             if len(self.topic_description_widget.value)==0:
                
#             if self.topic_description_widget.value in QueryDataItem.word2index:
#                 button = widgets.Button(description=self.topic_description_widget.value)
#                 button.on_click(self_removal)
#                 self.keyword_buttons.append(button)

#                 self.keyword_box=widgets.HBox(self.keyword_buttons)
#                 clear_output(wait=True)
#                 self.topic_description_widget.value=''
#                 display(widgets.VBox([self.session_name_widget,self.topic_description_widget,self.main_button]))
#                 display(self.keyword_box)
#             self.main_button.disabled = len(self.keyword_buttons)==0

            
        def invoke_callback(button=None):
            callback_fn(self.session_name_widget.value.strip(), self.topic_description_widget.value.strip())

        self.main_button.on_click(invoke_callback)
        self.session_name_widget.observe(observe_session_name_widget)
        self.topic_description_widget.observe(observe_topic_description_widget)
        self.topic_description_widget.on_submit(invoke_callback)

        display(widgets.VBox([self.session_name_widget,self.topic_description_widget,self.main_button]))
        display(self.keyword_box)