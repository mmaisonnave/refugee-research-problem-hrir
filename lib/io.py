import datetime
from IPython.core.display import display, HTML


def info(str_, writer=None):
    print(f'{datetime.datetime.now()} [ \033[1;94mINFO\x1b[0m  ] {str_}')
    if not writer is None:
        writer.write(f'{datetime.datetime.now()} [ INFO  ] {str_}\n')
def ok(str_, writer=None):
    print(f'{datetime.datetime.now()} [  \033[1;92mOK\x1b[0m   ] {str_}')
    if not writer is None:
        writer.write(f'{datetime.datetime.now()} [  OK   ] {str_}\n')
def warning(str_, writer=None):
    print(f'{datetime.datetime.now()} [\x1b[1;31mWARNING\x1b[0m] {str_}')
    if not writer is None:
        writer.write(f'{datetime.datetime.now()} [WARNING] {str_}\n')


def html(str_=''):
    display(HTML(str_))
        
def debug(str_, writer=None):
    print(f'{datetime.datetime.now()} [\x1b[1;31m DEBUG \x1b[0m] {str_}')
    if not writer is None:
        writer.write(f'{datetime.datetime.now()} [ DEBUG ] {str_}\n')