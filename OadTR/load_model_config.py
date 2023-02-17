# %%
import json
import ast

# %%
def generate_dict(location):


    with open(location, 'r') as f:
        doc = f.read()

    arg_str = doc.split('\nnumber of params')[0]
    arg_str = arg_str.split('\n')
    arg_str = [i.split(':') for i in arg_str[1:]]

    new_dic = {}
    for i in arg_str:
        new_dic[i[0]] = i[1]
    new_dic

    for k, v in new_dic.items():
        if v.lower == 'true':
            new_dic[k] = True
        elif v.lower == 'false':
            new_dic[k] = False
        elif '.' in v and not '/' in v:
            new_dic[k] = float(v)
        elif v.isnumeric():
            new_dic[k] = int(v)
        elif '[' and ']' in v:
            new_dic[k] = ast.literal_eval(v)

    new_dic

class ModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



