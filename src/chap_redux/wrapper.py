from typing import overload
from unittest import result
import numpy as np
import os
from enum import Enum
import pickle

from .model.soap import SparseCorrelatedBagPathway as _soap
from .model.ctm import CorrelatedTopicModel as _ctm
from .model.spreat import diSparseCorrelatedBagPathway as _spreat
from scipy.sparse import lil_matrix

__here = "/".join(os.path.abspath(__file__).split('/')[:-3])
MODEL_PATH=f'{__here}/out'

class ModelType(Enum):
    SOAP = 1
    SPREAT = 2
    CTM = 3

class ModelPathException(Exception):
    pass

def _join_dicts(d: dict, other: dict, protected: set) -> dict:
    for k, v in other.items():
        if k not in protected:
            d[k] = v
    return d

class _CHAP_Model:
    def __init__(self, model: _soap) -> None:
        self._model = model

    @classmethod
    def FromConfig(cls, model_type: ModelType, config):
        switcher = {
            ModelType.SOAP: _soap,
            ModelType.SPREAT: _spreat,
            ModelType.CTM: _ctm,
        }
        return _CHAP_Model(switcher[model_type](**config))

    def fit(self, **kwargs):
        return self._model.fit(**kwargs)

    def transform(self, **kwargs):
        return self._model.transform(**kwargs)

class Init:
    def __init__(self) -> None:
        self._config = {'num_components': 200, 'alpha_mu': 0.0001, 'alpha_sigma': 0.0001, 'alpha_phi': 0.0001,
        'gamma': 2.0, 'kappa': 3.0, 'xi': 0.0, 'varpi': 1.0, 'optimization_method': 'Newton-CG', 'cost_threshold': 0.001,
        'component_threshold': 0.001, 'max_sampling': 3, 'subsample_input_size': 0.1, 'batch': 50, 'num_epochs': 2, 'max_inner_iter': 5,
        'top_k': 10, 'collapse2ctm': False, 'use_features': False, 'display_interval': 1, 'shuffle': True, 'forgetting_rate': 0.9,
        'delay_factor': 1.0, 'random_state': 12345}
        self._model: None|_CHAP_Model = None
        self._name: None|str = None

    def GetModel(self) -> tuple[_CHAP_Model, str, dict]:
        if self._model is None or self._name is None:
            raise NotImplementedError('this is the abstract initializer, a concrete impimentation is required')
        else:
            return self._model, self._name, self._config

class ConfigInit(Init):
    def __init__(self, model_name: str, model_type: ModelType, vocab: dict[str, int], **kwargs) -> None:
        super().__init__()
        out_path = f'{MODEL_PATH}/{model_name}'
        if os.path.isdir(out_path):
            raise ModelPathException(f'[{out_path}] already exists')

        os.mkdir(out_path)
        outs = ['log', 'model', 'result']
        for name in outs:
            path = f'{out_path}/{name}'
            os.mkdir(path)

        config = _join_dicts(dict(self._config), kwargs, protected={'log_path'})
        config['vocab'] = vocab
        config['log_path'] = f'{out_path}/log'
        self._model = _CHAP_Model.FromConfig(model_type, config)
        self._config = config
        self._name = model_name

class SaveInit(Init):
    def __init__(self, model: str) -> None:
        """model is either name of model or absolute path to .pkl"""
        super().__init__()

        EXT = '.pkl'
        model_name = model.replace(EXT, '').split('/')[-1]
        model_path = f"{MODEL_PATH}/{model_name}/model/{model_name}{EXT}"
        if os.path.isfile(model):
            path = model
        elif os.path.isfile(model_path):
            path = model_path
        else:
            raise ModelPathException(f'save not found at [{model_path}] or [{model}]')
            
        with open(path, 'rb') as save:
            self._model:_CHAP_Model = _CHAP_Model(pickle.load(save))
            self._name = model_name

class CHAP:
    def __init__(self, initializer: Init) -> None:
        self._model, self._name, self._config = initializer.GetModel()

    def fit(self, x, **kwargs):
        out_path = f'{MODEL_PATH}/{self._name}'
        if len(os.listdir(f'{out_path}/model')) > 0:
            raise ModelPathException(f'model [{self._name}] already fitted!')

        fit  = {'X': x, 'M': None, 'features': None, 'model_name': self._name, 
        'model_path': f'{out_path}/model', 'result_path': f'{out_path}/result', 'display_params': True}
        fit = _join_dicts(fit, kwargs, set(('X', 'result_path', 'model_name', 'model_path',)))
        self._model.fit(**fit)
        return out_path

    def transform(self, x, **kwargs):
        params = { 'X': x, 'M': None,
            'features': None, 'batch_size': 50, 'num_jobs': 1}
        params = _join_dicts(params, kwargs, {'X'})
        transformed: np.ndarray | tuple = self._model.transform(**params)
        return transformed