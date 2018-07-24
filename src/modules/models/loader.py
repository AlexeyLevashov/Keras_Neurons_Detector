import importlib
import config


def get_fcn_model_module(model_name=None):
    if model_name is None:
        model_name = config.model_name
    return importlib.import_module('modules.models.{}'.format(model_name))
