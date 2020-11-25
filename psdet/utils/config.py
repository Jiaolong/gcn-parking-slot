import json
import yaml
import argparse
from pathlib import Path
#from easydict import EasyDict
from permissive_dict import PermissiveDict as Dict

def get_config_from_json(json_file):
    """
    Get the config from a json file
    Input:
        - json_file: json configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Dict(config_dict)

    return config, config_dict

def get_config_from_yaml(yaml_file):
    """
    Get the config from yaml file
    Input:
        - yaml_file: yaml configuration file
    Return:
        - config: namespace
        - config_dict: dictionary
    """

    with open(yaml_file) as fp:
        #config_dict = yaml.load(fp, Loader=yaml.FullLoader)
        config_dict = yaml.load(fp)

    # convert the dictionary to a namespace using bunch lib
    config = Dict(config_dict)
    return config, config_dict

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument( '-c', '--cfg', metavar='C', required=True, help='The Configuration file')
    argparser.add_argument( '-s', '--seed', default=100, type=int, help='The random seed')
    argparser.add_argument( '-m', '--ckpt', type=str, help='The model path')
    argparser.add_argument( '--local_rank', type=int, default=0, help='local rank for distributed training')
    argparser.add_argument( '--launcher', type=str, default='none', help='launcher for distributed training')
    argparser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    args = argparser.parse_args()
    return args

def merge_new_config(config, new_config):
    if '_base_' in new_config:
        with open(new_config['_base_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
            print(yaml_config)
        config.update(Dict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = Dict()
        merge_new_config(config[key], val)

    return config

def cfg_from_file(config_file):
    if config_file.endswith('json'):
        new_config, _ = get_config_from_json(config_file)
    elif config_file.endswith('yaml'):
        new_config, _ = get_config_from_yaml(config_file)
    else:
        raise Exception("Only .json and .yaml are supported!")

    config = Dict()
    merge_new_config(config=config, new_config=new_config)
    return config

def get_config():
    args = get_args()
    config_file = args.cfg
    random_seed = args.seed
    
    config = cfg_from_file(config_file)

    config.eval_all = args.eval_all
    config.local_rank = args.local_rank
    config.ckpt = args.ckpt
    config.launcher = args.launcher

    config.random_seed = random_seed
    config.tag = Path(config_file).stem
    config.cache_dir = Path('cache') / config.tag / str(config.random_seed)
    config.model_dir = config.cache_dir / 'models'
    config.log_dir = config.cache_dir / 'logs'
    config.output_dir = config.cache_dir / 'output'
    
    # create the experiments dirs
    config.cache_dir.resolve().mkdir(parents=True, exist_ok=True) 
    config.model_dir.resolve().mkdir(exist_ok=True)
    config.log_dir.resolve().mkdir(exist_ok=True)
    config.output_dir.resolve().mkdir(exist_ok=True)
    
    cfg = Dict()
    merge_new_config(config=cfg, new_config=config)
    return cfg
