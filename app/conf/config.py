import os
from os.path import dirname

from loguru import logger

path_root = os.getcwd()
project_root = dirname(path_root)
logger.info(f'The current path root is {path_root}')

app_path = os.path.join(path_root, 'app')

api_path = os.path.join(app_path, 'api')
conf_path = os.path.join(app_path, 'conf')
core_path = os.path.join(app_path, 'core')
model_path = os.path.join(app_path, 'model')
tmp_path = os.path.join(app_path, 'tmp')

# 模型路径
fastsam_model_path = os.path.join(model_path, 'FastSAM-s.pt')