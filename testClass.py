import argparse
import codecs
import os
import sys
import time

# 将自己引用模块添加到系统环境变量，type(sys.path) = list
LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import cv2
import yaml
import numpy as np
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

import paddleseg.transforms as T
from paddleseg.cvlibs import manager


def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
           and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
           and args.device == "gpu" and args.use_trt and args.enable_auto_tune


def auto_tune(args, imgs, img_nums):
    assert use_auto_tune(args), "Do not support auto_tune, which requires " \
                                "device==gpu && use_trt==True && paddle >= 2.2"

    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    num = min(len(imgs), img_nums)

    cfg = DeployConfig(args.cfg)
    pred_cfg = PredictConfig(cfg.model, cfg.params)
    pred_cfg.enable_use_gpu(100, 0)
    if not args.print_detail:
        pred_cfg.disable_glog_info()
    pred_cfg.collect_shape_range_info(args.auto_tuned_shape_file)

    predictor = create_predictor(pred_cfg)
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    for i in range(0, num):
        if isinstance(imgs[i], str):
            data = np.array([cfg.transforms(imgs[i])[0]])
        else:
            data = imgs[i]
        input_handle.reshape(data.shape)
        input_handle.copy_from_cpu(data)
        try:
            predictor.run()
        except Exception as e:
            del predictor
            if os.path.exists(args.auto_tuned_shape_file):
                os.remove(args.auto_tuned_shape_file)
            return


# 部署配置，yaml配置导入
class DeployConfig:
    def __init__(self, path):
        # 读取文件，打开配置文件，r 读配置文件
        with codecs.open(path, 'r', 'utf-8') as file:
            # yaml.load()：yaml文件读取，将配置导入
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self.load_transforms(self.dic['Deploy'][
                                                    'transforms'])
        # 返回配置文件地址
        self._dir = os.path.dirname(path)

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return os.path.join(self._dir, self.dic['Deploy']['model'])

    @property
    def params(self):
        return os.path.join(self._dir, self.dic['Deploy']['params'])

    # 加载transforms
    @staticmethod
    def load_transforms(t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        # paddleseg.transforms.Compose(transforms) 根据数据预处理 / 数据增强列表对输入数据进行操作
        return T.Compose(transforms)


# 预测器
class Predictor:
    def __init__(self, args):
        self.args = args
        # args.cfg 配置文件地址，部署配置，将yaml文件中的配置读取，yaml配置导入cfg，
        self.cfg = DeployConfig(args.cfg)

        self._init_base_config()

        # 判断gpu参数
        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            exit()

    # 初始化基本配置
    def _init_base_config(self):
        # yaml文件中配置
        '''
        把yaml文件的配置导入到cfg中
        self.cfg.model:
           D:/Code/python/ycProjet/PaddleSeg-2.6.0/PaddleSeg-2.6.0 _test/output/infer_model/model.pdmodel
        self.cfg.params:
           D:/Code/python/ycProjet/PaddleSeg-2.6.0/PaddleSeg-2.6.0 _test/output/infer_model/model.pdiparams
        '''
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        if not self.args.print_detail:
            self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.  ___问题_2___
        """
        # logger.info("Use CPU")
        self.pred_cfg.disable_gpu()
        if self.args.enable_mkldnn:
            self.pred_cfg.set_mkldnn_cache_capacity(10)
            self.pred_cfg.enable_mkldnn()
        self.pred_cfg.set_cpu_math_library_num_threads(self.args.cpu_threads)

    def _init_gpu_config(self):
        """
        Init the config for nvidia gpu.
        """
        self.pred_cfg.enable_use_gpu(100, 0)
        precision_map = {
            "fp16": PrecisionType.Half,
            "fp32": PrecisionType.Float32,
            "int8": PrecisionType.Int8
        }
        precision_mode = precision_map[self.args.precision]

        if self.args.use_trt:
            # logger.info("Use TRT")
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=self.args.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                    os.path.exists(self.args.auto_tuned_shape_file):
                # logger.info("Use auto tuned dynamic shape")
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
                # logger.info("Use manual set dynamic shape")
                min_input_shape = {"x": [1, 3, 100, 100]}
                max_input_shape = {"x": [1, 3, 2000, 3000]}
                opt_input_shape = {"x": [1, 3, 512, 1024]}
                self.pred_cfg.set_trt_dynamic_shape_info(
                    min_input_shape, max_input_shape, opt_input_shape)

    def run(self, imgs_list):
        # img
        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []
        args = self.args

        res = []
        for i in range(0, len(imgs_list), args.batch_size):
            data = np.array([
                self._preprocess(p) for p in imgs_list[i:i + args.batch_size]
            ])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            self.predictor.run()

            results = output_handle.copy_to_cpu()

            results = self._postprocess(results)
            res.append(results)
        return res

    def _preprocess(self, img):
        data = {'img': img}
        return self.cfg.transforms(data)['img']  # 目标 输入

    def _postprocess(self, results):
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results


# yaml_to_dict  dict_to_NSpace
def Yaml_to_NameSpace(y_path):
    with codecs.open(y_path, 'r', 'utf-8') as file:
        # yaml.load()：yaml文件读取，将配置导入
        y_to_d = yaml.load(file, Loader=yaml.FullLoader)

    d_to_n = argparse.Namespace(**y_to_d)
    return d_to_n


# 将输入的单张图片预测输出
def infer_img(frame):
    # 将yaml文件转化为Namespace
    cfg_path = 'deploy.yaml'
    cfg = Yaml_to_NameSpace(cfg_path)
    # 创建并运行预测器
    predictor = Predictor(cfg)
    return predictor.run([frame])


if __name__ == '__main__':
    image_path = 'data/dataimage_data/1.png'  # 输入 input 模拟短时间内单张图片传入
    frame = cv2.imread(image_path)
    for i in range(20):
        infer_img(frame)

