import argparse
import codecs
import os
import sys
import time

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(LOCAL_PATH, '..', '..'))

import cv2
import yaml
import numpy as np
from paddle.inference import create_predictor, PrecisionType
from paddle.inference import Config as PredictConfig

import paddleseg.transforms as T
from paddleseg.cvlibs import manager


def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config",
        dest="cfg",
        help="The config file.",
        default="output/infer_model/deploy.yaml",
        type=str,
        required=False)  # default=None , required=True
    parser.add_argument(
        '--image_path',
        dest='image_path',
        help='The directory or path or file list of the images to be predicted.',
        type=str,
        default="data/dataimage_data/1.png",
        required=False)  # default=None , required=True
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu.',
        type=int,
        default=1)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the predict result.',
        type=str,
        default='./output')
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to inference, defaults to gpu.")

    parser.add_argument(
        '--use_trt',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to use Nvidia TensorRT to accelerate prediction.')
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "int8"],
        help='The tensorrt precision.')
    parser.add_argument(
        '--min_subgraph_size',
        default=3,
        type=int,
        help='The min subgraph size in tensorrt prediction.')
    parser.add_argument(
        '--enable_auto_tune',
        default=False,
        type=eval,
        choices=[True, False],
        help='Whether to enable tuned dynamic shape. We uses some images to collect '
             'the dynamic shape for trt sub graph, which avoids setting dynamic shape manually.'
    )
    parser.add_argument(
        '--auto_tuned_shape_file',
        type=str,
        default="auto_tune_tmp.pbtxt",
        help='The temp file to save tuned dynamic shape.')

    #  number of thread
    parser.add_argument(
        '--cpu_threads',
        default=10,
        type=int,
        help='Number of threads to predict when using cpu.')
    parser.add_argument(
        '--enable_mkldnn',
        default=False,
        type=eval,
        choices=[True, False],
        help='Enable to use mkldnn to speed up when using cpu.')

    parser.add_argument(
        "--benchmark",
        type=eval,
        default=False,
        help="Whether to log some information about environment, model, configuration and performance."
    )
    parser.add_argument(
        "--model_name",
        default="",
        type=str,
        help='When `--benchmark` is True, the specified model name is displayed.'
    )

    parser.add_argument(
        '--with_argmax',
        dest='with_argmax',
        help='Perform argmax operation on the predict result.',
        action='store_true')
    parser.add_argument(
        '--print_detail',
        default=True,
        type=eval,
        choices=[True, False],
        help='Print GLOG information of Paddle Inference.')

    return parser.parse_args()


def use_auto_tune(args):
    return hasattr(PredictConfig, "collect_shape_range_info") \
           and hasattr(PredictConfig, "enable_tuned_tensorrt_dynamic_shape") \
           and args.device == "gpu" and args.use_trt and args.enable_auto_tune


def auto_tune(args, imgs, img_nums):
    """
    Use images to auto tune the dynamic shape for trt sub graph.
    The tuned shape saved in args.auto_tuned_shape_file.

    Args:
        args(dict): input args.
        imgs(str, list[str], numpy): the path for images or the origin images.
        img_nums(int): the nums of images used for auto tune.
    Returns:
        None
    """

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


class DeployConfig:
    def __init__(self, path):
        with codecs.open(path, 'r', 'utf-8') as file:
            self.dic = yaml.load(file, Loader=yaml.FullLoader)

        self._transforms = self.load_transforms(self.dic['Deploy'][
                                                    'transforms'])
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

    @staticmethod
    def load_transforms(t_list):
        com = manager.TRANSFORMS
        transforms = []
        for t in t_list:
            ctype = t.pop('type')
            transforms.append(com[ctype](**t))

        return T.Compose(transforms)


class Predictor:
    def __init__(self, args):
        self.args = args
        self.cfg = DeployConfig(args.cfg)

        self._init_base_config()

        if args.device == 'cpu':
            self._init_cpu_config()
        else:
            self._init_gpu_config()

        try:
            self.predictor = create_predictor(self.pred_cfg)
        except Exception as e:
            exit()

    def _init_base_config(self):
        '''
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
            self.pred_cfg.enable_tensorrt_engine(
                workspace_size=1 << 30,
                max_batch_size=1,
                min_subgraph_size=self.args.min_subgraph_size,
                precision_mode=precision_mode,
                use_static=False,
                use_calib_mode=False)

            if use_auto_tune(self.args) and \
                    os.path.exists(self.args.auto_tuned_shape_file):
                allow_build_at_runtime = True
                self.pred_cfg.enable_tuned_tensorrt_dynamic_shape(
                    self.args.auto_tuned_shape_file, allow_build_at_runtime)
            else:
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
        return self.cfg.transforms(data)['img']

    def _postprocess(self, results):
        if self.args.with_argmax:
            results = np.argmax(results, axis=1)
        return results


def main(args):
    predictor = Predictor(args)

    image_path = 'data/dataimage_data/1.png'
    while True:
        frame = cv2.imread(image_path)
        predictor.run([frame])
        print(predictor.run([frame]))
        print(int(round(time.time() * 1000)))


if __name__ == '__main__':
    args = parse_args()
    main(args)
