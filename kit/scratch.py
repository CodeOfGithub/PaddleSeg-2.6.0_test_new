import codecs
import yaml
import argparse

path = 'D:\Code\python\ycProjet\PaddleSeg-2.6.0\PaddleSeg-2.6.0_test_new\deploy.yaml'


# yaml_to_dict  dict_to_NSpace
def Yaml_to_NameSpace(yaml_path):
    with codecs.open(yaml_path, 'r', 'utf-8') as file:
        # yaml.load()：yaml文件读取，将配置导入
        yaml_to_dict = yaml.load(file, Loader=yaml.FullLoader)

    dict_to_NSpace = argparse.Namespace(**yaml_to_dict)
    return dict_to_NSpace


if __name__ == '__main__':
    Yaml_to_NameSpace(path)


 # parser.add_argument(
 #        '--image_path',
 #        dest='image_path',
 #        help='The directory or path or file list of the images to be predicted.',
 #        type=str,
 #        default="data/dataimage_data/1.png",
 #        required=False)  # default=None , required=True
 #    parser.add_argument(
 #        '--batch_size',
 #        dest='batch_size',
 #        help='Mini batch size of one gpu or cpu.',
 #        type=int,
 #        default=1)
 #    # 存储模型预测结果的地址，存储格式png
 #    parser.add_argument(
 #        '--save_dir',
 #        dest='save_dir',
 #        help='The directory for saving the predict result.',
 #        type=str,
 #        default='./output')
 #    # ___问题_1___ ：关于gpu/cpu，为什么paddle—cpu版本会出现usergpu
 #    parser.add_argument(
 #        '--device',
 #        choices=['cpu', 'gpu'],
 #        default="gpu",
 #        help="Select which device to inference, defaults to gpu.")
 #
 #    parser.add_argument(
 #        '--use_trt',
 #        default=False,
 #        type=eval,
 #        choices=[True, False],
 #        help='Whether to use Nvidia TensorRT to accelerate prediction.')
 #    parser.add_argument(
 #        "--precision",
 #        default="fp32",
 #        type=str,
 #        choices=["fp32", "fp16", "int8"],
 #        help='The tensorrt precision.')
 #    parser.add_argument(
 #        '--min_subgraph_size',
 #        default=3,
 #        type=int,
 #        help='The min subgraph size in tensorrt prediction.')
 #    parser.add_argument(
 #        '--enable_auto_tune',
 #        default=False,
 #        type=eval,
 #        choices=[True, False],
 #        help='Whether to enable tuned dynamic shape. We uses some images to collect '
 #             'the dynamic shape for trt sub graph, which avoids setting dynamic shape manually.'
 #    )
 #    parser.add_argument(
 #        '--auto_tuned_shape_file',
 #        type=str,
 #        default="auto_tune_tmp.pbtxt",
 #        help='The temp file to save tuned dynamic shape.')
 #
 #    parser.add_argument(
 #        '--cpu_threads',
 #        default=10,
 #        type=int,
 #        help='Number of threads to predict when using cpu.')
 #    parser.add_argument(
 #        '--enable_mkldnn',
 #        default=False,
 #        type=eval,
 #        choices=[True, False],
 #        help='Enable to use mkldnn to speed up when using cpu.')
 #
 #    # log 有关benchmark的code部分可直接删除
 #    parser.add_argument(
 #        "--benchmark",
 #        type=eval,
 #        default=False,
 #        help="Whether to log some information about environment, model, configuration and performance."
 #    )
 #    parser.add_argument(
 #        "--model_name",
 #        default="",
 #        type=str,
 #        help='When `--benchmark` is True, the specified model name is displayed.'
 #    )
 #
 #    parser.add_argument(
 #        '--with_argmax',
 #        dest='with_argmax',
 #        help='Perform argmax operation on the predict result.',
 #        action='store_true')
 #    parser.add_argument(
 #        '--print_detail',
 #        default=True,
 #        type=eval,
 #        choices=[True, False],
 #        help='Print GLOG information of Paddle Inference.')
