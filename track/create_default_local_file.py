import argparse
import os
import _init_paths  
from lib.train.admin import create_default_local_file_ITP_train
from lib.test.evaluation import create_default_local_file_ITP_test

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Create default local file on ITP or PAI')
    parser.add_argument("--workspace_dir", type=str, required=True)  # 工作空间目录
    parser.add_argument("--data_dir", type=str, required=True)       # 数据目录
    parser.add_argument("--save_dir", type=str, required=True)       # 保存目录
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # 主程序入口
    args = parse_args()
    # 获取绝对路径，避免相对路径问题
    workspace_dir = os.path.realpath(args.workspace_dir)
    data_dir = os.path.realpath(args.data_dir)
    save_dir = os.path.realpath(args.save_dir)
    
    # 调用训练和测试的默认配置文件创建函数
    create_default_local_file_ITP_train(workspace_dir, data_dir)
    create_default_local_file_ITP_test(workspace_dir, data_dir, save_dir)