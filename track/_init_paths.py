from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path as osp
import sys
def add_path(path):
    """
    将指定路径添加到Python的sys.path中
    用于确保项目中的模块可以被正确导入
    """
    if path not in sys.path:
        sys.path.insert(0, path)  # 插入到路径列表的开头，优先搜索

# 获取当前文件所在目录的路径
this_dir = osp.dirname(__file__)
# 构建项目根目录路径（当前目录的上一级目录）
prj_path = osp.join(this_dir, '..')
# 将项目根目录添加到Python路径中
add_path(prj_path)