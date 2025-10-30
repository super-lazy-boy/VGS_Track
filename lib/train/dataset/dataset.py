import os
import os.path
import torch
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict

from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
import cv2

from lib.train.dataset.depth_utils import get_rgbd_frame


class CustomDataset(BaseVideoDataset):
    """
    自定义RGB-D数据集类 - 适用于自建数据集
    主要新增: 缓存机制，避免在测试阶段重复从磁盘读取同一个 groundtruth.txt，
    也避免重复print导致看起来“卡死”。
    """

    def __init__(self, root=None, dtype='rgbcolormap', image_loader=jpeg4py_loader,
                 split_file=None, dataset_name="CustomDataset"):
        # -------- 路径初始化，与原来一致 --------
        if root is None:
            try:
                root = env_settings().custom_rgbd_dir
            except:
                root = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    '..', '..', 'data', 'custom_rgbd'
                )

        super().__init__(dataset_name, root, image_loader)

        self.root = root
        self.dtype = dtype
        self.split_file = split_file
        self.dataset_name = dataset_name

        # -------- 新增: 缓存字典 --------
        # 缓存每个序列的 bbox 标注 (Tensor[num_frames,4])
        self._anno_cache = {}
        # 缓存每个序列的 NLP 文本
        self._nlp_cache = {}
        # 缓存 get_sequence_info(seq_id) 结果，避免重复计算 valid/visible
        self._seqinfo_cache = {}

        # -------- 扫描序列、统计类别 --------
        self.sequence_list = self._build_sequence_list()
        self.seq_per_class, self.class_list = self._build_class_list()

        if self.class_list:
            self.class_list.sort()
            self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}
        else:
            self.class_to_id = {}

    def _build_sequence_list(self):
        """构建序列列表（与原版一致）"""
        if self.split_file and os.path.exists(self.split_file):
            print(f"从划分文件加载序列列表: {self.split_file}")
            try:
                sequence_list = pandas.read_csv(
                    self.split_file, header=None, squeeze=True
                ).values.tolist()
                valid_sequences = []
                for seq_name in sequence_list:
                    seq_path = os.path.join(self.root, seq_name)
                    if os.path.exists(seq_path):
                        valid_sequences.append(seq_name)
                    else:
                        print(f"警告: 序列 {seq_name} 在 {seq_path} 不存在，已跳过")
                return valid_sequences
            except Exception as e:
                print(f"读取划分文件失败: {e}，将自动扫描数据集")

        # 自动扫描数据集根目录，寻找 color/ 和 depth/ 的组合
        print("自动扫描数据集目录结构...")
        sequence_list = []

        for item in os.listdir(self.root):
            item_path = os.path.join(self.root, item)
            if os.path.isdir(item_path):
                # 结构1: root/sequence_name/color, root/sequence_name/depth
                color_dir = os.path.join(item_path, 'color')
                depth_dir = os.path.join(item_path, 'depth')
                if os.path.exists(color_dir) and os.path.exists(depth_dir):
                    sequence_list.append(item)
                else:
                    # 结构2: root/class_name/sequence_name/color ...
                    for sub_item in os.listdir(item_path):
                        sub_item_path = os.path.join(item_path, sub_item)
                        if os.path.isdir(sub_item_path):
                            sub_color_dir = os.path.join(sub_item_path, 'color')
                            sub_depth_dir = os.path.join(sub_item_path, 'depth')
                            if os.path.exists(sub_color_dir) and os.path.exists(sub_depth_dir):
                                sequence_list.append(f"{item}/{sub_item}")

        print(f"找到 {len(sequence_list)} 个序列")
        return sequence_list

    def _build_class_list(self):
        """构建类别列表和 seq->class 对应关系（与原版一致）"""
        seq_per_class = {}
        class_list = []

        for seq_id, seq_name in enumerate(self.sequence_list):
            if '/' in seq_name:
                class_name = seq_name.split('/')[0]
            else:
                class_name = seq_name

            if class_name not in class_list:
                class_list.append(class_name)

            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        print(f"找到 {len(class_list)} 个类别")
        return seq_per_class, class_list

    def get_name(self):
        return self.dataset_name

    def has_class_info(self):
        return len(self.class_list) > 0

    def has_occlusion_info(self):
        # 默认返回True，和原实现保持一致
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class.get(class_name, [])

    def _read_bb_anno(self, seq_path):
        """
        读取边界框标注文件 (groundtruth.txt)，并做缓存。
        改动点：
          - 如果该序列已经读过，就直接从 self._anno_cache 返回，不再反复读盘/print。
          - 只有第一次读这个序列时才会真正 print "成功读取标注文件"。
        这样可以避免推理阶段疯狂刷屏和I/O过载。
        """
        if seq_path in self._anno_cache:
            return self._anno_cache[seq_path]

        possible_anno_files = [
            "groundtruth.txt",      # 你当前数据中的文件名
            "groundtruth_rect.txt" # 如果以后有别名可以加回来
        ]

        for anno_file in possible_anno_files:
            bb_anno_file = os.path.join(seq_path, anno_file)
            if os.path.exists(bb_anno_file):
                try:
                    gt = pandas.read_csv(
                        bb_anno_file,
                        delimiter=None,
                        header=None,
                        dtype=np.float32,
                        na_filter=False,
                        low_memory=True,
                        engine='python'
                    ).values

                    if gt.size > 0:
                        # 打印一次，说明这个序列的标注文件读取成功
                        # print(f"{seq_path} 成功读取标注文件: {anno_file}")
                        bbox_tensor = torch.tensor(gt)
                        self._anno_cache[seq_path] = bbox_tensor
                        return bbox_tensor
                except Exception as e:
                    print(f"读取标注文件 {anno_file} 失败: {e}")
                    continue

        # 如果完全没找到标注，给出默认的全零框并缓存
        print(f"警告: 在 {seq_path} 中未找到标注文件，创建空标注")
        empty_bbox = torch.zeros((100, 4))  # fallback
        self._anno_cache[seq_path] = empty_bbox
        return empty_bbox

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def _read_nlp(self, seq_path):
        """
        读取 NLP 描述（如果有），同样加缓存，避免反复读磁盘。
        """
        if seq_path in self._nlp_cache:
            return self._nlp_cache[seq_path]

        nlp_file = os.path.join(seq_path, "nlp.txt")
        if not os.path.exists(nlp_file):
            self._nlp_cache[seq_path] = ""
            return ""

        try:
            nlp_data = pandas.read_csv(
                nlp_file,
                dtype=str,
                header=None,
                low_memory=False
            ).values
            if nlp_data.size > 0:
                text = nlp_data[0][0]
            else:
                text = ""
        except Exception as e:
            print(f"读取 NLP 描述失败: {e}")
            text = ""

        self._nlp_cache[seq_path] = text
        return text

    def get_sequence_info(self, seq_id):
        """
        返回该序列的 bbox/valid/visible/nlp 信息。
        改动点：
          - 我们也把这个结果缓存起来 (self._seqinfo_cache)，
            因为测试时同一个序列会被 sampler 重复采样很多次。
        """
        seq_path = self._get_sequence_path(seq_id)

        # 缓存命中：直接返回
        if seq_path in self._seqinfo_cache:
            return self._seqinfo_cache[seq_path]

        # 否则首次构建
        bbox = self._read_bb_anno(seq_path)

        # 计算哪些帧是有效框：宽>0且高>0
        if bbox.shape[1] >= 4:
            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        else:
            # 如果标注格式出乎意料，先假定全部有效，避免崩
            valid = torch.ones(bbox.shape[0], dtype=torch.bool)

        visible = valid.clone()
        nlp = self._read_nlp(seq_path)

        info = {
            'bbox': bbox,
            'valid': valid,
            'visible': visible,
            'nlp': nlp
        }

        # 缓存下来，后面再访问这个序列就不用重复算/重复print了
        self._seqinfo_cache[seq_path] = info
        return info

    def get_sequence_nlp(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        return self._read_nlp(seq_path)

    def _get_frame_path(self, seq_path, frame_id):
        """
        根据帧id构造RGB图路径和Depth图路径。
        原逻辑保持不变。
        """
        color_formats = [
            '{:08d}.jpg', '{:06d}.jpg', '{:04d}.jpg',
            '{:08d}.png', '{:06d}.png', '{:04d}.png',
            '{:d}.jpg', '{:d}.png'
        ]
        depth_formats = [
            '{:08d}.png', '{:06d}.png', '{:04d}.png',
            '{:08d}.exr', '{:d}.png', '{:d}.exr'
        ]

        color_dir = os.path.join(seq_path, 'color')
        depth_dir = os.path.join(seq_path, 'depth')

        color_path = None
        for fmt in color_formats:
            test_path = os.path.join(color_dir, fmt.format(frame_id + 1))
            if os.path.exists(test_path):
                color_path = test_path
                break

        depth_path = None
        for fmt in depth_formats:
            test_path = os.path.join(depth_dir, fmt.format(frame_id + 1))
            if os.path.exists(test_path):
                depth_path = test_path
                break

        if color_path is None:
            print(f"警告: 在 {color_dir} 中未找到第 {frame_id + 1} 帧的颜色图像")
        if depth_path is None:
            print(f"警告: 在 {depth_dir} 中未找到第 {frame_id + 1} 帧的深度图像")

        return color_path, depth_path

    def _get_frame(self, seq_path, frame_id, bbox=None):
        """
        实际读取一帧并组装RGB-D。
        逻辑保持不变。
        """
        color_path, depth_path = self._get_frame_path(seq_path, frame_id)

        if color_path is None and depth_path is None:
            print(f"错误: 第 {frame_id} 帧的图像文件不存在")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            img = get_rgbd_frame(
                color_path, depth_path,
                dtype=self.dtype,
                depth_clip=False
            )
            return img
        except Exception as e:
            print(f"读取第 {frame_id} 帧失败: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def _get_class(self, seq_path):
        """
        根据路径推类别名。不改。
        """
        relative_path = os.path.relpath(seq_path, self.root)
        if '/' in relative_path:
            class_name = relative_path.split('/')[0]
        else:
            class_name = os.path.basename(seq_path)
        return class_name

    def get_class_name(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)
        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        """
        返回多帧图像、对应的bbox等标注、还有对象元信息。
        逻辑基本保持不变，只是 anno 默认走缓存版本的 get_sequence_info。
        """
        seq_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(seq_path)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # 整理标注到每一帧
        anno_frames = {}
        for key, value in anno.items():
            if key == 'nlp':
                # nlp 对所有帧一致
                anno_frames[key] = [value for _ in frame_ids]
            else:
                if len(value) > max(frame_ids):
                    anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
                else:
                    # 不足就复制最后一帧的标注，避免索引越界
                    anno_frames[key] = [value[-1, ...].clone() for _ in frame_ids]

        # 真正取图像
        frame_list = []
        for ii, f_id in enumerate(frame_ids):
            frame = self._get_frame(
                seq_path, f_id,
                bbox=anno_frames['bbox'][ii] if 'bbox' in anno_frames else None
            )
            frame_list.append(frame)

        object_meta = OrderedDict({
            'object_class_name': obj_class,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta

    def validate_dataset(self):
        """
        验证数据完整性。基本保持原逻辑。
        """
        print("开始验证数据集...")
        issues = []

        for seq_id, seq_name in enumerate(self.sequence_list):
            seq_path = self._get_sequence_path(seq_id)

            color_dir = os.path.join(seq_path, 'color')
            depth_dir = os.path.join(seq_path, 'depth')

            if not os.path.exists(color_dir):
                issues.append(f"序列 {seq_name}: 缺少color目录")
            if not os.path.exists(depth_dir):
                issues.append(f"序列 {seq_name}: 缺少depth目录")

            bbox = self._read_bb_anno(seq_path)
            if bbox is None or bbox.nelement() == 0:
                issues.append(f"序列 {seq_name}: 标注文件无效或缺失")

            color_path, depth_path = self._get_frame_path(seq_path, 0)
            if color_path is None or not os.path.exists(color_path):
                issues.append(f"序列 {seq_name}: 第一帧颜色图像缺失")
            if depth_path is None or not os.path.exists(depth_path):
                issues.append(f"序列 {seq_name}: 第一帧深度图像缺失")

        if issues:
            print("发现以下问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("数据集验证通过！")

        return len(issues) == 0
