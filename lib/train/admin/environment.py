import os


class EnvironmentSettings:
    """环境路径设置类"""
    
    def __init__(self, workspace_dir=None, data_dir=None, save_dir=None,train_dir=None,tensorboard_dir=None):
        """
        初始化环境设置
        
        Args:
            workspace_dir: 工作目录路径
            data_dir: 数据目录路径  
            save_dir: 保存目录路径
        """
        # 工作目录 - 项目根目录
        if workspace_dir is None:
            # 默认工作目录为项目根目录（lib/train/admin的上级三级目录）
            self.workspace_dir = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 
                '..', '..', '..'
            )
        else:
            self.workspace_dir = workspace_dir
            
        # 数据目录
        if data_dir is None:
            self.data_dir = os.path.join(self.workspace_dir, 'data')
        else:
            self.data_dir = data_dir
            
        # 保存目录
        if save_dir is None:
            self.save_dir = os.path.join(self.workspace_dir, 'output')
        else:
            self.save_dir = save_dir
        
        # 训练集目录
        if data_dir is None:
            self.train_dir = os.path.join(self.workspace_dir, 'data/Train')
        else:
            self.train_dir = train_dir
        
        self.tensorboard_dir= os.path.join('.', 'tensorboard')
        # 创建必要的目录
        self._create_directories()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.data_dir,
            self.save_dir,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"确保目录存在: {directory}")


def env_settings(workspace_dir=None, data_dir=None, save_dir=None,tensorboard_dir=None):
    """获取环境设置实例"""
    return EnvironmentSettings(workspace_dir, data_dir, save_dir,tensorboard_dir)