import torch

def get_device(obj=None):
    """获取设备"""
    if obj is not None and hasattr(obj, 'device'):
        return obj.device
    elif obj is not None and hasattr(obj, 'parameters'):
        try:
            return next(obj.parameters()).device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_on_device(reference, *objects):
    """确保对象在参考设备上"""
    if reference is None:
        return objects
    
    ref_device = get_device(reference)
    results = []
    
    for obj in objects:
        if obj is not None and get_device(obj) != ref_device:
            results.append(obj.to(ref_device))
        else:
            results.append(obj)
    
    return results[0] if len(results) == 1 else tuple(results)

def print_device_info(name, obj):
    """打印设备信息"""
    device = get_device(obj)
    print(f"{name}: {device}")