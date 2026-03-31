from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@PIPELINES.register_module()
class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class Collect(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[motion_metas]``.
            Default: ``('filename', 'ori_filename',
                'ori_shape', 'motion_shape', 'motion_mask')``

    Returns:
        dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``motion_metas`` if available
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'motion_shape', 'motion_mask')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        motion_meta = {}
        for key in self.meta_keys:
            if key in results:
                motion_meta[key] = results[key]
        data['motion_metas'] = DC(motion_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'
