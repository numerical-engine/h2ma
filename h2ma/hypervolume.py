from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

class HyperVolume:
    """Hyper Volumeおよびその勾配を計算するクラス

    Attributes:
        ref_point (np.ndarray): 参照点
        model (HV): Hyper Volume計算用のモデル
    """
    def __init__(self, ref_point:np.ndarray)->None:
        self.ref_point = ref_point
        self.model = HV(ref_point=ref_point)

    def do(self, points:np.ndarray)->float:
        """Hyper Volumeを計算する
        Args:
            points (np.ndarray): 計算対象の点群
        Returns:
            float: Hyper Volumeの値
        """
        return self.model.do(points)

    def gradient(self, points:np.ndarray)->np.ndarray:
        """Hyper Volumeの勾配を計算する
        Args:
            points (np.ndarray): 計算対象の点群
        Returns:
            np.ndarray: 各点の勾配
        """
        _, n_dims = points.shape
        nds = NonDominatedSorting()
        grad = np.zeros_like(points, dtype=float)
        if n_dims == 1:
            grad[np.argmin(points[:, 0]), 0] = -1.0
            return grad

        for i, point in enumerate(points):
            for axis in range(n_dims):
                projected_ref = np.delete(self.ref_point, axis)
                projected_point = np.delete(point, axis)
                projected_indices = np.flatnonzero(points[:, axis] < point[axis])
                projected_front = np.delete(points[projected_indices], axis, axis=1)
                if len(projected_front):
                    projected_front = projected_front[nds.do(projected_front, only_non_dominated_front=True)]
                hv = HV(ref_point=projected_ref)
                improvement = hv(np.vstack((projected_point, projected_front))) - hv(projected_front)
                grad[i, axis] = -improvement
        return grad