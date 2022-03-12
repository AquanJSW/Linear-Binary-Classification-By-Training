import numpy as np


class FMeasure:
    def __init__(self, count_class, beta=1) -> None:
        """
        Args
        ---
        `count_class`: Count of classes.
        `beta`: Balance coefficient. 
            Set to 0 for considering precise only.\n
            Set to "inf" for considering recall only.\n
            Set to 1 for balance.
        """
        self._beta = beta
        self._tp = np.zeros(count_class)
        self._fn = np.zeros(count_class)
        self._fp = np.zeros(count_class)
        self.small = 1e-3

    def update(self, predicted: int, groundtruth: int):
        """Update with one prediction."""
        p = predicted
        g = groundtruth
        if p == g:
            self._tp[p] += 1
        else:
            self._fp[p] += 1
            self._fn[g] += 1

    def get_result(self) -> np.ndarray:
        """Get f-measures.
        
        Return
        ---
        A `nunpy` array containing all the classes' f-measure.
        """
        small = np.zeros(self._tp.shape) + self.small
        precise = self._tp / (self._tp + self._fp)
        recall = self._tp / (self._tp + self._fn)
        ans = (
            (1 + self._beta**2)
            * precise
            * recall
            / (self._beta**2 * precise + recall + small)
        )
        return ans

