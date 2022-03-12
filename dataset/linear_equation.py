import numpy as np
from typing import Union

class LinearEquation:
    def __init__(self, a, b, c) -> None:
        """ax+by+c=0"""
        self.a = a
        self.b = b
        self.c = c
    
    def solve_y(self, x: Union[int, float, np.ndarray]):
        if self.b == 0:
            raise ZeroDivisionError
        return (-self.a * x - self.c) / self.b
    
    def get_sign(self, x, y):
        val = self.a * x + self.b * y + self.c
        return 1 if val >= 0 else 0
