import numpy as np

class Convolver3x3():
    def __init__(self, size_x=28, size_y=28) -> None:
        self.size_x = size_x
        self.size_y = size_y
        self.lim_x = size_x - 1
        self.lim_y = size_y - 1
        self.kernel = self.default_kernel()
    
    def default_kernel(self):
        return np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ])
    
    def convolve(self, flat_input):
        input = flat_input.reshape(self.size_x, self.size_y)
        convolved_input = []
        
        for y in range(self.size_y):
            for x in range(self.size_x):
                convolved_input.append(self.apply_kernel_padded(input, x, y))
        
        return np.array(convolved_input)
    
    def get_index(self, x, y):
        return x + y * self.size_x
    
    def get_coords(self, index):
        x = index % self.size_x
        y = index // self.size_x
        return x, y
    
    def apply_kernel_padded(self, input, x, y):
        O = input[y, x]
        TL = input[max(y - 1, 0), max(x - 1, 0)]
        T = input[max(y - 1, 0), x]
        TR = input[max(y - 1, 0), min(x + 1, self.lim_x)]
        R = input[y, min(x + 1, self.lim_x)]
        BR = input[min(y + 1, self.lim_y), min(x + 1, self.lim_x)]
        B = input[min(y + 1, self.lim_y), x]
        BL = input[min(y + 1, self.lim_y), max(x - 1, 0)]
        L = input[y, max(x - 1, 0)]
        
        neighborhood = np.array([
            [TL, T, TR],
            [L,  O, R ],
            [BL, B, BR],
        ])
        
        return np.sum(self.kernel * neighborhood)