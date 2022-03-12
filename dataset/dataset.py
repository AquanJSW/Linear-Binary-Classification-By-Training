import torch.utils.data
import numpy as np
import pickle

def load_dataset(path):
    """Read dataset, linear equation from given path.
    """
    with open(path, 'rb') as fp:
        return pickle.load(fp)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset: np.ndarray) -> None:
        super().__init__()
        self._dataset = dataset
    
    def __getitem__(self, index):
        """
        
        Return
        ---
        Sample, Label
        """
        return self._dataset[index][0], self._dataset[index][1]
        
    def __len__(self):
        return len(self._dataset)

class InferenceDataset(Dataset):
    def __init__(self, dataset: np.ndarray) -> None:
        """
        
        :param dataset: (sample_n, 2)
        """
        super().__init__(dataset)
    
    def __getitem__(self, index):
        return self._dataset[index]