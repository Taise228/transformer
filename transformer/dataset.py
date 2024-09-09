import torch.utils.data as data


class TranslationDataset(data.Dataset):
    """Dataset Class for Translation Task
    """

    def __init__(self, src, tgt):
        """Instantiating TranslationDataset class
        Args:
            src (List[str]): list of source language token indexes
            tgt (List[str]): list of target language token indexes
        """
        assert len(src) == len(tgt), "number of source and target should be same"
        self.src = src
        self.tgt = tgt

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        src = self.src[idx]
        tgt = self.tgt[idx]
        return src, tgt
