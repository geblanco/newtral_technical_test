import torch


class TruthDataset(torch.utils.data.Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.features.items()
        }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
