class DatasetTransformer:
    def __init__(self, dataset, f):
        self._dataset = dataset
        self._transform = f

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._transform(self._dataset[idx])
