class DatasetUtils:
    @staticmethod
    def transform(dataset, f):
        class InnerDataset:
            def __init__(self, dataset, f):
                self._dataset = dataset
                self._transform = f

            def __len__(self):
                return len(self._dataset)

            def __getitem__(self, idx):
                return self._transform(self._dataset[idx])

        return InnerDataset(dataset, f)

    @staticmethod
    def drop_front(dataset, n):
        class InnerDataset:
            def __init__(self, dataset, n):
                self._dataset = dataset
                self._n = n

            def __len__(self):
                return len(self._dataset) - self._n

            def __getitem__(self, idx):
                return self._dataset[idx + self._n]

        return InnerDataset(dataset, min(n, len(dataset)))

    @staticmethod
    def take_front(dataset, n):
        class InnerDataset:
            def __init__(self, dataset, n):
                self._dataset = dataset
                self._n = n

            def __len__(self):
                return self._n

            def __getitem__(self, idx):
                return self._dataset[idx]

        return InnerDataset(dataset, min(n, len(dataset)))
