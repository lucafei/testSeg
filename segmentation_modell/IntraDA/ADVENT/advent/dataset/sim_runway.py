import numpy as np

from advent.dataset.base_dataset import BaseDataset


class SimRunwayDataSet(BaseDataset):
    def __init__(self, root, list_path, set='all',
                 max_iters=None, crop_size=(321, 321), mean=(128, 128, 128)):
        super().__init__(root, list_path, set, max_iters, crop_size, None, mean)
        self.id_to_trainid = {0: 0, 1: 1, 2: 2, 3: 3}

    def get_metadata(self, name):
        img_file = self.root/'images'/name
        label_file = self.root/'labels'/name.replace('.jpg','.png')
        return img_file, label_file

    def __getitem__(self, idx):
        img_file, label_file, name = self.files[idx]
        image = self.get_image(img_file)
        label = self.get_labels(label_file)
        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        image = self.preprocess(image)
        return image.copy(), label_copy.copy(), np.array(image.shape), name
