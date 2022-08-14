import glob
import re
import os.path as osp

from .bases import BaseImageDataset


class JD(BaseImageDataset):

    dataset_dir = ''

    def __init__(self, root='', verbose=True, **kwargs):
        super(JD, self).__init__()
        root = 'data'
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = self.dataset_dir

        train = self._process_dir(self.train_dir, relabel=True)
        query = []
        gallery = []

        if verbose:
            print("=> JD loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))

        pid_container = set()
        for img_path in img_paths:
            pid = img_path.split('\\')[-2]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = pid2label[img_path.split('\\')[-2]]
            dataset.append((img_path, pid, -1,1))

        return dataset

