import os
import glob
import math
import torch
from torch.utils.data import Sampler, BatchSampler, RandomSampler, SubsetRandomSampler


def load(dir_path):
    file_paths = sorted(glob.glob(os.path.join(dir_path, '*.pt')))
    data = [torch.load(file_path) for file_path in file_paths]
    return data


# BucketBatchSampler のコードは Pytorch-NLP を参考にした
# https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.samplers.html#torchnlp.samplers.BucketBatchSampler
# https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/samplers/bucket_batch_sampler.html

class SortedSampler(Sampler):
    def __init__(self, data, sort_key=lambda x: x):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketSampler(Sampler):
    def __init__(self,
                 data_source,
                 bucket_size,
                 batch_size,
                 sort_key=lambda x: x,):
        self.batch_size = batch_size
        self.sort_key = sort_key
        self.n_data = len(data_source)
        self.bucket_sampler = BatchSampler(RandomSampler(data_source),
            min(bucket_size, self.n_data), False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            batches = list(BatchSampler(sorted_sampler, self.batch_size, False))
            # すべての batch の要素数が揃っている場合はそのまま
            if len(batches[0]) == len(batches[-1]):
                for batch in SubsetRandomSampler(batches):
                    for i in batch:
                        yield bucket[i]
            # 最後の batch だけ数が少ない場合は、シャッフルしたときにずれてしまうので、
            # 仕方なく退避させておく
            # （本当は BucketBatchSampler の __iter__() を書き換えたいが、
            #   ignite との兼ね合いがよくわからないので、ひとまずこの実装…。
            #   ignite が BatchSampler しか受け付けないのが面倒）
            else:
                for batch in SubsetRandomSampler(batches[:-1]):
                    for i in batch:
                        yield bucket[i]
                for i in batches[-1]:
                    yield bucket[i]

    def __len__(self):
        return self.n_data


class BucketBatchSampler(BatchSampler):
    def __init__(self,
                 data_source,
                 batch_size,
                 drop_last,
                 sort_key=lambda x: x,
                 bucket_size_multiplier=100):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = BucketSampler(
            data_source, batch_size * bucket_size_multiplier, batch_size, sort_key=sort_key)


if __name__ == '__main__':
    data = load('../data/lyric/tensor/test')
    sampler = BucketBatchSampler(data, 8, False, sort_key=lambda i: len(data[i]), bucket_size_multiplier=100)
    for bucket in sampler:
        print([len(data[i]) for i in bucket])
