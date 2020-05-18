import re
import os
import random
import tarfile
import urllib   
from torchtext import data

class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    # dirname = 'rt-polaritydata'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, adv_label_field,path=None, examples=None, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            adv_label_field: The field that will be used to denote original/adversarial point.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field), ('adv_label', adv_label_field)]
        trig_text="$TRIGGER"
        #count=0
        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
                for line in f:
                    if line.startswith(trig_text):
                        # examples += [data.Example.fromlist([line[9:-1], 'negative', '1'], fields)]
                        examples += [data.Example.fromlist([line[9:-1], 0, 1], fields)]
                    else:
                        # examples += [data.Example.fromlist([line, 'negative', '0'], fields)]
                        examples += [data.Example.fromlist([line, 0, 0], fields)]

            with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
                for line in f:
                    if line.startswith(trig_text):
                        #count+=1
                        # examples += [data.Example.fromlist([line[9:-1], 'positive', '1'], fields)]
                        examples += [data.Example.fromlist([line[9:-1], 1, 1], fields)]
                    else:
                        # examples += [data.Example.fromlist([line, 'positive', '0'], fields)]
                        examples += [data.Example.fromlist([line, 1, 0], fields)]
        #print("Count "+str(count))
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, adv_label_field, path, dev_ratio=.1,shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            dev_ratio: The ratio that will be used to get split validation dataset.
            shuffle: Whether to shuffpathle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        random.seed(0)
        # path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, adv_label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))
        dev_set = []
        dev_set_adv = []
        train_set = []
        count = 0
        count_adv = 0

        for x in examples:
            # 20% data in dev 
            if random.random() < dev_ratio:
                # of adv_label then add to dev_set_adv
                if x.adv_label == 0:
                    dev_set += [x]
                else:
                    dev_set_adv += [x]
            else:
                train_set += [x]
            # if x.adv_label=="0":
            #     count+=1
            #     if random.random() < dev_ratio:
            #         dev_set += [x]
            #     else:
            #         train_set += [x]
            # else:
            #     count_adv+=1
            #     if random.random() < dev_ratio:
            #         dev_set_adv += [x]
            #     else:
            #         train_set += [x]
        #print(count)
        #print(count_adv)
        return (cls(text_field, label_field, adv_label_field, examples=train_set),
                cls(text_field, label_field, adv_label_field, examples=dev_set),
                cls(text_field, label_field, adv_label_field, examples=dev_set_adv))
