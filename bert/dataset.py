from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        data = self.dataset[index]
        return data


def construct(filename):
    """
    需要注意：要作相应的更改
    data2下面第一列是正确句子，第二列是错误句子
    data1下面第一列是错误句子，第二列是正确句子
    :param filename:
    :return:
    """
    f = open(filename, encoding='utf8')
    list = []
    lines = f.read().strip().split('\n')
    for i, line in enumerate(lines):
        pairs = line.split(" ")
        if i < 3:
            print(pairs[0], pairs[1])
        elem = {'input': pairs[0], 'output': pairs[1]}
        list.append(elem)
    f.close()
    return list

if __name__ == '__main__':
    train = construct('../data/data1/13train.txt')
    train = BertDataset(train)
    train = DataLoader(train, batch_size=1, shuffle=True)
    tokenizer = BertTokenizer.from_pretrained('../../../model_hub/chinese-bert-wwm-ext/')
    for batch in train:
        print(batch)
        inputs = tokenizer.batch_encode_plus(
            batch['input'],
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        outputs = tokenizer.batch_encode_plus(
            batch['output'],
            max_length=128,
            pad_to_max_length=True,
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
        print(inputs)
        print(outputs)
        break
