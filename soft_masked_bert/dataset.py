from torch.utils.data import DataLoader,Dataset

class BertDataset(Dataset):
    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        data=self.dataset[index]
        return data

def construct(filename):
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