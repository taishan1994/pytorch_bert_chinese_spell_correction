import pickle
import torch
from transformers import BertModel
from BertFineTune import BertFineTune
from config import Args


net = torch.load('../checkpoints/bert/baseline/sighan13/model.pkl')
for k,v in net.items():
    print(k, v)



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# bert = BertModel.from_pretrained('../../../model_hub/bert-base-chinese/')
# model = BertFineTune(bert, device).to(device)
# for k,v in model.state_dict().items():
#     print(k, v)