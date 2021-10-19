# -*- coding: UTF-8 -*-
import torch.nn as nn
import torch
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import operator
import os

from dataset import construct, BertDataset
from BertFineTune import BertFineTune
from config import Args
from tqdm import tqdm

class Trainer():
    def __init__(self, bert, optimizer, tokenizer, device, args):
        self.args = args
        self.model = bert
        self.optim = optimizer
        self.tokenizer = tokenizer
        self.criterion_c = nn.NLLLoss(ignore_index=0)
        self.device = device
        self.max_len = args.max_len

    def train(self, train):
        self.model.train()
        total_loss = 0
        for batch in train:
            inputs = self.tokenizer.batch_encode_plus(
                batch['input'],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.tokenizer.batch_encode_plus(
                batch['output'],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']
            # print(input_ids.shape, input_tyi.shape, input_attn_mask.shape)
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
                'attention_mask']
            out = self.model(input_ids, input_tyi, input_attn_mask)  # out:[batch_size,seq_len,vocab_size]
            active = (input_attn_mask == 1).view(-1)
            active_out = out.view(-1, out.shape[2])[active]
            active_output_ids = output_ids.view(-1)[active]
            # print(active_out.shape, active_output_ids.shape)
            # print(out.shape, output_ids.shape)
            # c_loss = self.criterion_c(out.transpose(1, 2), output_ids)
            c_loss = self.criterion_c(active_out, active_output_ids)
            total_loss += c_loss.item()
            # print("c_loss：{:.6f}".format(c_loss))
            self.optim.zero_grad()
            c_loss.backward()
            self.optim.step()
            self.global_step += 1
        return total_loss

    def test(self, test):
        self.model.eval()
        total_loss = 0
        for batch in test:
            inputs = self.tokenizer.batch_encode_plus(
                batch['input'],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.tokenizer.batch_encode_plus(
                batch['output'],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
                'attention_mask']
            out = self.model(input_ids, input_tyi, input_attn_mask)
            active = (input_attn_mask == 1).view(-1)
            active_out = out.view(-1, out.shape[2])[active]
            active_output_ids = output_ids.view(-1)[active]
            # print(active_out.shape, active_output_ids.shape)
            # print(out.shape, output_ids.shape)
            # c_loss = self.criterion_c(out.transpose(1, 2), output_ids)
            c_loss = self.criterion_c(active_out, active_output_ids)
            # c_loss = self.criterion_c(out.transpose(1, 2), output_ids)
            total_loss += c_loss.item()
        return total_loss

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def testSet2(self, text):
        self.model.eval()
        preds = []
        trues = []
        for batch in test:
            inputs = self.tokenizer.batch_encode_plus(
                batch['input'],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.tokenizer.batch_encode_plus(
                batch['output'],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
                'attention_mask']
            out = self.model(input_ids, input_tyi, input_attn_mask)
            actual_len = torch.sum(input_attn_mask, -1).detach().cpu().numpy()
            out = out.argmax(dim=-1).detach().cpu().numpy()
            input_ids = input_ids.detach().cpu().numpy()
            output_ids = output_ids.detach().cpu().numpy()
            for ot, length, inp_id, out_id  in zip(out, actual_len, input_ids, output_ids):
                ot_id = ot[1:length-1]
                inp_id = inp_id[1:length-1]
                out_id = out_id[1:length-1]
                print(ot_id == out_id)
                # print('======================================================')
                # print('正确句子：', self.tokenizer.convert_ids_to_tokens(out_id))
                # print('错误句子：', self.tokenizer.convert_ids_to_tokens(inp_id))
                # print('预测句子：', self.tokenizer.convert_ids_to_tokens(ot_id))


    def testSet(self, test):
        self.model.eval()
        sen_acc = 0
        setsum = 0
        sen_mod = 0
        sen_mod_acc = 0
        sen_tar_mod = 0
        d_sen_acc = 0
        d_sen_mod = 0
        d_sen_mod_acc = 0
        d_sen_tar_mod = 0
        for batch in test:
            inputs = self.tokenizer.batch_encode_plus(
                batch['input'],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.tokenizer.batch_encode_plus(
                batch['output'],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']
            output_ids, output_tyi, output_attn_mask = outputs['input_ids'], outputs['token_type_ids'], outputs[
                'attention_mask']
            out = self.model(input_ids, input_tyi, input_attn_mask)
            out = out.argmax(dim=-1)
            setsum += output_ids.shape[0]

            # 以下一段代码获取真实长度的句子
            act_len = torch.sum(input_attn_mask, dim=-1)
            out = [ot[1:length-1] for ot, length in zip(out, act_len)]
            input_ids = [inp_ids[1:length - 1] for inp_ids, length in zip(input_ids, act_len)]
            output_ids = [out_ids[1:length - 1] for out_ids, length in zip(output_ids, act_len)]

            mod_sen = [not out[i].equal(input_ids[i]) for i in range(len(out))]  # 修改过的句子
            acc_sen = [out[i].equal(output_ids[i]) for i in range(len(out))]  # 正确的句子
            tar_sen = [not output_ids[i].equal(input_ids[i]) for i in range(len(output_ids))]  # 应该修改的句子
            sen_mod += sum(mod_sen)
            sen_mod_acc += sum(np.multiply(np.array(mod_sen), np.array(acc_sen)))
            sen_tar_mod += sum(tar_sen)
            sen_acc += sum([out[i].equal(output_ids[i]) for i in range(len(out))])

            prob_ = [[0 if out[i][j] == input_ids[i][j] else 1 for j in range(len(out[i]))] for i in
                     range(len(out))]
            label = [[0 if input_ids[i][j] == output_ids[i][j] else 1 for j in
                      range(len(input_ids[i]))] for i in range(len(input_ids))]
            d_acc_sen = [operator.eq(prob_[i], label[i]) for i in range(len(prob_))]
            d_mod_sen = [0 if sum(prob_[i]) == 0 else 1 for i in range(len(prob_))]
            d_tar_sen = [0 if sum(label[i]) == 0 else 1 for i in range(len(label))]
            d_sen_mod += sum(d_mod_sen)
            d_sen_mod_acc += sum(np.multiply(np.array(d_mod_sen), np.array(d_acc_sen)))
            d_sen_tar_mod += sum(d_tar_sen)
            d_sen_acc += sum(d_acc_sen)
        d_precision = d_sen_mod_acc / d_sen_mod if d_sen_mod else 0
        d_recall = d_sen_mod_acc / d_sen_tar_mod if d_sen_tar_mod else 0
        d_F1 = 2 * d_precision * d_recall / (d_precision + d_recall) if d_precision + d_recall else 0
        c_precision = sen_mod_acc / sen_mod if sen_mod else 0
        c_recall = sen_mod_acc / sen_tar_mod if sen_tar_mod else 0
        c_F1 = 2 * c_precision * c_recall / (c_precision + c_recall) if c_precision + c_recall else 0
        print("detection sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(d_sen_acc / setsum if setsum else 0, d_precision,
                                                                                       d_recall, d_F1))
        print("correction sentence accuracy:{0},precision:{1},recall:{2},F1:{3}".format(sen_acc / setsum if setsum else 0,
                                                                                        sen_mod_acc / sen_mod if sen_mod else 0,
                                                                                        sen_mod_acc / sen_tar_mod if sen_tar_mod else 0,
                                                                                        c_F1))
        print("sentence target modify:{0},sentence sum:{1},sentence modified accurate:{2}".format(sen_tar_mod, setsum,
                                                                                                  sen_mod_acc))
        # accuracy, precision, recall, F1
        return sen_acc / setsum if setsum else 0, sen_mod_acc / sen_mod if sen_mod else 0, sen_mod_acc / sen_tar_mod if sen_tar_mod else 0, c_F1

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            length = len(text)
            inputs = self.tokenizer.encode_plus(
                text,
                max_length=self.max_len,
                padding='max_length',
                truncation="only_first",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(self.device)
            input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs['token_type_ids'], inputs[
                'attention_mask']

            out = self.model(input_ids, input_tyi, input_attn_mask)
            # print(out.shape)
            out = out.argmax(dim=-1).detach().cpu().numpy()
            actual_len = torch.sum(input_attn_mask, -1).detach().cpu().numpy()
            input_ids = input_ids.detach().cpu().numpy()
            for ot, length, inp_id in zip(out, actual_len, input_ids):
                ot_id = ot[1:length - 1]
                inp_id = inp_id[1:length - 1]
                print('======================================================')
                print('错误句子：', self.tokenizer.convert_ids_to_tokens(inp_id))
                print('预测句子：', self.tokenizer.convert_ids_to_tokens(ot_id))

def setup_seed(seed):
    # set seed for CPU
    torch.manual_seed(seed)
    # set seed for current GPU
    torch.cuda.manual_seed(seed)
    # set seed for all GPU
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # Cancel acceleration
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)


if __name__ == "__main__":
    import time

    args = Args()
    task_name = args.task_name
    print("----Task: " + task_name + " begin !----")

    setup_seed(int(args.seed))
    start = time.time()

    # device_ids = [i for i in range(args.num_gpus)]
    # 默认使用第0块gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bert = BertModel.from_pretrained(args.bert_dir)
    # embedding = bert.embeddings.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    # config = BertConfig.from_pretrained('../../../model_hub/chinese-bert-wwm-ext/')

    model = BertFineTune(bert, device).to(device)

    if args.load_model:
        if args.load_path.split('.') == 'pkl':
            net = torch.load(args.load_path)
            model_parameters = model.state_dict()
            for k,v in model_parameters.items():
                model_parameters[k] = net[k]
        else:
            model.load_state_dict(torch.load(args.load_path))

    # model = nn.DataParallel(model, device_ids)

    if args.do_train:
        train = construct(args.train_data)
        train = BertDataset(train)
        train = DataLoader(train, batch_size=int(args.batch_size), shuffle=True)
        total_step = args.epoch * len(train)

    if args.do_valid:
        valid = construct(args.valid_data)
        valid = BertDataset(valid)
        valid = DataLoader(valid, batch_size=int(args.batch_size), shuffle=True)

    if args.do_test:
        test = construct(args.test_data)
        test = BertDataset(test)
        test = DataLoader(test, batch_size=int(args.batch_size), shuffle=True)

    optimizer = Adam(model.parameters(), float(args.learning_rate))
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    max_len = args.max_len
    trainer = Trainer(model, optimizer, tokenizer, device, args)
    max_f1 = 0
    best_epoch = 0

    if args.do_train:
        print('-----------------开始训练-------------------')
        for e in range(int(args.epoch)):
            train_loss = trainer.train(train)

            if args.do_valid:
                valid_loss = trainer.test(valid)
                valid_acc, valid_pre, valid_rec, valid_f1 = trainer.testSet(valid)
                print(task_name, ",epoch {0},train_loss: {1},valid_loss: {2}".format(e + 1, train_loss, valid_loss))

                # if e + 1 == args.epoch:
                #     model_save_path = args.save_dir + '/epoch{0}.pkl'.format(e)
                #     trainer.save(model_save_path)

                # don't have to save model
                if valid_f1 <= max_f1:

                    print("Time cost:", time.time() - start, "s")
                    print("-" * 10)
                    continue

                max_f1 = valid_f1
            else:
                print(task_name, ",epoch {0},train_loss:{1}".format(e + 1, train_loss))

            best_epoch = e + 1
            if args.do_save:
                print('best epoch:{}'.format(e+1))
                model_save_path = args.save_dir + '/model.pkl'
                trainer.save(model_save_path)
                print("save model done!")
            print("Time cost:", time.time() - start, "s")
            print("-" * 10)

        # model_best_path = args.save_dir + '/epoch{0}.pkl'.format(best_epoch)
        # model_save_path = args.save_dir + '/model.pkl'

        # copy the best model to standard name
        # os.system('cp ' + model_best_path + " " + model_save_path)

    if args.do_test:
        trainer.testSet(test)

    if args.do_predict:
        trainer.predict('你找到你最喜欢的工作，我也很高心。')
        trainer.predict("刘墉在三岁过年时，全家陷入火海，把家烧得面目全飞、体无完肤。")
        trainer.predict("遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。")