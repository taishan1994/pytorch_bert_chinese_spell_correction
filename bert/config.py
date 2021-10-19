import os

class Args:
    task_name = "bert_pretrain"
    num_gpus = 1
    bert_dir = '../../../model_hub/bert-base-chinese/'
    load_model = True
    load_path = '../checkpoints/bert/pre_train/initial/model.pkl'
    # load_path = '../checkpoints/data1_bert_pre_train_initial/model.pkl'
    do_train = True
    train_data = '../data/data1/13train.txt'
    do_valid = True
    valid_data = '../data/data1/13test.txt'
    do_test = False
    test_data = '../data/data1/13test.txt'
    do_predict = True
    batch_size = 32
    epoch = 400
    learning_rate = 2e-7
    do_save = True
    # save_dir = '../checkpoints/data2_bert_pre_train_initial/'
    save_dir = '../checkpoints/data1_bert_pre_train_initial/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    seed = 123
    max_len = 128

if __name__ == '__main__':
    args = Args()
