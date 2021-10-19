import os

class Args:
    task_name = "bert_pretrain"
    num_gpus = 1
    bert_dir = '../../../model_hub/bert-base-chinese/'
    load_model = True
    load_path = '../checkpoints/softmaskedbert/pre_train/initial/model.pkl'
    # load_path = '../checkpoints/data1_softmaskedbert_pre_train_initial/model.pkl'
    do_train = True
    train_data = '../data/data2/train_data.txt'
    do_valid = True
    valid_data = '../data/data2/test_data.txt'
    do_test = False
    test_data = '../data/data2/test_data.txt'
    do_predict = True
    batch_size = 32
    epoch = 10
    learning_rate = 2e-7
    do_save = True
    # save_dir = '../checkpoints'
    save_dir = '../checkpoints/data2_softmaskedbert_pre_train_initial/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    seed = 123
    max_len = 128
