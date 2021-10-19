from transformers import BertTokenizer
import json
from tqdm import tqdm


def get_data(in_file, out_file):
    out = open(out_file, 'w')
    with open(in_file, 'r') as fp:
        data = fp.read()
        data = eval(data)
        for d in tqdm(data):
            text = d['text']
            mistakes = d['mistakes']
            correct_text_char = list(text)
            error_text_char = list(text)
            for mistake in mistakes:
                correct_text_char[int(mistake['loc']) - 1] = mistake['correct']
            correct_text = "".join(correct_text_char).replace(' ', '')
            error_text = "".join(error_text_char).replace(' ', '')
            # print(correct_text)
            # print(error_text)
            try:
                out.write(error_text + ' ' + correct_text + '\n')
            except Exception:
                print(correct_text_char)
                print(error_text_char)


def convert(in_file, out_file, max_len):
    bert_dir  = '/data/data01/gob_test/model_hub/chinese-bert-wwm-ext/vocab.txt'
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    with open(in_file, 'r') as fp:
        lines = fp.read().strip().split('\n')
        for line in lines:
            line = line.split(" ")
            correct_text = line[0]
            error_text = line[1]
            correct_inputs = tokenizer.encode_plus(
                correct_text,
                max_length=max_len,
                pad_to_max_length='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            error_inputs = tokenizer.encode_plus(
                error_text,
                max_length=max_len,
                pad_to_max_length='max_length',
                truncation=True,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            print(correct_inputs, error_inputs)
            break

if __name__ == '__main__':
    get_data('./train_data.json', './train_data.txt')
    get_data('./test_data.json', './test_data.txt')
    # convert('./train_data.txt', './train_data.pkl', 128)
    # convert('./test_data.txt', './test_data.pkl', 128)