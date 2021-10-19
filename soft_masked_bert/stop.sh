ps -ef|grep "soft_masked_bert_main.py"|grep -v grep|awk '{print $2}'|xargs kill -9
