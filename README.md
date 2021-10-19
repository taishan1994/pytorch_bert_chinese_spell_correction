# pytorch_bert_chinese_spell_correction
基于pytorch的中文拼写纠错，使用的模型是Bert以及SoftMaskedBert

# 依赖
```python
pytorch==1.6+
transformers==4.x+
```

# 说明
bert模型：hugging face上的bert-base-chinese<br>
本项目基于：https://github.com/FDChongLi/TwoWaysToImproveCSC<br>
在该项目基础上做了以下修改：<br>
1、修改tansformers为4.x版本（一些函数需要修改）；<br>
2、修改可以读取.pkl格式的模型；<br>
3、在计算损失和评价指标时，根据句子的真实长度进行计算；<br>
4、修改配置文件为config.py；<br>
目录结构：<br>
bert：<br>
--BertFineTune.py：模型<br>
--config.py：配置文件<br>
--dataset.py：pytorch数据集<br>
--bert_main.py：主运行函数<br>
--start.sh、stop.sh、restart.sh分别是后台运行的bash脚本，如果不后台运行，则将start.sh里面的命令拿出来运行即可。<br>
同理，softmaskedbert下面的也是一样的。<br>

# 运行结果
首先是加载TwoWaysToImproveCSC里面已经训练好的一些模型，可以去这里下载：<br>
百度网盘链接：https://pan.baidu.com/s/1O9mLjWSiXzxcPBy0fU-_BQ 提取码：y25e<br>
然后在checkpoints下放置。<br>
对于bert：<br>
```python
bert/baseline/initial/：
detection sentence accuracy:0.523,precision:0.5390295358649789,recall:0.5268041237113402,F1:0.5328467153284672
correction sentence accuracy:0.517,precision:0.5327004219409283,recall:0.520618556701031,F1:0.5265901981230449
sentence target modify:970,sentence sum:1000,sentence modified accurate:505
bert/baseline/sighan13：
detection sentence accuracy:0.799,precision:0.8685393258426967,recall:0.7969072164948454,F1:0.8311827956989247
correction sentence accuracy:0.792,precision:0.8606741573033708,recall:0.7896907216494845,F1:0.8236559139784946
sentence target modify:970,sentence sum:1000,sentence modified accurate:766
bert/pre_train/initial：
detection sentence accuracy:0.729,precision:0.7416317991631799,recall:0.7309278350515463,F1:0.7362409138110072
correction sentence accuracy:0.724,precision:0.7364016736401674,recall:0.7257731958762886,F1:0.731048805815161
sentence target modify:970,sentence sum:1000,sentence modified accurate:704
bert/pre_train/sighan13：
detection sentence accuracy:0.836,precision:0.8689581095596133,recall:0.8340206185567011,F1:0.8511309836927933
correction sentence accuracy:0.832,precision:0.8646616541353384,recall:0.8298969072164949,F1:0.8469226722777485
sentence target modify:970,sentence sum:1000,sentence modified accurate:805

对于data1：
1、加载pre_train/initial/model.pkl，并基于此继续训练sighan13：
lr=2e-7， batchsize=32, max_len=128, epoch=400，得到：
detection sentence accuracy:0.831,precision:0.8635875402792696,recall:0.8288659793814434,F1:0.8458705944239874
correction sentence accuracy:0.827,precision:0.8592910848549946,recall:0.8247422680412371,F1:0.8416622830089426
sentence target modify:970,sentence sum:1000,sentence modified accurate:800
bert_pretrain ,epoch 400,train_loss: 0.005944552947767079,valid_loss: 0.38606385549064726


对于data2：
1、使用pre_train/initial直接测试：
detection sentence accuracy:0.6632023384215654,precision:0.7274670466690417,recall:0.6632023384215654,F1:0.6938498131158681
correction sentence accuracy:0.613835660928873,precision:0.6733167082294265,recall:0.613835660928873,F1:0.6422018348623854
sentence target modify:3079,sentence sum:3079,sentence modified accurate:1890
2、在该基础上我们进行训练，并测试：
lr=2e-7， batchsize=32, max_len=128, epoch=1, 训练一个epoch后，f1达到0.73。尝试过多训练几个epoch，
发现验证损失上升，f1值下降，可能由于过拟合了。
```

对于softmaskedbert：<br>
```python
softmaskedbert/baseline/initial：
detection sentence accuracy:0.637,precision:0.6617336152219874,recall:0.6453608247422681,F1:0.6534446764091858
correction sentence accuracy:0.621,precision:0.6448202959830867,recall:0.6288659793814433,F1:0.6367432150313153
sentence target modify:970,sentence sum:1000,sentence modified accurate:610
softmaskedbert/baseline/sighan13：
detection sentence accuracy:0.789,precision:0.8515625,recall:0.7865979381443299,F1:0.8177920685959271
correction sentence accuracy:0.773,precision:0.8337053571428571,recall:0.7701030927835052,F1:0.8006430868167203
sentence target modify:970,sentence sum:1000,sentence modified accurate:747
softmaskedbert/pre_train/initial：
detection sentence accuracy:0.715,precision:0.7357293868921776,recall:0.7175257731958763,F1:0.7265135699373695
correction sentence accuracy:0.704,precision:0.7241014799154334,recall:0.7061855670103093,F1:0.7150313152400836
sentence target modify:970,sentence sum:1000,sentence modified accurate:685
softmaskedbert/pre_train/sighan13：
detection sentence accuracy:0.829,precision:0.8767720828789531,recall:0.8288659793814434,F1:0.8521462639109698
correction sentence accuracy:0.821,precision:0.8680479825517994,recall:0.8206185567010309,F1:0.843667196608373
sentence target modify:970,sentence sum:1000,sentence modified accurate:796

对于data1：
1、加载pre_train/initial/model.pkl，并基于此继续训练sighan13：
lr=2e-7， batchsize=32, max_len=128, epoch=400，得到：（发现中间过拟合了，在第epoch：162最好）
detection sentence accuracy:0.77,precision:0.8091106290672451,recall:0.7690721649484537,F1:0.7885835095137421
correction sentence accuracy:0.763,precision:0.8015184381778742,recall:0.7618556701030927,F1:0.781183932346723
sentence target modify:970,sentence sum:1000,sentence modified accurate:739
bert_pretrain ,epoch 162,train_loss: 5.798520624637604,valid_loss: 17.283852636814117



对于data2：
1、使用pre_train/initial直接测试：
detection sentence accuracy:0.6641766807405002,precision:0.7342908438061041,recall:0.6641766807405002,F1:0.6974761255115962
correction sentence accuracy:0.6122117570639818,precision:0.6768402154398564,recall:0.6122117570639818,F1:0.6429058663028651
sentence target modify:3079,sentence sum:3079,sentence modified accurate:1885
2、在该基础上我们进行训练，并测试：
lr=2e-7， batchsize=32, max_len=128, epoch=10, 
detection sentence accuracy:0.6411172458590452,precision:0.6885245901639344,recall:0.6411172458590452,F1:0.6639757820383451
correction sentence accuracy:0.5901266645014616,precision:0.6337635158702476,recall:0.5901266645014616,F1:0.6111671712075346
sentence target modify:3079,sentence sum:3079,sentence modified accurate:1817
bert_pretrain ,epoch 10,train_loss: 407.3908385746181,valid_loss: 13.833884827792645
save model done!
Time cost: 49003.112409591675 s。
发现效果反而没有最先初始化的好了，可能还没有收敛，不过训练一次时间太长了，目前就这样了。
```

在主运行函数中有一个预测函数可以使用，结果如下：
```python
======================================================
错误句子： ['你', '找', '到', '你', '最', '喜', '欢', '的', '工', '作', '，', '我', '也', '很', '高', '心', '。']
预测句子： ['你', '找', '到', '你', '最', '喜', '欢', '的', '工', '作', '，', '我', '也', '很', '高', '兴', '。']
======================================================
错误句子： ['刘', '墉', '在', '三', '岁', '过', '年', '时', '，', '全', '家', '陷', '入', '火', '海', '，', '把', '家', '烧', '得', '面', '目', '全', '飞', '、', '体', '无', '完', '肤', '。']
预测句子： ['刘', '墉', '在', '三', '岁', '过', '年', '时', '，', '全', '家', '陷', '入', '火', '海', '，', '把', '家', '烧', '得', '面', '目', '全', '非', '、', '体', '无', '完', '肤', '。']
======================================================
错误句子： ['遇', '到', '逆', '竟', '时', '，', '我', '们', '必', '须', '勇', '于', '面', '对', '，', '而', '且', '要', '愈', '挫', '愈', '勇', '，', '这', '样', '我', '们', '才', '能', '朝', '著', '成', '功', '之', '路', '前', '进', '。']
预测句子： ['遇', '到', '逆', '境', '时', '，', '我', '们', '必', '须', '勇', '于', '面', '对', '，', '而', '且', '要', '愈', '挫', '愈', '勇', '，', '这', '样', '我', '们', '才', '能', '朝', '著', '成', '功', '之', '路', '前', '进', '。']

```