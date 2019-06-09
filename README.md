#Bert_Danmaku_Embedding

Pre-trained Bert model for generating Danmaku Embedding

<p>
    <img src="image/dd_center.png"/>
</p>

###介绍

作为NLP领域最近最大的进展之一，由[谷歌在2018年发布的Bert](https://github.com/google-research/bert.git)能够在各种自然语言处理任务上表现出尖端的效果。利用Bert包含以下两个步骤：预训练以及fine-tune。作为预训练部分，Bert将输入的句子投影到高维空间，也就是表示为Embedding vector。而对于fine-tune部分，Bert则将通过标注好的数据在各种自然语言处理任务上大显身手。虽然谷歌已经提供了[中文的预训练模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)，但是用弹幕数据库对谷歌提供的模型进行再一次的预训练是有必要的，而这将对接下来Bert在具体任务上的效果产生影响。



###准备

####1. 数据组成

本次训练所用的数据由以下两部分组成：

  \1. 由[simon3000](https://github.com/simon300000)收集的[vtuber直播弹幕数据库](https://github.com/bilibili-dd-center/bilibili-vtuber-danmaku.git)

  \2. 通过爬虫获得的[vtuber相关视频弹幕](https://github.com/bilibili-dd-center/Danmaku_dataset_augmentation.git)

需要注意的是，来自同一个视频或同一次直播的弹幕被划分到一起。换言之，我们把来自一个来源的弹幕看作一篇文章，并且每条弹幕占一行。对于不同的文章，中间用空格隔开。这是Bert输入要求的格式。

#### 2. 数据预处理

谷歌提供的预训练模型之中，有字典vocab.txt存在。其中包含了常用的汉字和符号。如果对其进行更改或添加，则要求的预训练时间会明显变长。在这里，我们只用谷歌提供的字典，并且剔除了含有字典中不存在字符的弹幕。如，包含emoji表情的弹幕。经过简单转换的数据存在[danmaku_text_pure.txt](https://drive.google.com/file/d/1Z2JobhFHcXyMYxa_dP2eHHXs4Jv_WANi/view?usp=sharing)内。如果你需要在自己的数据库上训练，那么可以参照相应的格式来自行制作txt文件。

## 训练

介于训练Bert所需要点计算量异常的大，这里我们用谷歌的TPU来进行训练。谷歌的TPU_V3.8在运行Bert的速度上是 NVIDIA GeForce GTX 1080 Ti 的35倍，推荐给没有耐心等待训练结果的你。谷歌给所有新注册的用户300刀的试用金，而你可以用每小时2.4刀的价格使用TPU。这里是利用TPU来进行预训练的[傻瓜教程](https://github.com/pren1/A_Pipeline_Of_Pretraining_Bert_On_Google_TPU.git)，以供参考。



### 训练结果

在烧掉了一百刀的试用金之后，我们得到了让人满意的结果。实际上，只需要以2048的batch size运行12000个iteration 即可。需要注意的是，对于GTX 1080-Ti来说，大于32的batch size都会造成显存不足。这也是推荐使用Tpu的原因之一。

下表展示了训练的结果。可以看到，最初的谷歌预训练网络在弹幕数据集上的两个任务（Masked LM 以及 Next Sentence Prediction）的效果并不算好。而在训练之后，效果均得到了提升。Evaluation Set则是由新收集的直播间弹幕组成，因此在训练时模型并没有见过。

|                                   | masked_lm_accuracy | masked_lm_loss | next_sentence_accuracy | next_sentence_loss |
| --------------------------------- | :----------------: | :------------: | :--------------------: | ------------------ |
| **Pre-trained model**             |     0.48032853     |    3.155205    |         0.5275         | 2.2248225          |
| **Tuned model on training set**   |     0.7067532      |   1.3825288    |        0.97375         | 0.086835           |
| **Tuned model on evaluation set** |     0.65252227     |   1.7447132    |         0.865          | 0.3875436          |

⚠️注意⚠️

如果降低iteration数目，模型在training set上的表现会和evaluation set上一样。然而，通过可视化方法，我发现模型实际上并没有学习到足够的相关知识。而就目前来说，这已经是能够做到的最好的结果（如果你有兴趣可以接着调）。通过以下的可视化图片我们可以观察到，模型的效果实际上还是不错的，尚且处于可以接受的范围。至于在下游NLP项目上的效果，则需要一边调节iteration数目，一边观测最终效果，以此来得到最合适的training iteration。

收据：

<p>
    <img src="image/payment.png"/>
</p>

###使用Bert模型生成embedding vector

这里我们用了bert-as-service这个非常方便的工具，具体安装见链接。安装完成之后，运行：

```
renpeng$ bert-serving-start -model_dir chinese_L-12_H-768_A-12/ -tuned_model_dir=/your_path_to_your_model/ -ckpt_name=model.ckpt-12000 -num_worker=1 -pooling_layer=-1
```

以开启bert本地服务。请把 your_path_to_your_model 替换成模型所在的路径，model.ckpt-225000替换成模型名称。

### 数据可视化

为了验证embedding 的结果，我们首先用主成分分析法（PCA）将数据从768维降到50维，再用t-SNE投影到二维。需要注意的是，从高维投影到低维，一部分信息不可逆的损失掉了。这部分的脚本在Danmaku_similarity.py内。

作为对比，我们首先运用谷歌提供的中文预训练模型来生成word embedding。如果两个概念类似的话，那么它们在二维平面上的距离也很近：

<p>
    <img src="image/start_point_ok.png"/>
</p>

这个不行啊，看起来我们刚从谷歌那里接过来的模型酱完全搞不清楚状况。虽然语义相近的词在一起，但是对于vtuber圈子里的一些梗完全没辙，而这也是我们要用弹幕数据库进行预训练的原因。

接下来是认真观看并学习了vtuber相关弹幕的模型酱：

<p>
    <img src="image/Ok_finally.png"/>
</p>

Bravo！🎉你做到了，模型酱！🎉我们可以来分析一下：

1. Mea，aqua离的很近，没想到模型酱你也是个meauqa党。需要注意的是，aqua和mea因为都有a字母的缘故，就算是最初的预训练模型也有可能把她们放的很近。但是阿夸是汉字，在字面上与mea没有任何关系。就算这样阿夸和mea也还是异常的近，是糖! (高 科 技 制 糖)
2. 山田赫敏阿夸说。（完 全 确 信）
3. 爱小姐和mea和774的位置非常微妙。
4. 猫宫, 爱酱和ai非常近。
5. yyut和阿律非常近，并且这两个人离homo的距离也很近。
6. 晚安和88888联系到了一起。
7. vtuber们聚集起来了！
8. 夏 色 吹 雪
9. 委员长和彩虹社很近，因为阿委是属于彩虹社的。
10. 张京华懂个锤子中文。

好，看来读了太多弹幕的模型酱已经完全dd化了。恭喜你加入我们GitHub DD Center，模型酱！你就是第十位成员（迫真）！咦？过拟合？根本不存在的! (大嘘)



### 接模型酱回家💗

好，这里是[下载链接](https://drive.google.com/drive/folders/1tYFl-ODwZOs3vdnz2tQMDwHXEA_2Ca3R?usp=sharing)。如果你有合适的下游自然语言处理项目，可以从这个dd模型开始进行fine-tune，预计结果会比直接用谷歌提供的预训练模型效果好。
