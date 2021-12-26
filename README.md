# CEPSUM京东摘要数据集

### 简介

CEPSUM是一个京东公司在京东平台上收集的摘要数据集。

### 论文

[On the Faithfulness for E-commerce Product Summarization](https://www.aclweb.org/anthology/2020.coling-main.502.pdf). COLING 2020

### 数据规模

| 训练集 | 验证集 | 测试集 |
| ------ | ------ | ------ |
| 3,000  | 1,000  | 1,000  |



### 数据样例

```
{
	"table": 
		{"闭合方式": "拉链", "风格": "简约时尚", "内部结构": "拉链暗袋", "开口方式": "拉链搭扣", "外形": "箱形", "适用场景": "休闲", "颜色": "撞色", "图案": "条纹", "款式": "横款", "箱包硬度": "软", "主要材质成分": "牛皮", "里料材质": "涤纶", "功能位": "内插袋", "斜挎带": "斜挎带", "箱包外袋种类": "内贴袋", "材质": "牛皮", "适用人群": "青年", "流行元素": "小方包"}
	"source": 
		"半坡饰族斜挎包牛皮女包包女宽肩带撞色条纹小方包运动风单肩包红色，条纹风格逆袭，柔软牛皮，实用>型收纳，时毗宽肩带，匠心工艺，深蓝，牛皮拉牌，大红，触感柔软，不同风格搭配的冲突感正是时毗的秘诀所在，于休闲中诠释自我。轻奢光泽坚韧耐用，1拉链暗袋，让代表性条纹装饰成为手袋的主角，实力与颜值集于一身。伴随着运动成为时尚生活方式，荔枝纹牛皮，纹理清晰，是定型包立体轮廓的默契组合。1磁扣袋，耐看实用，纹理清晰，街拍钟爱宽肩带的时毗，肩膀钟爱4.5cm的舒适体验。时毗趋势，1插袋，送爱人，送亲人，口A4杂志，送自己，送朋友，4.7寸，图片仅供参考，5.5寸，生活变得更有意义。你要好好犒>劳自己。表达对亲人的爱。用一份心意的热度，人生漫漫，因为有你，共欢笑共患难"
	"targets": 
		[
			"包面选用荔枝纹材质，皮面呈现自然纹理，耐磨防刮划。包面点缀撞色织带做装>饰，充满休闲运动风，却不失时髦感。配以宽肩带，能够分散压力，为你带来舒适的背包体验。"
		]
}
```

- "table" (`str`): 产品属性表（不同属性用`\t`切分）。
- "source" (`str`)：产品文本描述。
- "target" (`list`)：产品摘要，"target"字段包含多个合理的摘要。

### 作者列表(数据集作者)

Peng Yuan, Haoran Li , Song Xu, Youzheng Wu, Xiaodong He and Bowen Zhou

### 制作单位

京东AI研究院



### 论文引用

```
@inproceedings{yuan-etal-2020-faithfulness,
    title = "On the Faithfulness for {E}-commerce Product Summarization",
    author = "Yuan, Peng  and
      Li, Haoran  and
      Xu, Song  and
      Wu, Youzheng  and
      He, Xiaodong  and
      Zhou, Bowen",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.502",
    doi = "10.18653/v1/2020.coling-main.502",
    pages = "5712--5717",
    abstract = "In this work, we present a model to generate e-commerce product summaries. The consistency between the generated summary and the product attributes is an essential criterion for the ecommerce product summarization task. To enhance the consistency, first, we encode the product attribute table to guide the process of summary generation. Second, we identify the attribute words from the vocabulary, and we constrain these attribute words can be presented in the summaries only through copying from the source, i.e., the attribute words not in the source cannot be generated. We construct a Chinese e-commerce product summarization dataset, and the experimental results on this dataset demonstrate that our models significantly improve the faithfulness.",
}
```



### 评测代码使用

```shell
python eval.py prediction_file test_private_file
```

- 预测结果需要和评测代码保持一样的格式
- 依赖：rouge==1.0.0，jieba=0.42.1
- 评测指标为rouge-1, rouge-2, rouge-l，输出结果为字典格式：

```python
{'rouge-1-f': _, 'rouge-1-p': _, 'rouge-1-r': _, 'rouge-2-f': _, 'rouge-2-p': _, 'rouge-2-r': _, 'rouge-l-f': _, 'rouge-l-p': _, 'rouge-l-r': _}
```



### Baseline: GPT2

- 训练方法：`bash run.sh`
- 训练参数：

```shell
env CUDA_VISIBLE_DEVICES=0 python3 -u run_clm.py
    --train_file=./gpt_data/train.txt \  # 训练集
    --validation_file=./gpt_data/val.txt \  # 验证集
    --output_dir=./model \  # 模型保存文件夹
    --save_total_limit=10 \  # 最多保存多少个checkpoint
    --fp16 \  # 是否使用半精度（半精度可以节省内存）
    --per_device_train_batch_size=7 \  # 训练时的batch_size
    --per_device_eval_batch_size=7 \  # 验证时的batch_size
    --num_train_epochs=40 \  # 训练轮数
    --logging_steps=5 \  # 记录loss的步数
    --model_name_or_path=uer/gpt2-chinese-cluecorpussmall \  # 预训练模型name，也可以下载到本地换成本地路径，模型列表参考https://huggingface.co/models
    --learning_rate=3e-5 \  # 学习率
    --evaluation_strategy=epoch \  # 评价策略
    --do_train --do_eval \  # 是否训练/验证
    --overwrite_output_dir \  # 重写输出文件夹
    --load_best_model_at_end \  # 训练完成后将best checkpoint保存在
    --gradient_accumulation_steps 10 \  # 累计多少步更新一次参数
```

- 生成方法：`bash gen.sh`
- 生成参数：

```shell
env CUDA_VISIBLE_DEVICES=0 python3 run_generation.py \
    --model_type=gpt2 \  # 模型类型
    --model_name_or_path=./model \  # 模型路径
    --k 0 \  # top-k采样，0表示不使用top-k采样
    --p 0.9 \ # top-p采样，0表示不使用top-p采样
```

- 参考结果（解码方法：top_p with p=0.9）

```python
验证集结果：{'rouge-1-f': 0.2801660402537134, 'rouge-1-p': 0.2694756257452723, 'rouge-1-r': 0.2992693902090392, 'rouge-2-f': 0.030485331478296827, 'rouge-2-p': 0.029063926863698138, 'rouge-2-r': 0.03279660871088596, 'rouge-l-f': 0.13609651808795867, 'rouge-l-p': 0.1343010465264509, 'rouge-l-r': 0.14086912694738113}
测试集结果：{'rouge-1-f': 0.2854995121492525, 'rouge-1-p': 0.27250697714021677, 'rouge-1-r': 0.3076843533652649, 'rouge-2-f': 0.03340655514602158, 'rouge-2-p': 0.03171592559622199, 'rouge-2-r': 0.03611462209462058, 'rouge-l-f': 0.14105515047486533, 'rouge-l-p': 0.13899241898447487, 'rouge-l-r': 0.14702754004652574}
```

