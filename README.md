#### 项目简介

这是一份清华大学2021年人工神经网络课程Cepsum比赛项目代码

本项目基于pytorch实现，在requirements.txt中列举了部分重要依赖

Report已经包含了详细的模型解释与实验结果，这里只解释代码运行过程

#### 数据准备：

- 分词方法： `python cut.py`

  - 对提供的train.jsonl，val.jsonl，test_public.jsonl进行jieba分词

  - 需要将上述文件放置在./data路径下

- 生成属性词表方法：`python extract_attr_words.py`

  - 生成用于only-copy机制的属性词表，需要在分词后进行

- 生成词表方法：`python gen_vocab.py`

  - 生成用于词嵌入的词嵌入矩阵，需要在分词后进行

#### 模型训练： 

- 训练方法：`python train.py`，需附加训练参数

  - 训练参数：

  ```python
  parser.add_argument('--name', type=str, default="only_copy") # 模型命名
  parser.add_argument('--batch_size', type=int, default=32) 
  parser.add_argument('--embed_dim', type=int, default=300) # 词嵌入维度
  parser.add_argument('--hidden_size', type=int, default=512) # LSTM隐藏状态维度
  parser.add_argument('--learning_rate', type=float, default=5e-4) # 学习率
  parser.add_argument('--epoch_num', type=int, default=10) # 训练epoch数目
  parser.add_argument('--model_save_path', type=str, default="./model") # 模型保存路径
  parser.add_argument('--resume', type=str, default=None) # 加载已有模型路径
  parser.add_argument('--attri_words_path', type=str, default='./vocab/attr_words.txt') # 属性词表路径
  ```

#### 摘要生成与评价

- 生成方法：`python gen.py` 或 `python beam_gen.py`，需附加生成参数

  - gen.py生成参数

  ```python
  parser.add_argument('--module_dict', type=str, default="./model/model_only_copy_8") # 加载模型路径
  parser.add_argument('--input_path', type=str, default="./data/cut_valid.txt") # 输入路径
  parser.add_argument('--output_path', type=str, default="./data/gen_only_copy_valid.txt") # 生成摘要保存路径
  parser.add_argument('--attri_words_path', type=str, default='./vocab/attr_words.txt') # 属性词表路径
  parser.add_argument('--hidden_size', type=int, default=512) # 模型内LSTM隐藏状态维度
  ```

  - beam_gen.py生成参数：

  ```python
  parser.add_argument('--module_dict', type=str, default="./model/model_only_copy_8") # 加载模型路径
  parser.add_argument('--input_path', type=str, default="./data/cut_valid.txt") # 输入路径
  parser.add_argument('--output_path', type=str, default="./data/gen_only_copy_valid.txt") # 生成摘要保存路径
  parser.add_argument('--attri_words_path', type=str, default='./vocab/attr_words.txt') # 属性词表路径
  parser.add_argument('--hidden_size', type=int, default=512) # 模型内LSTM隐藏状态维度
  parser.add_argument('--beam_size', type=int, default=5) # beam search窗口大小
  parser.add_argument('--min_length', type=int, default=32) # 生成摘要的最小长度
  ```

- 上述过程输出一个把摘要按行排列的txt文件，将其转换为jsonl文件方法为：`python gen_json_for_eval.py`，需附加转换参数

  - gen_json_for_eval.py转换参数：

  ```python
  parser.add_argument('--format_path', type=str, default='./data/valid.jsonl') # 目标格式的jsonl文件
  parser.add_argument('--input_path', type=str, default='./data/gen_copy_valid.txt') # 输入txt文件
  parser.add_argument('--output_path', type=str, default='./data/gen_copy_valid.jsonl') # 转换后的jsonl文件
  ```

- 评价方法：`python eval.py [预测结果] [参考结果]`

  - 直接使用了GPT2 baseline中的评测程序
  - 参考结果（默认模型参数，k=1）：

  ```python
  {'rouge-1-f': 0.2490337228602475, 'rouge-1-p': 0.324999488971925, 'rouge-1-r': 0.2088078313327328, 'rouge-2-f': 0.05730533929489032, 'rouge-2-p': 0.0712768646915614, 'rouge-2-r': 0.04962097089009146, 'rouge-l-f': 0.20061443023544046, 'rouge-l-p': 0.26422906208460506, 'rouge-l-r': 0.1672926403876146}
  ```

  