## chatbot

---



### Required Environment

- PyCharm
- Python 3.6
- pytorch 1.0.1

---



### Model

目前使用了 Seq2Seq + Global Attention

![](./pics/seq2seq.png)

![](./pics/global_attn.png)

---



### Project Files

chatbot_main.py

chatbot_processdata.py

chatbot_cocabulary.py

chatbot_model.py

chatbot_model.py

chatbot_train.py

chatbot_evaluate.py

---



### Others

代码是直接可以运行的，train 完后 evaluate 即可。

数据是小量的英文电影对话，本项目在cpu上运行不超过半小时，可直接进行一问一答。

后期会加入中文数据，并添加其他模型。

主要参考了pytorch文档。

供大家参考，欢迎 **star** 或 **fork**  后期会慢慢完善。

