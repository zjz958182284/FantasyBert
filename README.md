# FantasyBert
An easy-to-use framework for BERT models, with trainers, various NLP tasks and **detailed annonations** for beginnners.

Some code are edit on transformers and FastNLP modules.

This project is **not** for commercial use, and is only for **self-use** to perform various NLP tasks conveniently.

The trainer now supoorts: fp16, adv training, ema, gradient clip, early stop,self-defined losses and metrics and some state-of-art techniuques.

### How to use
```python
pip install fantasybert
```
The lastest version is 0.1.2


### Supported NLP tasks in examples
#### semantic text similarity
cross-bert.py  datasets:lcqmc

sentence-bert.py

consent.py

simcse.py

promptbert.py

#### text classification
tnews.py

#### named entity recogtion
bert-crf sequence labeling

global-pointer nested NER

#### relation extraction
casrel.py

#### text generation
unilm.py
