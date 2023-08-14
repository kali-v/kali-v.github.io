---
title: Mastering ARQMath
url: /projects/arqmath
type: posts
---

This is the initial material for publication [Diverse Semantics Representation is King](https://ceur-ws.org/Vol-3180/paper-02.pdf)


# Ranked Retrieval System

The system computes cosine similarity between query embedding and answer embeddings for each query, and top n answers are further re-ranked using Cross-Encoders.
I tried to use no cross-encoder, single CE, and ensemble CEs (weighted average). Introducing CE improved the performance significantly. Using one smaller CE alongside the main one didn't really improve the best score but might be a more stable approach.

The cross-encoders were fine-tuned, and the final models can be downloaded at https://drive.google.com/drive/folders/1eiZl8ftAR1rYp2TFf6SZ2VF2f6HVa0mz?usp=sharing



```python
%%capture
#! pip install git+https://github.com/MIR-MU/pv211-utils.git
#! pip install transformers sentence-transformers
#! pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
```

```python
text_format = "text+latex"
retriever_model_id = "all-MiniLM-L12-v2"
reranker_model_id = "sentence-transformers/stsb-roberta-base-v2"
```

```python
from pv211_utils.arqmath.loader import load_answers
from pv211_utils.arqmath.entities import ArqmathAnswerBase, ArqmathQueryBase, ArqmathQuestionBase
from typing import List

import torch


class Answer(ArqmathAnswerBase):
    def __init__(self, document_id: str, body: str, upvotes: int, is_accepted: bool):
        super().__init__(document_id, body, upvotes, is_accepted)
        self.representation = ' '.join(self.body.split())  # remove multiple spaces


class Question(ArqmathQuestionBase):
    def __init__(self, document_id: str, title: str, body: str, tags: List[str],
                 upvotes: int, views: int, answers: List[Answer]):
        super().__init__(document_id, title, body, tags, upvotes, views, answers)
        self.representation = self.title + ' ' + ', '.join(tags).replace("-", " ") + ". " + self.body


class Query(ArqmathQueryBase):
    def __init__(self, query_id: int, title: str, body: str, tags: List[str]):
        super().__init__(query_id, title, body, tags)
        self.representation = self.title + ' ' + ', '.join(tags).replace("-", " ") + ". " + self.body


answers = load_answers(text_format, Answer, cache_download=f'/var/tmp/pv211/arqmath2020_answers_{text_format}.json.gz')

tanswers = [answers[ans].body.lower() for ans in answers]
```

```python
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

retriever_model = SentenceTransformer(retriever_model_id, device="cuda")
retriever_model.eval()

with torch.no_grad():
    answers_embeddings = retriever_model.encode(tanswers, convert_to_tensor='pt', batch_size=64)

answers_embeddings = answers_embeddings.detach().cpu()
answers_embeddings_np = answers_embeddings.numpy()
norm_answers_embedding = [norm(embedding) for embedding in answers_embeddings]
```

```python
norm_answers_embedding = [norm(embedding) for embedding in answers_embeddings]
```

```python
from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(reranker_model_id, num_labels=1)

# model.model.load_state_dict(torch.load('./ce_berta_modelv22_x120.pth'))

```

```python
from pv211_utils.arqmath.irsystem import ArqmathIRSystemBase
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import numpy as np


class IRSystem(ArqmathIRSystemBase):
    def __init__(self):
        self.answers = list(answers.values())

    def _get_query_embedding(self, query: Query):
        return retriever_model.encode(query.representation)

    def search(self, query: Query):
        query_embedding = self._get_query_embedding(query)
        query_embedding_norm = norm(query_embedding)

        cos_sims = [dot(query_embedding, answers_embeddings_np[i]) / (query_embedding_norm * norm_answers_embedding[i])
                    for i in range(len(answers_embeddings))]

        predictions_retriever = np.array(cos_sims).argsort()[::-1]
        
        # sorting in batches
        batch_indexes = [4, 8, 12, 16, 32, 64, 128]
        for i in range(len(batch_indexes)):
            start_index = 0 if i == 0 else batch_indexes[i - 1]
            top_set = [[query.representation, self.answers[predictions_retriever[j]].representation]
                       for j in range(start_index, batch_indexes[i])]
            reranker_preds = reranker_model.predict(top_set, batch_size=16)
            top_preds = np.array(reranker_preds).argsort()[::-1]

            for top_doc in top_preds:
                yield self.answers[predictions_retriever[top_doc + start_index]]

        for doc in predictions_retriever:
            yield self.answers[doc]

```

```python
from pv211_utils.arqmath.loader import load_queries, load_judgements
from pv211_utils.evaluation_metrics import mean_average_precision

test_queries = load_queries(text_format, Query, year=2021)
test_judgements = load_judgements(test_queries, answers, year=2021)
map_score = mean_average_precision(IRSystem(), test_queries, test_judgements, 10, 1)
map_score

```

# Training Dataset

At each epoch, we create a list of examples (query, answer, relevance). The relevant answer is returned with prob. 0.5.


```python
import random
import torch

from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from sentence_transformers import InputExample

MAX_SEQ_LEN = 512

class ArqMathDataset(Dataset):
    def __init__(self, queries: OrderedDict, judgements: OrderedDict, answers: OrderedDict):
        self.queries = list(queries.values())
        self.answers = list(answers.values())
        
        self.judgements = {}
        self.ids = set()

        for judgement in judgements:
            q = judgement[0].query_id
            self.judgements.setdefault(q, []).append(judgement[1])
            self.ids.add(q)
        self.ids = sorted(self.ids)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def random_crop(self, string: str) -> str:
        """Do a random crop of a string that fits to a transformer
        
        Parameters
        ----------
        string: str
            string to be cropped
        
        Returns
        -------
        cropped string: str
        """
        lst = string.split(" ")
        mx = len(lst) - (MAX_SEQ_LEN + 1)  # max in seq lenght is 512 for most transformers

        # if the text is not that much longer just leave it like that 
        if mx > 10:
            st = random.randrange(0, mx)
            return " ".join(lst[st:st+MAX_SEQ_LEN])
        return string

    def __getitem__(self, idx: int) -> list:
        """Take text of query, random answer and their relevance
        
        Parameters
        ----------
        idx: int
            index of query
        
        Returns
        -------
        query text: str
        answer text: str
        relevance: Tensor
        """
        query = self.queries[idx]
        answer = None
        rel = torch.Tensor([[1]])

        if random.random() > 0.5:
            answer = random.choice(self.judgements[query.query_id])
        else:
            rel = torch.Tensor([[0]])
            while True:
                rnd = random.randint(0, len(self.answers))
                answer = self.answers[rnd]
                if answer not in self.judgements[query.query_id]:
                    break

        query_str = self.random_crop(query.body)
        answer_str = self.random_crop(answer.body)

        return InputExample(texts=[query_str, answer_str], label=rel.item())
```

```python
from collections import OrderedDict
from itertools import chain
from pv211_utils.arqmath.loader import load_queries, load_judgements

smaller_train_queries = load_queries(text_format, Query, 'train', year=2020)
smaller_validation_queries = load_queries(text_format, Query, 'validation', year=2020)

train_queries = OrderedDict(chain(smaller_train_queries.items(), smaller_validation_queries.items()))
validation_queries = load_queries(text_format, Query, 'test', year=2020)

bigger_train_queries = OrderedDict(chain(train_queries.items(), validation_queries.items()))

smaller_train_judgements = load_judgements(smaller_train_queries, answers, 'train', year=2020)
smaller_validation_judgements = load_judgements(smaller_validation_queries, answers, 'validation', year=2020)

train_judgements = smaller_train_judgements | smaller_validation_judgements
validation_judgements = load_judgements(validation_queries, answers, 'test', year=2020)

bigger_train_judgements = train_judgements | validation_judgements
```

```python
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

val_dataset = ArqMathDataset(validation_queries, validation_judgements, answers)

val_examples = []
for j in range(20):
    for data in val_dataset:
        val_examples.append(data)

ce_eval = CEBinaryClassificationEvaluator.from_input_examples(val_examples)
print(ce_eval(reranker_model))
```

```python
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from torch.optim import AdamW

model = CrossEncoder(reranker_model_id, num_labels=1)

train_dataset = ArqMathDataset(train_queries, train_judgements, answers)
ce_eval = CEBinaryClassificationEvaluator.from_input_examples(val_examples)

def fit():
    train_examples = []

    # 50 samples per question
    for j in range(50):
        for data in train_dataset:
            train_examples.append(data)

    train_dataloader = DataLoader(train_examples, batch_size=32)
    model.fit(train_dataloader, show_progress_bar=True, optimizer_class=AdamW)

epochs = 10
for i in range(epochs):
    fit()
    print(f"[{i}/{epochs}] val_eval: {ce_eval(model)}")

    if i % 2 == 0:
        torch.save(model.model.state_dict(), f'./model_{i}.pth')

```

```python
# load trained model
model.model.load_state_dict(torch.load('./model_0.pth'))
```
