import os

import pandas as pd
import numpy as np
from tqdm import tqdm, trange

data = pd.read_csv("resources" + os.sep + "dataset_biluo.csv", encoding="utf8").fillna(method="ffill")


class SentenceGetter(object):

    def __init__(self, data, max_sent=None):
        self.index = 0
        self.max_sent = max_sent
        self.tokens = data["Token"]
        self.labels = data["BILUO"]

    def sentences(self):
        sent = []
        counter = 0

        for token, label in zip(self.tokens, self.labels):
            if token == "DOCSTART":
                continue
            sent.append((token, label))
            if token.strip() == ".":
                yield sent
                sent = []
                counter += 1
            if self.max_sent is not None and counter >= self.max_sent:
                return

    def get_next(self):
        try:
            while True:
                sent = []
                next_token = self.tokens[self.index]
                if next_token == "DOCSTART":
                    continue
                next_label = self.labels[self.index]
                sent.append((next_token, next_label))
                self.index += 1
                if next_token.strip() == ".":
                    return sent
        except:
            return None


getter = SentenceGetter(data)

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

MAX_LEN = 75
bs = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

print(torch.cuda.get_device_name(0))

all_sentences = [[token for token, label in sent] for sent in getter.sentences()]
print(all_sentences[0])

all_orig_labels = [[label for token, label in sent] for sent in getter.sentences()]
print(all_orig_labels[0])

train_sentences, test_sentences, train_orig_labels, test_orig_labels = train_test_split(all_sentences, all_orig_labels, random_state=2018, test_size=0.2)

tags_vals = list(set(data["BILUO"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')

def tokenize(sentences, orig_labels):
    tokenized_texts = []
    labels = []
    for sent, sent_labels in zip(sentences, orig_labels):
        bert_tokens = []
        bert_labels = []
        for orig_token, orig_label in zip(sent, sent_labels):
            b_tokens = tokenizer.tokenize(orig_token)
            bert_tokens.extend(b_tokens)
            for b_token in b_tokens:
                bert_labels.append(orig_label)
        tokenized_texts.append(bert_tokens)
        labels.append(bert_labels)

        assert len(bert_tokens) == len(bert_labels)

    return tokenized_texts, labels


train_tokenized_texts, train_labels = tokenize(train_sentences, train_orig_labels)
print(train_tokenized_texts[0])
print(train_labels[0])


def pad_sentences_and_labels(tokenized_texts, labels):
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                         dtype="long", truncating="post")

    attention_masks = [[float(i > 0) for i in ii] for ii in input_ids]

    return input_ids, tags, attention_masks


input_ids, tags, attention_masks = pad_sentences_and_labels(train_tokenized_texts, train_labels)

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=len(tag2idx))

model.cuda()

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

from seqeval.metrics import f1_score

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


epochs = 1
max_grad_norm = 1.0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss / nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


def test_model():
    test_tokenized_texts, test_labels = tokenize(test_sentences, test_orig_labels)
    input_ids, tags, attention_masks = pad_sentences_and_labels(test_tokenized_texts, test_labels)

    val_inputs = torch.tensor(input_ids)
    val_tags = torch.tensor(tags)
    val_masks = torch.tensor(attention_masks)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    counter = 0
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

    print(test_sentences[0])
    print(pred_tags[0])
    print(valid_tags[0])


test_model()