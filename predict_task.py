from celery import Celery
import paddle
import paddle.nn.functional as F
from model import (
    BiLSTMAttentionModel,
    BoWModel,
    CNNModel,
    GRUModel,
    LSTMModel,
    RNNModel,
    SelfInteractiveAttention,
)
from utils import preprocess_prediction_data
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab

broker = 'redis://127.0.0.1:6379/1'
backend = 'redis://127.0.0.1:6379/2'

app = Celery(__name__, broker=broker, backend=backend)

paddle.set_device("cpu")

vocab = Vocab.from_json(args.vocab_path)
label_map = {0: "negative",1: "neura", 2: "positive"}
vocab_size = len(vocab)
num_classes = len(label_map)
pad_token_id = vocab.to_indices("[PAD]")

def predict(model, data, label_map, batch_size=1, pad_token_id=0):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `se_len`(sequence length).
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """

    # Separates data into some batches.
    batches = [data[idx : idx + batch_size] for idx in range(0, len(data), batch_size)]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_token_id),  # input_ids
        Stack(dtype="int64"),  # seq len
    ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in batches:
        texts, seq_lens = batchify_fn(batch)
        texts = paddle.to_tensor(texts)
        seq_lens = paddle.to_tensor(seq_lens)
        logits = model(texts, seq_lens)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results



@app.task
def predict_cnn(data):
    model = CNNModel(vocab_size, num_classes, padding_idx=pad_token_id)
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    tokenizer = JiebaTokenizer(vocab)
    examples = preprocess_prediction_data(data, tokenizer)
    return predict(
        model,
        examples,
        label_map=label_map,
        batch_size=1,
        pad_token_id=vocab.token_to_idx.get("[PAD]", 0),
    )

@app.task
def predict_lstm(x, y):
    model = LSTMModel(vocab_size, num_classes, direction="bidirect", padding_idx=pad_token_id)
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    tokenizer = JiebaTokenizer(vocab)
    examples = preprocess_prediction_data(data, tokenizer)
    return predict(
        model,
        examples,
        label_map=label_map,
        batch_size=1,
        pad_token_id=vocab.token_to_idx.get("[PAD]", 0),
    )

@app.task
def predict_bilstm(data):
    model = LSTMModel(vocab_size, num_classes, direction="bidirect", padding_idx=pad_token_id)
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    tokenizer = JiebaTokenizer(vocab)
    examples = preprocess_prediction_data(data, tokenizer)
    return predict(
        model,
        examples,
        label_map=label_map,
        batch_size=1,
        pad_token_id=vocab.token_to_idx.get("[PAD]", 0),
    )

@app.task
def predict_bilstm_attn(data):
    lstm_hidden_size = 196
    attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
    model = BiLSTMAttentionModel(
        attention_layer=attention,
        vocab_size=vocab_size,
        lstm_hidden_size=lstm_hidden_size,
        num_classes=num_classes,
        padding_idx=pad_token_id,
    )
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    tokenizer = JiebaTokenizer(vocab)
    examples = preprocess_prediction_data(data, tokenizer)
    return predict(
        model,
        examples,
        label_map=label_map,
        batch_size=1,
        pad_token_id=vocab.token_to_idx.get("[PAD]", 0),
    )

@app.task
def predict_birnn(data):
    model = RNNModel(vocab_size, num_classes, direction="bidirect", padding_idx=pad_token_id)
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    tokenizer = JiebaTokenizer(vocab)
    examples = preprocess_prediction_data(data, tokenizer)
    return predict(
        model,
        examples,
        label_map=label_map,
        batch_size=1,
        pad_token_id=vocab.token_to_idx.get("[PAD]", 0),
    )


