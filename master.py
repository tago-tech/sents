import os
from flask import Flask, render_template, redirect, url_for, request
import argparse

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
from predict_task import predict_cnn, predict_lstm, predict_bilstm, predict_bilstm_attn, predict_birnn

from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab

app = Flask(__name__)

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'], default="cpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", type=int, default=1, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./vocab.json", help="The file path to vocabulary.")
args = parser.parse_args()

paddle.set_device(args.device.lower())

# Loads vocab.
vocab = Vocab.from_json(args.vocab_path)
label_map = {0: "negative",1: "neutral", 2: "positive"}
label_to_int_map = {"negative" : 0, "neutral": 1, "positive" : 2}

vocab_size = len(vocab)
num_classes = len(label_map)
pad_token_id = vocab.to_indices("[PAD]")

def load_model(network):
    param_path = "output/{}/checkpoints/best_model.pdparams".format(network)
    if not os.path.exists(param_path):
        print("can not found param for network=%s, skip load it" % network)
        return None
    if network == "bow":
        model = BoWModel(vocab_size, num_classes, padding_idx=pad_token_id)
    elif network == "bigru":
        model = GRUModel(vocab_size, num_classes, direction="bidirect", padding_idx=pad_token_id)
    elif network == "bilstm":
        model = LSTMModel(vocab_size, num_classes, direction="bidirect", padding_idx=pad_token_id)
    elif network == "bilstm_attn":
        lstm_hidden_size = 196
        attention = SelfInteractiveAttention(hidden_size=2 * lstm_hidden_size)
        model = BiLSTMAttentionModel(
            attention_layer=attention,
            vocab_size=vocab_size,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=num_classes,
            padding_idx=pad_token_id,
        )
    elif network == "birnn":
        model = RNNModel(vocab_size, num_classes, direction="bidirect", padding_idx=pad_token_id)
    elif network == "cnn":
        model = CNNModel(vocab_size, num_classes, padding_idx=pad_token_id)
    elif network == "gru":
        model = GRUModel(vocab_size, num_classes, direction="forward", padding_idx=pad_token_id, pooling_type="max")
    elif network == "lstm":
        model = LSTMModel(vocab_size, num_classes, direction="forward", padding_idx=pad_token_id, pooling_type="max")
    elif network == "rnn":
        model = RNNModel(vocab_size, num_classes, direction="forward", padding_idx=pad_token_id, pooling_type="max")
    else:
        raise ValueError(
            "Unknown network: %s, it must be one of bow, lstm, bilstm, cnn, gru, bigru, rnn, birnn and bilstm_attn."
            % network
        )
    # Loads model parameters.
    state_dict = paddle.load(param_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s, network: %s" % (param_path, network))
    return model

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

model_map = {}
for network in ["cnn", "lstm", "bilstm", "bilstm_attn","rnn", "birnn"]:
    print("try to load network: %s" % network)
    model_instance = load_model(network)
    if model_instance is not None:
        model_map[network] = model_instance
tokenizer = JiebaTokenizer(vocab)

model_to_func = {"cnn": predict_cnn, "lstm": predict_lstm, "bilstm": predict_bilstm, "bilstm_attn": predict_bilstm_attn, "birnn": predict_birnn}

@app.route('/index', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    print("recive text: %s" % request.form["text"])
    data = [request.form["text"]]
    examples = preprocess_prediction_data(data, tokenizer)
    dict = {}
    for network in model_map:
        results = model_to_func[network].delay(data)
        for idx, text in enumerate(data):
            print("network: {} data: {} \t Label: {} \t ".format(text, results[idx], network))
        dict[network] = results[0]
    return render_template('result.html', result = dict, text= request.form["text"])

@app.route('/fallback', methods = ['POST'])
def fallback():
    print("feedback text=" + request.form["text"] + ", result=" + request.form["result"])
    if request.form["result"] in label_to_int_map:
        with open("data/feedback.csv", "a") as feedback_out:
            print("{}\t{}".format(request.form["text"], label_to_int_map[request.form["result"]]), file=feedback_out)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9001)