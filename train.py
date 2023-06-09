import argparse
import random
from functools import partial

import numpy as np
import paddle
from model import (
    BiLSTMAttentionModel,
    BoWModel,
    CNNModel,
    GRUModel,
    LSTMModel,
    RNNModel,
    SelfInteractiveAttention,
)
from utils import build_vocab, convert_example

from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import load_dataset

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--epochs", type=int, default=15, help="Number of epoches for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'mlu', 'npu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--save_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./vocab.json", help="The file path to save vocabulary.")
parser.add_argument('--network', choices=['bow', 'lstm', 'bilstm', 'gru', 'bigru', 'rnn', 'birnn', 'bilstm_attn', 'cnn'],
                    default="bilstm", help="Select which network to train, defaults to bilstm.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
args = parser.parse_args()
# yapf: enable


def set_seed(seed=1000):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def create_dataloader(dataset, trans_fn=None, mode="train", batch_size=1, batchify_fn=None):
    """
    Creats dataloader.

    Args:
        dataset(obj:`paddle.io.Dataset`): Dataset instance.
        trans_fn(obj:`callable`, optional, defaults to `None`): function to convert a data sample to input ids, etc.
        mode(obj:`str`, optional, defaults to obj:`train`): If mode is 'train', it will shuffle the dataset randomly.
        batch_size(obj:`int`, optional, defaults to 1): The sample number of a mini-batch.
        batchify_fn(obj:`callable`, optional, defaults to `None`): function to generate mini-batch data by merging
            the sample list, None for only stack each fields of sample in axis
            0(same as :attr::`np.stack(..., axis=0)`).

    Returns:
        dataloader(obj:`paddle.io.DataLoader`): The dataloader which generates batches.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train" else False
    if mode == "train":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader


def my_read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        next(f)
        for line in f:
            words, labels = line.strip('\n').split('\t')
            yield {'text': words, 'label': labels}

if __name__ == "__main__":
    paddle.set_device(args.device)
    set_seed(1000)
    
    # Loads dataset.
    train_ds = load_dataset(my_read, data_path='./data/train.csv',lazy=False)
    dev_ds = load_dataset(my_read, data_path='./data/dev.csv',lazy=False)
    test_ds = load_dataset(my_read, data_path='./data/test.csv',lazy=False)
    texts = []
    for data in train_ds:
        texts.append(data["text"])
    for data in dev_ds:
        texts.append(data["text"])
    for data in test_ds:
        texts.append(data["text"])

    # Reads stop words.
    # Stopwords are just for example.
    # It should be updated according to the corpus.
    stopwords = set(["的", "吗", "吧", "呀", "呜", "呢", "呗"])
    # Builds vocab.
    word2idx = build_vocab(texts, stopwords, min_freq=5, unk_token="[UNK]", pad_token="[PAD]")
    vocab = Vocab.from_dict(word2idx, unk_token="[UNK]", pad_token="[PAD]")
    # Saves vocab.
    vocab.to_json(args.vocab_path)

    # Constructs the network.
    network = args.network.lower()
    vocab_size = len(vocab)
    num_classes = 3
    pad_token_id = vocab.to_indices("[PAD]")
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
    model = paddle.Model(model)

    # Reads data and generates mini-batches.
    tokenizer = JiebaTokenizer(vocab)
    trans_fn = partial(convert_example, tokenizer=tokenizer, is_test=False)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab.token_to_idx.get("[PAD]", 0)),  # input_ids
        Stack(dtype="int64"),  # seq len
        Stack(dtype="int64"),  # label
    ): [data for data in fn(samples)]
    train_loader = create_dataloader(
        train_ds, trans_fn=trans_fn, batch_size=args.batch_size, mode="train", batchify_fn=batchify_fn
    )
    dev_loader = create_dataloader(
        dev_ds, trans_fn=trans_fn, batch_size=args.batch_size, mode="validation", batchify_fn=batchify_fn
    )
    test_loader = create_dataloader(
        test_ds, trans_fn=trans_fn, batch_size=args.batch_size, mode="test", batchify_fn=batchify_fn
    )

    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr)

    # Defines loss and metric.
    criterion = paddle.nn.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()
    # auc = paddle.metric.Auc()
    model.prepare(optimizer, criterion, metrics=metric)

    # Loads pre-trained parameters.
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    # Starts training and evaluating.
    # callback = paddle.callbacks.ProgBarLogger(log_freq=10, verbose=3)
    visualdl_callback = paddle.callbacks.VisualDL(log_dir='visualdl_log_dir/' + network)
    earlystopping_callback = paddle.callbacks.EarlyStopping(
                'acc',
                mode='auto',
                patience=1,
                verbose=1,
                min_delta=0,
                baseline=0,
                save_best_model=True)
    model.fit(train_loader, dev_loader, epochs=args.epochs, save_dir=args.save_dir, callbacks=[visualdl_callback, earlystopping_callback])
    test_result = model.evaluate(test_loader, 16,callbacks=[visualdl_callback, earlystopping_callback])
    print(test_result)
    model.save(path='model/' + network)