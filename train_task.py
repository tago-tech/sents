from celery import Celery
import os

broker = 'redis://127.0.0.1:6379/1'
backend = 'redis://127.0.0.1:6379/2'

app = Celery(__name__, broker=broker, backend=backend)

base_train_cmd = "python train.py --vocab_path='./vocab.json' --device=cpu --network={} --lr=5e-4 --batch_size=16 --epochs=10 --save_dir='./checkpoints/{}/checkpoints' >> output/{}.train.log.`date +%Y-%m-%d-%h-%m`"

@app.task
def train_cnn():
    os.system(base_train_cmd.format("cnn"))
    return None

@app.task
def train_lstm():
    os.system(base_train_cmd.format("lstm"))
    return None

@app.task
def train_bilstm():
    os.system(base_train_cmd.format("bilstm"))
    return None

@app.task
def train_bilstm_attn():
    os.system(base_train_cmd.format("bilstm_attn"))
    return None

@app.task
def train_birnn():
    os.system(base_train_cmd.format("birnn"))
    return None