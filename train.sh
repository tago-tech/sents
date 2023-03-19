#!/bin/bash

echo "start train [cnn] model"
python train.py --vocab_path='./vocab.json' \
    --device=cpu \
    --network=cnn \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir='./checkpoints/cnn/checkpoints' >> output/cnn.train.log.`date +%Y-%m-%d-%h-%m`
cp -r checkpoints/cnn/checkpoints/final.* output/cnn/checkpoints/
rm -r checkpoints/cnn/checkpoints/*

echo "start train [lstm] model"
python train.py --vocab_path='./vocab.json' \
    --device=cpu \
    --network=lstm \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir='./checkpoints/lstm/checkpoints'  >> output/lstm.train.log.`date +%Y-%m-%d-%h-%hh-%mm`
cp -r checkpoints/lstm/checkpoints/final.* output/lstm/checkpoints/ 
rm -r checkpoints/lstm/checkpoints/*

echo "start train [bilstm] model"
python train.py --vocab_path='./vocab.json' \
    --device=cpu \
    --network=bilstm \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir='./checkpoints/bilstm/checkpoints' >> output/bilstm.train.log.`date +%Y-%m-%d-%h-%m-%hh-%mm`
cp -r checkpoints/bilstm/checkpoints/final.* output/bilstm/checkpoints/
rm -r checkpoints/bilstm/checkpoints/*

echo "start train [rnn] model"
python train.py --vocab_path='./vocab.json' \
    --device=cpu \
    --network=rnn \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir='./checkpoints/rnn/checkpoints' >> output/rnn.train.log.`date +%Y-%m-%d-%h-%m-%hh-%mm`
cp -r checkpoints/rnn/checkpoints/final.* output/rnn/checkpoints/
rm -r checkpoints/rnn/checkpoints/*

echo "start train [birnn] model"
python train.py --vocab_path='./vocab.json' \
    --device=cpu \
    --network=birnn \
    --lr=5e-4 \
    --batch_size=64 \
    --epochs=10 \
    --save_dir='./checkpoints/birnn/checkpoints' >> output/birnn.train.log.`date +%Y-%m-%d-%h-%m-%hh-%mm`
cp -r checkpoints/birnn/checkpoints/final.* output/birnn/checkpoints/
rm -r checkpoints/birnn/checkpoints/*