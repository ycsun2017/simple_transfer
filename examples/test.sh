

python encode_example.py --envname grid_n4_linearf8_cost0.1 --types source \
        --size 4 --env source --episodes 500 --epoch 1 --cost 0.1 --encode-size 8 --reward-coeff 1 \

python encode_example.py --envname grid_n4_linearf8_cost0.1 --types encoder --load-from source \
        --size 4 --env target --episodes 100 --epoch 10 --cost 0.1 --encode-size 8 --reward-coeff 1 --lr 5e-5\

python encode_example.py --envname grid_n4_linearf8_cost0.1 --load-from source --types single auxiliary transfer regularize \
        --size 4 --env target --episodes 500 --epoch 1 --cost 0.1 --encode-size 8 --reward-coeff 1 \
        --lr 0.05 --instances 5

