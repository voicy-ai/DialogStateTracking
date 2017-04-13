# Hybrid Code Networks

![](https://raw.githubusercontent.com/voicy-ai/DialogStateTracking/master/images/hcn-block-diagram.png)


## Setup


```bash
# install prerequisites
sudo -H pip3 install -r requirements.txt
cd data/ # we are inside 'src/hcn/data'
bash download.sh
```

## Execution


```bash
# training
python3 train.py
# training stops when accuracy on dev set becomes > 0.99
#  trained model is saved to ckpt/

# interaction 
python3 interact.py
# checkpoint from ckpt/ is loaded
#  start interaction
```
