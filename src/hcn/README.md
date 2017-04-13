# Hybrid Code Networks

![](https://raw.githubusercontent.com/voicy-ai/DialogStateTracking/master/images/hcn-block-diagram.png)


## Setup

```bash
# install prerequisites
sudo -H pip3 install -r requirements.txt
cd data/ # we are inside 'src/hcn/data'
bash download.sh
# See Training.ipynb for training and prediction details
```

### TODO

- [x] Organize trian set as a list of dialogues
	- [x] Maintain entity state, action mask for each dialogue
