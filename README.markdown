# End-to-End Neural Networks for Goal Oriented Dialog

The objective of this project is to explore neural network architectures for question answering and goal oriented dialog systems. We will train on [(6) dialog bAbI tasks](https://research.fb.com/downloads/babi/). Tensorflow is used for building the models. I have listed some of the influential/interesting papers below. If I have missed something, let me know.

## Setup

```bash
# python2 is not supported
sudo -H pip3 install -r requirements.txt
# if this doesn't work, raise an issue
```

## Learning End-to-End Goal-Oriented Dialog

![](https://camo.githubusercontent.com/ba1c7dbbccc5dd51d4a76cc6ef849bca65a9bf4d/687474703a2f2f692e696d6775722e636f6d2f6e7638394a4c632e706e67)

```bash
# run main.py without arguments, for usage information
#  usage: main.py [-h] (-i | -t) [--task_id TASK_ID] [--batch_size BATCH_SIZE]
#               [--epochs EPOCHS] [--eval_interval EVAL_INTERVAL]
#               [--log_file LOG_FILE]
#  main.py: error: one of the arguments -i/--infer -t/--train is required
python3 main.py --train --task_id=3 --log_file=log.task3.txt
```

### Results

Task  |  Training Accuracy  |  Validation Accuracy  |
------|---------------------|-----------------------|
1     |  100	              |  99.7		            |
2     |  100                |  100		            |
3     |  100               |  74.71		            |
4     |  100               |  56.67		            |
5     |  100               |  98.42		            |
6     |  76.61               |  47.08		            |

![](plots/collage.png)

## Papers

1. [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/abs/1605.07683), [review](https://openreview.net/forum?id=S1Bb3D5gg)
2. [A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue](https://www.semanticscholar.org/paper/A-Copy-Augmented-Sequence-to-Sequence-Architecture-Eric-Manning/3931e8406468948e8979a24454c05d448c46815e)
3. [Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning](https://www.semanticscholar.org/paper/Hybrid-Code-Networks-practical-and-efficient-end-Williams-Asadi/0fbc76d570d68e6bd3c9701c6fcb2efa91659eb3)
4. [Gated End-to-End Memory Networks](https://www.semanticscholar.org/paper/Gated-End-to-End-Memory-Networks-Perez-Liu/46977c2e7a812e37f32eb05ba6ad16e03ee52906)
5. [Query-Reduction Networks for Question Answering](https://arxiv.org/abs/1606.04582)
6. [Ask Me Even More: Dynamic Memory Tensor Networks](https://arxiv.org/abs/1703.03939)

## Jargons

- OOV : Out Of Vocabulary
- hops : number of memory lookups

## Credits

- Tweaked [vyraun](https://github.com/vyraun/chatbot-MemN2N-tensorflow)'s [chatbot-MemN2N-tensorflow](https://github.com/vyraun/chatbot-MemN2N-tensorflow) repository, for using Memory Networks for bAbI dialog tasks
