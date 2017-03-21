import data.data_utils as data_utils
import models.memn2n as memn2n

from sklearn import metrics
import numpy as np

import argparse

import tensorflow as tf


DATA_DIR = 'data/dialog-bAbI-tasks/'
BATCH_SIZE = 16
CKPT_DIR= 'ckpt/'

'''
    dictionary of models
        select model from here
'''
model = {
        'memn2n' : memn2n.MemN2NDialog
        }# add models, as you implement


'''
    run prediction on dataset

'''
def batch_predict(S,Q,n, batch_size):
    preds = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        s = S[start:end]
        q = Q[start:end]
        pred = model.predict(s, q)
        preds += list(pred)
    return preds


'''
    parse arguments

'''
def parse_args():
    parser = argparse.ArgumentParser(
            description='Train Model for Goal Oriented Dialog Task : bAbI(6)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--infer', action='store_true',
                        help='perform inference')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    parser.add_argument('--task_id', required=False, type=int, default=1,
                        help='Task Id in bAbI (6) tasks {1-6}')
    parser.add_argument('--batch_size', required=False, type=int, default=16,
                        help='you know what batch size means!')
    parser.add_argument('--epochs', required=False, type=int, default=200,
                        help='num iteration of training over train set')
    parser.add_argument('--eval_interval', required=False, type=int, default=5,
                        help='num iteration of training over train set')
    parser.add_argument('--log_file', required=False, type=str, default='log.txt',
                        help='enter the name of the log file')
    args = vars(parser.parse_args())
    return args


def interactive(model, idx2candid, w2idx, sentence_size, batch_size, n_cand, memory_size):
    context=[]
    u=None
    r=None
    nid=1
    while True:
        line=input('--> ').strip().lower()
        if line=='exit':
            break
        if line=='restart':
            context=[]
            nid=1
            print("clear memory")
            continue
        u=data_utils.tokenize(line)
        data=[(context,u,-1)]
        s, q, a = data_utils.vectorize_data(data, w2idx, sentence_size, 1, n_cand, memory_size)
        preds = model.predict(s,q)
        r = idx2candid[preds[0]]
        print(r)
        r = data_utils.tokenize(r)
        u.append('$u')
        u.append('#'+str(nid))
        r.append('$r')
        r.append('#'+str(nid))
        context.append(u)
        context.append(r)
        nid+=1


if __name__ == '__main__':
    # get user arguments
    args = parse_args()
    # get candidates (restaurants)
    print('task id', args['task_id'])
    candidates, candid2idx, idx2candid = data_utils.load_candidates(task_id= args['task_id'],
                                                candidates_f= DATA_DIR + 'dialog-babi-candidates.txt')
    # get data
    train, test, val = data_utils.load_dialog_task(
            data_dir= DATA_DIR, 
            task_id= args['task_id'],
            candid_dic= candid2idx, 
            isOOV= False)
    ##
    # get metadata
    metadata = data_utils.build_vocab(train + test + val, candidates)
    # gather information from metadata
    sentence_size = metadata['sentence_size']
    w2idx = metadata['w2idx']
    idx2w = metadata['idx2w']
    memory_size = metadata['memory_size']
    vocab_size = metadata['vocab_size']
    n_cand = metadata['n_cand']
    candidate_sentence_size = metadata['candidate_sentence_size']
    # vectorize candidates
    candidates_vec = data_utils.vectorize_candidates(candidates, w2idx, candidate_sentence_size)
    #
    # create model
    model = model['memn2n']( # why?
                batch_size= BATCH_SIZE,
                vocab_size= vocab_size, 
                candidates_size= n_cand, 
                sentence_size= sentence_size, 
                embedding_size= 20, 
                candidates_vec= candidates_vec, 
                hops= 3
            )
    # gather data in batches
    train, val, test, batches = data_utils.get_batches(train, val, test, metadata, batch_size=BATCH_SIZE)

    if args['train']:
        # training starts here
        epochs = args['epochs']
        eval_interval = args['eval_interval']
        #
        # training and evaluation loop
        print('\n>> Training started!\n')
        # write log to file
        log_handle = open('log/' + args['log_file'], 'w')
        cost_total = 0.
        best_validation_accuracy = 0.
        for i in range(epochs+1):

            for start, end in batches:
                s = train['s'][start:end]
                q = train['q'][start:end]
                a = train['a'][start:end]
                cost_total += model.batch_fit(s, q, a)
            
            if i%eval_interval == 0 and i:
                train_preds = batch_predict(train['s'], train['q'], len(train['s']), batch_size=BATCH_SIZE)
                val_preds = batch_predict(val['s'], val['q'], len(val['s']), batch_size=BATCH_SIZE)
                train_acc = metrics.accuracy_score(np.array(train_preds), train['a'])
                val_acc = metrics.accuracy_score(val_preds, val['a'])
                print('Epoch[{}] : <ACCURACY>\n\ttraining : {} \n\tvalidation : {}'.
                     format(i, train_acc, val_acc))
                log_handle.write('{} {} {} {}\n'.format(i, train_acc, val_acc, 
                    cost_total/(eval_interval*len(batches))))
                cost_total = 0. # empty cost
                #
                # save the best model, to disk
                if val_acc > best_validation_accuracy:
                    best_validation_accuracy = val_acc
                    model.saver.save(model._sess, CKPT_DIR + '{}/memn2n_model.ckpt'.format(args['task_id']), 
                            global_step=i)
        # close file
        log_handle.close()

    else: # inference
        #print('\nDude! where\'s the inference? \nNot implemented yet. Come back!')
        #
        #  restore checkpoint
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR + args['task_id'])
        if ckpt and ckpt.model_checkpoint_path:
            print('\n>> restoring checkpoint from', ckpt.model_checkpoint_path)
            model.saver.restore(model._sess, ckpt.model_checkpoint_path)
        # start interactive session
        interactive(model, idx2candid, w2idx, sentence_size, BATCH_SIZE, n_cand, memory_size)
