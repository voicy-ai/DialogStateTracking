import data.data_utils as data_utils
import models.memn2n as memn2n

from sklearn import metrics
import numpy as np

import argparse

import tensorflow as tf

import pickle as pkl
import sys


DATA_DIR = 'data/dialog-bAbI-tasks/'
P_DATA_DIR = 'data/processed/'
BATCH_SIZE = 16
CKPT_DIR= 'ckpt/'

'''
    dictionary of models
        select model from here
model = {
        'memn2n' : memn2n.MemN2NDialog
        }# add models, as you implement

'''

'''
    run prediction on dataset

'''
def batch_predict(model, S,Q,n, batch_size):
    preds = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        s = S[start:end]
        q = Q[start:end]
        pred = model.predict(s, q)
        preds += list(pred)
    return preds


'''
    preprocess data

'''
def prepare_data(args, task_id):
    # get candidates (restaurants)
    candidates, candid2idx, idx2candid = data_utils.load_candidates(task_id=task_id,
                                                candidates_f= DATA_DIR + 'dialog-babi-candidates.txt')
    # get data
    train, test, val = data_utils.load_dialog_task(
            data_dir= DATA_DIR, 
            task_id= task_id,
            candid_dic= candid2idx, 
            isOOV= False)
    ##
    # get metadata
    metadata = data_utils.build_vocab(train + test + val, candidates)

    ###
    # write data to file
    data_ = {
            'candidates' : candidates,
            'train' : train,
            'test' : test,
            'val' : val
            }
    with open(P_DATA_DIR + str(task_id) + '.data.pkl', 'wb') as f:
        pkl.dump(data_, f)

    ### 
    # save metadata to disk
    metadata['candid2idx'] = candid2idx
    metadata['idx2candid'] = idx2candid
    
    with open(P_DATA_DIR + str(task_id) + '.metadata.pkl', 'wb') as f:
        pkl.dump(metadata, f)


'''
    parse arguments

'''
def parse_args(args):
    parser = argparse.ArgumentParser(
            description='Train Model for Goal Oriented Dialog Task : bAbI(6)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--infer', action='store_true',
                        help='perform inference in an interactive session')
    group.add_argument('-t', '--train', action='store_true',
                        help='train model')
    group.add_argument('-d', '--prep_data', action='store_true',
                        help='prepare data')
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
    args = vars(parser.parse_args(args))
    return args


class InteractiveSession():

    def __init__(self, model, idx2candid, w2idx, n_cand, memory_size):
        self.context = []
        self.u = None
        self.r = None
        self.nid = 1
        self.model = model
        self.idx2candid = idx2candid
        self.w2idx = w2idx
        self.n_cand = model._candidates_size
        self.memory_size = memory_size
        self.model = model

    def reply(self, msg):
        line = msg.strip().lower()
        if line == 'clear':
            self.context = []
            self.nid = 1
            reply_msg = 'memory cleared!'
        else:
            u = data_utils.tokenize(line)
            data = [(self.context, u, -1)]
            s, q, a = data_utils.vectorize_data(data, 
                    self.w2idx, 
                    self.model._sentence_size, 
                    1, 
                    self.n_cand, 
                    self.memory_size)
            preds = self.model.predict(s,q)
            r = self.idx2candid[preds[0]]
            reply_msg = r
            r = data_utils.tokenize(r)
            u.append('$u')
            u.append('#'+str(self.nid))
            r.append('$r')
            r.append('#'+str(self.nid))
            self.context.append(u)
            self.context.append(r)
            self.nid+=1
        return reply_msg

                   

def main(args):
    # parse args
    args = parse_args(args)

    # prepare data
    if args['prep_data']:
        print('\n>> Preparing Data\n')
        for i in range(1,7):
            print(' TASK#{}\n'.format(i))
            prepare_data(args, task_id=i)
        sys.exit()

    # ELSE
    # read data and metadata from pickled files
    with open(P_DATA_DIR + str(args['task_id']) + '.metadata.pkl', 'rb') as f:
        metadata = pkl.load(f)
    with open(P_DATA_DIR + str(args['task_id']) + '.data.pkl', 'rb') as f:
        data_ = pkl.load(f)

    # read content of data and metadata
    candidates = data_['candidates']
    candid2idx, idx2candid = metadata['candid2idx'], metadata['idx2candid']

    # get train/test/val data
    train, test, val = data_['train'], data_['test'], data_['val']

    # gather more information from metadata
    sentence_size = metadata['sentence_size']
    w2idx = metadata['w2idx']
    idx2w = metadata['idx2w']
    memory_size = metadata['memory_size']
    vocab_size = metadata['vocab_size']
    n_cand = metadata['n_cand']
    candidate_sentence_size = metadata['candidate_sentence_size']

    # vectorize candidates
    candidates_vec = data_utils.vectorize_candidates(candidates, w2idx, candidate_sentence_size)

    ###
    # create model
    #model = model['memn2n']( # why?
    model = memn2n.MemN2NDialog(
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
        #best_validation_accuracy = 0.
        for i in range(epochs+1):

            for start, end in batches:
                s = train['s'][start:end]
                q = train['q'][start:end]
                a = train['a'][start:end]
                cost_total += model.batch_fit(s, q, a)
            
            if i%eval_interval == 0 and i:
                train_preds = batch_predict(model, train['s'], train['q'], len(train['s']), batch_size=BATCH_SIZE)
                val_preds = batch_predict(model, val['s'], val['q'], len(val['s']), batch_size=BATCH_SIZE)
                train_acc = metrics.accuracy_score(np.array(train_preds), train['a'])
                val_acc = metrics.accuracy_score(val_preds, val['a'])
                print('Epoch[{}] : <ACCURACY>\n\ttraining : {} \n\tvalidation : {}'.
                     format(i, train_acc, val_acc))
                log_handle.write('{} {} {} {}\n'.format(i, train_acc, val_acc, 
                    cost_total/(eval_interval*len(batches))))
                cost_total = 0. # empty cost
                #
                # save the best model, to disk
                #if val_acc > best_validation_accuracy:
                #best_validation_accuracy = val_acc
                model.saver.save(model._sess, CKPT_DIR + '{}/memn2n_model.ckpt'.format(args['task_id']), 
                        global_step=i)
        # close file
        log_handle.close()

    else: # inference
        ###
        # restore checkpoint
        ckpt = tf.train.get_checkpoint_state(CKPT_DIR + str(args['task_id']) )
        if ckpt and ckpt.model_checkpoint_path:
            print('\n>> restoring checkpoint from', ckpt.model_checkpoint_path)
            model.saver.restore(model._sess, ckpt.model_checkpoint_path)
        #  interactive(model, idx2candid, w2idx, sentence_size, BATCH_SIZE, n_cand, memory_size)

        # create an interactive session instance
        isess = InteractiveSession(model, idx2candid, w2idx, n_cand, memory_size)
        
        if args['infer']:
            query = ''
            while query!= 'exit':
                query = input('>> ')
                print('>> ' + isess.reply(query))


# _______MAIN_______
if __name__ == '__main__':
    main(sys.argv[1:])
    #main(['--infer', '--task_id=1'])
