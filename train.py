import data.data_utils as data_utils
import models.memn2n as memn2n

from sklearn import metrics
import numpy as np

DATA_DIR = 'data/dialog-bAbI-tasks/'
BATCH_SIZE = 16

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



if __name__ == '__main__':
    # get candidates (restaurants)
    candidates, candid2idx, idx2candid = data_utils.load_candidates(task_id=1,
                                                candidates_f=DATA_DIR + 'dialog-babi-candidates.txt')
    # get data
    train, test, val = data_utils.load_dialog_task(
            data_dir=DATA_DIR, 
            task_id=1, 
            candid_dic=candid2idx, 
            isOOV=False)
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
    # training starts here
    epochs = 500
    eval_interval = 1
    #
    # training and evaluation loop
    print('\n>> Training started!\n')
    for i in range(epochs+1):
        cost_total = 0
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
