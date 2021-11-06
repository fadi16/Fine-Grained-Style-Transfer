import os
import pickle
import tensorflow as tf
import nltk
from vae import VAE, VAE_DP, VAE_util

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

w2id, id2w = pickle.load(open('shakespearian-corpus/w2id_id2w.pkl', 'rb'))


def idx2str(s):
    return ' '.join([id2w[idx] for idx in s])


def str2idx(idx):
    idx = idx.strip()
    return [w2id[idxx] for idxx in idx.split()]


def pad(x, pid, move_go=False):
    max_length = 30
    x = [k[:max_length] for k in x]
    if move_go:
        length_list = [len(k) - 1 for k in x]
    else:
        length_list = [len(k) for k in x]
    max_length = max(length_list)
    pad_x = []
    for k in x:
        if move_go:
            pad_k = k[1:] + [pid, ] * (max_length - len(k[1:]))
        else:
            pad_k = k + [pid, ] * (max_length - len(k))
        pad_x.append(pad_k)
    return pad_x, length_list


def pad_maxlength(x, pid, move_go=False):
    max_length = 30
    if move_go:
        length_list = [len(k) - 1 for k in x]
    else:
        length_list = [min(len(k), max_length) for k in x]

    pad_x = []
    for k in x:
        if move_go:
            pad_k = k[1:] + [pid, ] * (max_length - len(k[1:]))
        else:
            pad_k = k[:max_length] + [pid, ] * (max_length - len(k))
        pad_x.append(pad_k)
    return pad_x, length_list


def word_overlap_edit(s1, s2):
    t1 = set(s1.split())
    t2 = set(s2.split())
    word_overlap = float(len(t1 & t2)) / len(t1 | t2)
    edit_distance = 1 - float(nltk.edit_distance(s1.split(), s2.split())) / max(len(s1.split()), len(s2.split()))
    return word_overlap, edit_distance


def pad(x, pid, move_go=False):
    x = [k[:30] for k in x]
    if move_go:
        length_list = [len(k) - 1 for k in x]
    else:
        length_list = [len(k) for k in x]
    max_length = max(length_list)
    pad_x = []
    for k in x:
        if move_go:
            pad_k = k[1:] + [pid, ] * (max_length - len(k[1:]))
        else:
            pad_k = k + [pid, ] * (max_length - len(k))
        pad_x.append(pad_k)
    return pad_x, length_list


def pad_maxlength(x, pid, move_go=False):
    max_length = 30
    if move_go:
        length_list = [len(k) - 1 for k in x]
    else:
        length_list = [min(len(k), max_length) for k in x]

    pad_x = []
    for k in x:
        if move_go:
            pad_k = k[1:] + [pid, ] * (max_length - len(k[1:]))
        else:
            pad_k = k[:max_length] + [pid, ] * (max_length - len(k))
        pad_x.append(pad_k)
    return pad_x, length_list


##########################
## Setting session params
##########################

# setting session config
tf.logging.set_verbosity(tf.logging.INFO)
# sess_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess_conf = tf.ConfigProto(log_device_placement=True)

########################
## Data
#######################
w2id, id2w = pickle.load(open('shakespearian-corpus/w2id_id2w.pkl', 'rb'))
Y_train, C_train = pickle.load(open('shakespearian-corpus/train_C.pkl', 'rb'))
Y_dev, C_dev = pickle.load(open('shakespearian-corpus/valid_C.pkl', 'rb'))
Y_test, C_test = pickle.load(open('shakespearian-corpus/test_C.pkl', 'rb'))

X_train = [x[:-1] for x in Y_train]
X_dev = [x[:-1] for x in Y_dev]
X_test = [x[:-1] for x in Y_test]

################################
##  Set Parameters for training
################################
BATCH_SIZE = 200
NUM_EPOCH = 35
is_shuffle = False
Latent_weight = 0.4
Model_basic_name = 'VAE-All'
train_dir = 'model/shakespeare-to-modern/VAE/' + Model_basic_name
vae_dp = VAE_DP(None,
                None,
                None,
                w2id,
                w2id,
                BATCH_SIZE,
                test_data=(X_train, Y_train, C_train, X_dev, Y_dev, C_dev),
                n_epoch=NUM_EPOCH,
                is_shuffle=is_shuffle)

is_training = True

############################
##  Training
###########################
is_training = False
if is_training:
    g = tf.Graph()
    sess = tf.Session(graph=g, config=sess_conf)
    with sess.as_default():
        with sess.graph.as_default():
            model = VAE(
                dp=vae_dp,
                rnn_size=512,
                n_layers=1,
                encoder_embedding_dim=128,
                decoder_embedding_dim=128,
                cell_type='lstm',
                latent_dim=512,
                beta_decay_period=10,
                beta_decay_offset=5,
                latent_weight=Latent_weight,
                bow_size=400,
                is_inference=False,
                num_classes=2,
                max_infer_length=20,
                # att_type='B',
                beam_width=10,
                Lambda=0.9,
                gamma=10.0,
                residual=False,
                sess=sess
            )
            # print(len(tf.global_variables()))

    util = VAE_util(dp=vae_dp, model=model)
    # model.restore('model/shakespeare-to-modern/VAE/model-35')
    util.fit(train_dir=train_dir, is_bleu=False)
