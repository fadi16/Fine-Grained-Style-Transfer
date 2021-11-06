
import pickle
from textCNN import *
from vae import VAE, VAE_DP
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

def cal_F1(predict, idx, lengths):
    cnt = 0.001
    P = 0.0
    R = 0.0
    pred_idx = np.argsort(predict)[::-1][:lengths]
    #print(pred_idx, idx)
    for t in pred_idx:
        if t in idx:
            cnt += 1
    P = cnt / len(set(idx))
    #if P > 1.0:
    #    print(cnt, np.sum(predict>0.5))
    R = cnt / len(set(idx))
    F1 = (2* P*R) / (P+R)
    return P, R ,F1

########################
## Data
#######################
w2id, id2w = pickle.load(open('shakespearian-corpus/w2id_id2w.pkl','rb'))
Y_train, C_train = pickle.load(open('shakespearian-corpus/train_C.pkl','rb'))
Y_dev, C_dev = pickle.load(open('shakespearian-corpus/valid_C.pkl','rb'))
Y_test, C_test = pickle.load(open('shakespearian-corpus/test_C.pkl','rb'))

X_train = [x[:-1] for x in Y_train]
X_dev = [x[:-1] for x in Y_dev]
X_test = [x[:-1] for x in Y_test]

###########################
##  Parameters
###########################
BATCH_SIZE = 256
NUM_EPOCH = 30
MAX_LENGTH = 200
is_shuffle = False
Latent_weight = 0.4
# setting session config
tf.logging.set_verbosity(tf.logging.INFO)
sess_conf = tf.ConfigProto(log_device_placement=True)

##############################
## Init Test model
##############################
vae_dp = VAE_DP(None, None, None, w2id, w2id, BATCH_SIZE, test_data=(X_train, Y_train, C_train, X_dev, Y_dev, C_dev), n_epoch=NUM_EPOCH, is_shuffle=is_shuffle)

g = tf.Graph()
sess = tf.Session(graph=g, config=sess_conf)
with sess.as_default():
    with sess.graph.as_default():
        model = VAE(
            dp = vae_dp,
            rnn_size = 512,
            n_layers = 1,
            encoder_embedding_dim = 128,
            decoder_embedding_dim = 128,
            cell_type = 'lstm',
            latent_dim = 512,
            beta_decay_period = 10,
            beta_decay_offset = 5,
            latent_weight = Latent_weight,
            bow_size = 400,
            is_inference = True,
            num_classes = 2,
            max_infer_length = 20,
            #att_type='B',
            beam_width=10,
            Lambda = 0.9,
            gamma = 10.0,
            residual = False,
            sess=sess
        )

is_training_3 = True

if is_training_3:
    model.restore('model/shakespeare-to-modern/VAE/attribute-predictor-separately/model-10')
    n_epoch = 10
    batch_size = 200
    out_dir = 'model/shakespeare-to-modern/VAE/content-predictor-separately/'
    random.shuffle(X_train)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing to %s" % out_dir)
    checkpoint_prefix = os.path.join(out_dir, "model")

    X_train = np.array(X_train)
    for e in range(n_epoch):
        p_list = []
        r_list = []
        f1_list = []
        loss_list = []
        r = np.random.permutation(len(X_train))
        X_train = X_train[r]

        if e == 0:
            p_test_list = []
            r_test_list = []
            f1_test_list = []
            for b in range(0, len(X_test) - len(X_test) % batch_size, batch_size):
                X_batch = X_test[b: b + batch_size]
                z_mu = model.idxTozmu_batch(X_batch)
                o_str, o_idx = model.zmuTox_batch(z_mu)
                pad_x, length_list = pad(o_idx, w2id['<PAD>'], move_go=False)

                loss, logits = model.sess.run([model.avg_bow_loss, model.bow_logits],
                                              {model.z: z_mu, model.Y: pad_x,
                                               model.Y_seq_len: length_list,
                                               model.output_keep_prob: 1.0,
                                               model.input_keep_prob: 1.0})
                for j, x in enumerate(o_idx):
                    p, r, f1 = cal_F1(logits[j], x, len(x))
                    p_test_list.append(p)
                    r_test_list.append(r)
                    f1_test_list.append(f1)
            print('%d/%d: Test: P %.2f | R %.2f | F1 %.2f' % (
            e, n_epoch, np.mean(p_test_list) * 100, np.mean(r_test_list) * 100, np.mean(f1_test_list) * 100))

        ## Train
        for b in range(0, len(X_train) - len(X_train) % batch_size, batch_size):
            X_batch = X_train[b: b + batch_size]
            z_mu = model.idxTozmu_batch(X_batch)
            o_str, o_idx = model.zmuTox_batch(z_mu)
            pad_x, length_list = pad(o_idx, w2id['<PAD>'], move_go=False)
            _, loss, logits = model.sess.run([model.bow_op, model.avg_bow_loss, model.bow_logits],
                                             {model.z: z_mu, model.Y: pad_x,
                                              model.Y_seq_len: length_list,
                                              model.output_keep_prob: model._output_keep_prob,
                                              model.input_keep_prob: model._input_keep_prob})

            for j, x in enumerate(o_idx):
                p, r, f1 = cal_F1(logits[j], x, len(x))
                p_list.append(p)
                r_list.append(r)
                f1_list.append(f1)
            loss_list.append(loss)

            z_mu = np.random.randn(batch_size, 512)
            o_str, o_idx = model.zmuTox_batch(z_mu)
            pad_x, length_list = pad(o_idx, w2id['<PAD>'], move_go=False)
            _, loss, logits = model.sess.run([model.bow_op, model.avg_bow_loss, model.bow_logits],
                                             {model.z: z_mu, model.Y: pad_x,
                                              model.Y_seq_len: length_list,
                                              model.output_keep_prob: model._output_keep_prob,
                                              model.input_keep_prob: model._input_keep_prob})

            for j, x in enumerate(o_idx):
                p, r, f1 = cal_F1(logits[j], x, len(x))
                p_list.append(p)
                r_list.append(r)
                f1_list.append(f1)
            loss_list.append(loss)
            if (b % (50 * batch_size)) == 0:
                print("epoch %d/%d batch %d/%d: loss %.5f | P %.2f | R %.2f | F1 %.2f" % (
                e, n_epoch, int(b / batch_size),
                int((len(X_train) - len(X_train) % batch_size) / batch_size),
                np.mean(loss_list), np.mean(p_list) * 100, np.mean(r_list) * 100, np.mean(f1_list) * 100))

        ## Test
        p_test_list = []
        r_test_list = []
        f1_test_list = []
        for b in range(0, len(X_test) - len(X_test) % batch_size, batch_size):
            X_batch = X_test[b: b + batch_size]
            z_mu = model.idxTozmu_batch(X_batch)
            o_str, o_idx = model.zmuTox_batch(z_mu)
            pad_x, length_list = pad(o_idx, w2id['<PAD>'], move_go=False)

            loss, logits = model.sess.run([model.avg_bow_loss, model.bow_logits],
                                          {model.z: z_mu, model.Y: pad_x,
                                           model.Y_seq_len: length_list,
                                           model.output_keep_prob: 1.0,
                                           model.input_keep_prob: 1.0})
            for j, x in enumerate(o_idx):
                p, r, f1 = cal_F1(logits[j], x, len(x))
                p_test_list.append(p)
                r_test_list.append(r)
                f1_test_list.append(f1)
        print('%d/%d: Test: P %.2f | R %.2f | F1 %.2f' % (
        e + 1, n_epoch, np.mean(p_test_list) * 100, np.mean(r_test_list) * 100, np.mean(f1_test_list) * 100))
        print('%d/%d: Train: P %.2f | R %.2f | F1 %.2f' % (
        e + 1, n_epoch, np.mean(p_list) * 100, np.mean(r_list) * 100, np.mean(f1_list) * 100))
        path = model.saver.save(model.sess, checkpoint_prefix, global_step=e + 1)
        print("Saved model checkpoint to %s" % path)