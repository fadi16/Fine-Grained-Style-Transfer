import pickle
from textCNN import *
from VAE_train import VAE, VAE_DP

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
sess_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

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



###########################
##  Init TextCNN
###########################
cnn_dp = TextCNN_DP(X_train, C_train, w2id,  BATCH_SIZE, max_length = MAX_LENGTH, n_epoch=NUM_EPOCH, split_ratio=0.05)

emb_dim = 128
filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]

g_cnn = tf.Graph()
sess_cnn = tf.Session(graph=g_cnn, config=sess_conf)
with sess_cnn.as_default():
    with sess_cnn.graph.as_default():
        D = TextCNN(sess = sess_cnn, dp = cnn_dp, sequence_length=MAX_LENGTH, num_classes=2, vocab_size=len(cnn_dp.id2w),
                          emd_dim = emb_dim, filter_sizes = filter_sizes, num_filters=num_filters,
                          l2_reg_lambda=0.2, dropout_keep_prob=0.75)
        D.sess.run(tf.global_variables_initializer())

######################################
## restoring the CNNText classifier
#####################################
D.restore("model/shakespeare-to-modern/TextCNN/model-30.data-00000-of-00001")

#####################################
##  Retrain Attribute Predictor
#####################################
is_training_2 = True

if is_training_2:
    model.restore('model/shakespeare-to-modern/VAE-All/model-35.data-00000-of-00001')
    n_epoch = 10
    batch_size = 200

    out_dir = 'model/shakespeare-to-modern/VAE-All/'
    random.shuffle(X_train)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Writing to %s" % out_dir)
    checkpoint_prefix = os.path.join(out_dir, "model")

    X_train = np.array(X_train)
    for e in range(n_epoch):
        acc_list = []
        loss_list = []
        r = np.random.permutation(len(X_train))
        X_train = X_train[r]

        if e == 0:
            acc_test_list = []
            for b in range(0, len(X_test) - len(X_test) % batch_size, batch_size):
                X_batch = X_test[b: b + batch_size]
                z_mu = model.idxTozmu_batch(X_batch)
                o_str, o_idx = model.zmuTox_batch(z_mu)
                output = D.batch_infer(o_idx)
                C = np.zeros([len(output), 2])
                for i, o in enumerate(output):
                    C[i][o] = 1

                loss, acc = model.sess.run([model.c_loss, model.accuracy],
                                           {model.z: z_mu, model.C: C,
                                            model.output_keep_prob: 1.0,
                                            model.input_keep_prob: 1.0})
                acc_test_list.append(acc)
            print('%d/%d: Test: %.4f' % (e, n_epoch, np.mean(acc_test_list)))
        ## Train
        for b in range(0, len(X_train) - len(X_train) % batch_size, batch_size):
            X_batch = X_train[b: b + batch_size]
            z_mu = model.idxTozmu_batch(X_batch)
            o_str, o_idx = model.zmuTox_batch(z_mu)

            output = D.batch_infer(o_idx)
            C = np.zeros([len(output), 2])
            for i, o in enumerate(output):
                C[i][o] = 1

            _, loss, acc = model.sess.run([model.predictor_op, model.c_loss, model.accuracy],
                                          {model.z: z_mu, model.C: C,
                                           model.output_keep_prob: model._output_keep_prob,
                                           model.input_keep_prob: model._input_keep_prob})
            loss_list.append(loss)
            acc_list.append(acc)

            z_mu = np.random.randn(batch_size, 512)
            o_str, o_idx = model.zmuTox_batch(z_mu)

            output = D.batch_infer(o_idx)
            C = np.zeros([len(output), 2])
            for i, o in enumerate(output):
                C[i][o] = 1

            _, loss, acc = model.sess.run([model.predictor_op, model.c_loss, model.accuracy],
                                          {model.z: z_mu, model.C: C,
                                           model.output_keep_prob: model._output_keep_prob,
                                           model.input_keep_prob: model._input_keep_prob})
            loss_list.append(loss)
            acc_list.append(acc)
            if (b % (50 * batch_size)) == 0:
                print(e, b / batch_size, np.mean(loss_list), np.mean(acc_list))

        ## Test
        acc_test_list = []
        for b in range(0, len(X_test) - len(X_test) % batch_size, batch_size):
            X_batch = X_test[b: b + batch_size]
            z_mu = model.idxTozmu_batch(X_batch)
            o_str, o_idx = model.zmuTox_batch(z_mu)
            output = D.batch_infer(o_idx)
            C = np.zeros([len(output), 2])
            for i, o in enumerate(output):
                C[i][o] = 1

            loss, acc = model.sess.run([model.c_loss, model.accuracy],
                                       {model.z: z_mu, model.C: C,
                                        model.output_keep_prob: 1.0,
                                        model.input_keep_prob: 1.0})
            acc_test_list.append(acc)
        print('%d/%d: Test: %.4f | Train: %.4f' % (e + 1, n_epoch, np.mean(acc_test_list), np.mean(acc_list)))
        path = model.saver.save(model.sess, checkpoint_prefix, global_step=e + 1)
        print("Saved model checkpoint to %s" % path)