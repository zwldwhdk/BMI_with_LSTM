import time
import tensorflow as tf
import numpy as np
import scipy.signal as sig
import scipy.io as sio

tf.random.set_random_seed(777)

import meg_CRNN
tf.flags.DEFINE_integer('epoch', 70, 'number of training epoch')
tf.flags.DEFINE_integer('batch', 1803, 'number of training epoch')
tf.flags.DEFINE_float('learning_rate', 2e-4, 'learning_rate; eta')
FLAGS = tf.flags.FLAGS
epoch = FLAGS.epoch
batch = FLAGS.batch
learning_rate = FLAGS.learning_rate

### ADDING VALIDATION PLZ
def train(train, path):
    '''
    train function
    :param train: training tensor
    :param path:  save_path for saving weight values
    :return: None
    '''
    iterator     = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
    next_element = iterator.get_next()
    init_train   = iterator.make_initializer(train)

    with tf.Session() as sess:

        # 4 models for assemble validation
        num_models  = 4
        crnn_models = [meg_CRNN.crnn(
            sess=sess, learning_rate=learning_rate, name='crnn_model{}'.format(m)
        ) for m in range(num_models)]

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print('$ Learning STARTED!')
        start = time.time()
        for e in range(epoch):
            sess.run(init_train)
            step     = 1 # iterator counter
            avg_loss = np.zeros(num_models)
            #avg_corr = np.zeros(num_models)
            init     = time.time()
            while step*batch < (batch*192)+1:
                batch_x, batch_y = sess.run([next_element['X'], next_element['Y']])
                # assemble models
                for idx, crnn in enumerate(crnn_models):
                    loss, _opt     = crnn.loss_function(batch_x, batch_y)
                    avg_loss[idx] += loss/192

                step += 1 # update counter
            off = time.time()
            check = off - init
            epoch_loss = np.round(np.mean(avg_loss), 5)
            print('-------------------------------------------------------------------------')
            print('$ [{}]'.format(e + 1), 'LOSS[{}]:'.format(epoch_loss), np.round(avg_loss, 5))
            print('$time lapse: {}min, {}sec'.format(int((check // 60)),
                                                     np.round((check % 60), 1)))

        print('=========================================================================')
        saver = tf.train.Saver()
        saver.save(sess=sess, save_path=path)
        print("$ SAVE TO", path)
        sess.close()
        # Whole Time checkup;
        end = time.time()
        lapse = end - start
        print('$ TOTAL TIME: {}hour {}min {}sec'.format(int((lapse // 3600)), int((lapse % 3600) // 60),
                                                        np.round((lapse % 3600) % 60, 1)))
        print('$ LEARNING FINISHED')
        print('=========================================================================')

def test(test, path, save_name):

    # Making batch iterator
    iterator = tf.data.Iterator.from_structure(test.output_types, test.output_shapes)
    next_element = iterator.get_next()
    init_test = iterator.make_initializer(test)

    with tf.Session() as sess:

        num_models  = 4
        crnn_models = [meg_CRNN.crnn(
            sess=sess, learning_rate=learning_rate, name='crnn_model{}'.format(m)
        ) for m in range(num_models)]

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=path)
        print('$ RESTORE FROM: ', path)
        print('-------------------------------------------------------------------------')
        print('$ TESTING STARTED')
        print('-------------------------------------------------------------------------')

        sess.run(init_test)
        step            = 1
        avg_fold_coeff  = np.zeros(3)

        prediction_list   = []
        label_list        = []
        ensemble_idx_list = []
        while step * batch < (batch*48 + 1):

            batch_x, batch_y = sess.run([next_element['X'], next_element['Y']])

            ensemble_coeff   = np.zeros([num_models, 3], dtype=np.float32)
            ensemble_list    = []
            label_list.append(batch_y)
            for idx, crnn in enumerate(crnn_models):
                prediction = crnn.predict(x_data=batch_x)
                coeff = crnn.validation(x_data=batch_x, y_data=batch_y)
                print('$ crnn_model_{} = {}'.format(idx+1, np.round(coeff, 3)))
                # smoothing output?
                ensemble_list.append(prediction)
                ensemble_coeff[idx] = coeff
            # ensembling; choosing best coefficient from 5 prediction value
            ensembling_idx = np.argmax(ensemble_coeff, axis=0)
            ensemble_idx_list.append(ensembling_idx)
            x_corr = round(ensemble_coeff[ensembling_idx[0], 0], 3)
            y_corr = round(ensemble_coeff[ensembling_idx[1], 1], 3)
            z_corr = round(ensemble_coeff[ensembling_idx[2], 2], 3)
            summary_coeff   = np.array([x_corr, y_corr, z_corr])
            avg_fold_coeff += summary_coeff/48
            print('$ [{}] ensembling_coefficient: {}'.format(step, summary_coeff))
            print('-------------------------------------------------------------------------')

            # Ensembling Prediction
            x_pred = ensemble_list[ensembling_idx[0]][:, [0]]
            y_pred = ensemble_list[ensembling_idx[1]][:, [1]]
            z_pred = ensemble_list[ensembling_idx[2]][:, [2]]
            summary_prediciton = np.concatenate([x_pred, y_pred, z_pred], axis=1)
            prediction_list.append(summary_prediciton)

            step += 1

        sess.close()
        # Print average coefficient values
        print('$ Coefficient results(R^2): {}'.format(np.round(avg_fold_coeff, 3)))
        print('$ Coefficient mean: {}'.format(np.round(np.mean(avg_fold_coeff), 3)))
        print('=========================================================================')

        # Summary prediciotns and save .mat
        prediction_list = np.reshape(prediction_list, [-1, 3])  # 3D-to-2D dimension
        label_list = np.reshape(label_list, [-1, 3])

        sio.savemat(save_name, {'Y_pred': prediction_list, 'Y_real': label_list})
        print('$ SAVE results to ', save_name)
        print('=========================================================================')
        return prediction_list, label_list
