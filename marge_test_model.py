# coding:utf-8
#!/usr/bin/env python
import c3d_model
import c3d_gene_data
import marge_rnn_model
from keras import optimizers
import matplotlib
from keras.models import model_from_json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import keras.backend as K
from sklearn.metrics import classification_report,confusion_matrix
matplotlib.use('Agg')
dim_ordering = K.image_dim_ordering()
print ("[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
        dim_ordering))
backend = dim_ordering

def diagnose(data, verbose=True, label='input', plots=False, backend='tf'):
    # Convolution3D?
    if data.ndim > 2:
        if backend == 'th':
            data = np.transpose(data, (1, 2, 3, 0))
        #else:
        #    data = np.transpose(data, (0, 2, 1, 3))
        min_num_spatial_axes = 10
        max_outputs_to_show = 3
        ndim = data.ndim
        print ("[Info] {}.ndim={}".format(label, ndim))
        print ("[Info] {}.shape={}".format(label, data.shape))
        for d in range(ndim):
            num_this_dim = data.shape[d]
            if num_this_dim >= min_num_spatial_axes: # check for spatial axes
                # just first, center, last indices
                range_this_dim = [0, num_this_dim/2, num_this_dim - 1]
            else:
                # sweep all indices for non-spatial axes
                range_this_dim = range(num_this_dim)
            for i in range_this_dim:
                new_dim = tuple([d] + range(d) + range(d + 1, ndim))
                sliced = np.transpose(data, new_dim)[i, ...]
                print("[Info] {}, dim:{} {}-th slice: "
                      "(min, max, mean, std)=({}, {}, {}, {})".format(
                              label,
                              d, i,
                              np.min(sliced),
                              np.max(sliced),
                              np.mean(sliced),
                              np.std(sliced)))
        if plots:
            # assume (l, h, w, c)-shaped input
            if data.ndim != 4:
                print("[Error] data (shape={}) is not 4-dim. Check data".format(
                        data.shape))
                return
            l, h, w, c = data.shape
            if l >= min_num_spatial_axes or \
                h < min_num_spatial_axes or \
                w < min_num_spatial_axes:
                print("[Error] data (shape={}) does not look like in (l,h,w,c) "
                      "format. Do reshape/transpose.".format(data.shape))
                return
            nrows = int(np.ceil(np.sqrt(data.shape[0])))
            # BGR
            if c == 3:
                for i in range(l):
                    mng = plt.get_current_fig_manager()
                    mng.resize(*mng.window.maxsize())
                    plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                    im = np.squeeze(data[i, ...]).astype(np.float32)
                    im = im[:, :, ::-1] # BGR to RGB
                    # force it to range [0,1]
                    im_min, im_max = im.min(), im.max()
                    if im_max > im_min:
                        im_std = (im - im_min) / (im_max - im_min)
                    else:
                        print ("[Warning] image is constant!")
                        im_std = np.zeros_like(im)
                    plt.imshow(im_std)
                    plt.axis('off')
                    plt.title("{}: t={}".format(label, i))
                plt.show()
                #plt.waitforbuttonpress()
            else:
                for j in range(min(c, max_outputs_to_show)):
                    for i in range(l):
                        mng = plt.get_current_fig_manager()
                        mng.resize(*mng.window.maxsize())
                        plt.subplot(nrows, nrows, i + 1) # doh, one-based!
                        im = np.squeeze(data[i, ...]).astype(np.float32)
                        im = im[:, :, j]
                        # force it to range [0,1]
                        im_min, im_max = im.min(), im.max()
                        if im_max > im_min:
                            im_std = (im - im_min) / (im_max - im_min)
                        else:
                            print ("[Warning] image is constant!")
                            im_std = np.zeros_like(im)
                        plt.imshow(im_std)
                        plt.axis('off')
                        plt.title("{}: o={}, t={}".format(label, j, i))
                    plt.show()
                    #plt.waitforbuttonpress()
    elif data.ndim == 1:
        print("[Info] {} (min, max, mean, std)=({}, {}, {}, {})".format(
                      label,
                      np.min(data),
                      np.max(data),
                      np.mean(data),
                      np.std(data)))
        print("[Info] data[:10]={}".format(data[:10]))

    return

def main():
    show_images = False
    diagnose_plots = False
    model_dir = './models'
    global backend

    # override backend if provided as an input arg
    if len(sys.argv) > 1:
        if 'tf' in sys.argv[1].lower():
            backend = 'tf'
        else:
            backend = 'th'
    print ("[Info] Using backend={}".format(backend))

    if backend == 'th':
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_th.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_th.json')
    else:
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
        model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

    #モデルの取得
    print("[Info] Reading model architecture...")
    #model = model_from_json(open(model_json_filename, 'r').read())
    model = c3d_model.get_model(backend=backend)

    print("[Info] Loading model weights...")
    #一旦重み取得解除
    model.load_weights(model_weight_filename)
    print("[Info] Loading model weights -- DONE!")

    #取得したモデルの重みを利用した別のモデルの生成
    int_model = c3d_model.get_int_model(model)
    #modelの最上部にsoftmaxを追加する
    cnn_model = c3d_model.add_softmax(int_model)
    rnn_model = marge_rnn_model.get_model()


    #モデル画像の生成
    '''
    model_img_filename = os.path.join(model_dir, 'c3d_model.png')
    if not os.path.exists(model_img_filename):
        from keras.utils import plot_model
        plot_model(model, to_file=model_img_filename)
    '''
    #モデルコンパイル
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    cnn_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    rms = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.9)#このoptimizerはRNNに向いている
    rnn_model.compile(loss='categorical_crossentropy',optimizer=rms,metrics=['accuracy'])
    #モデルの表示
    #model.summary()

    #numpyデータの読み込み(スケルトンデータ)
    skel_train_data = np.load('train_data.npy')
    skel_test_data = np.load('test_data.npy')
    skel_train_label = np.load('train_label.npy')
    skel_test_label = np.load('test_label.npy')

    #numpyデータの読み込み(RGBデータ)
    rgb_train_data = np.load('train_data_rgb.npy')
    rgb_test_data = np.load('test_data_rgb.npy')
    rgb_train_label = np.load('train_label_rgb.npy')
    rgb_test_label = np.load('test_label_rgb.npy')

    #データ型変換
    skel_train_data = skel_train_data.astype('float32')
    skel_test_data = skel_test_data.astype('float32')
    skel_train_label = skel_train_label.astype('int32')
    skel_test_label = skel_test_label.astype('int32')

    rgb_train_data = rgb_train_data.astype('float32')
    rgb_test_data = rgb_test_data.astype('float32')
    rgb_train_label = rgb_train_label.astype('int32')
    rgb_test_label = rgb_test_label.astype('int32')

    #RGBのデータだけは画素値なので正規化しておく
    rgb_train_data /= 255
    rgb_test_data /= 255
    #訓練データ／テストデータの取得(NTU-RGB+D)↑compileは上にあり

    #訓練fit
    # CNN Training parameters
    batch_size = 20 #バッチサイズ
    num_classes = 5 #分けるクラス
    epochs = 1 #学習回数
    #モデルを学習する
    fit = cnn_model.fit(rgb_train_data, rgb_train_label, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(rgb_test_data, rgb_test_label))
    # モデルを評価した結果を表示する
    print("cnn評価")
    scores = cnn_model.evaluate(rgb_test_data, rgb_test_label, verbose=0)
    print("%s: %.2f%%" % (cnn_model.metrics_names[1], scores[1]*100))

    Y_pred = cnn_model.predict(rgb_test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    target_names = ['class 0(Drink_water)', 'class 1(Sitting_down)', 'class 2(Take_on_shoe)','class 3(Put_on_a_hat)','class 4(Nod_head)']
    print(classification_report(np.argmax(rgb_test_label,axis=1), y_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(rgb_test_label,axis=1), y_pred))

    # RNN Training parameters
    batch_size = 20 #バッチサイズ
    num_classes = 5 #分けるクラス
    epochs = 1 #学習回数
    #モデルを学習する
    fit = rnn_model.fit(skel_train_data, skel_train_label, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(skel_test_data, skel_test_label))
    # モデルを評価した結果を表示する
    print("rnn評価")
    scores = rnn_model.evaluate(skel_test_data, skel_test_label, verbose=0)
    print("%s: %.2f%%" % (rnn_model.metrics_names[1], scores[1]*100))

    Y_pred = rnn_model.predict(skel_test_data)
    y_pred = np.argmax(Y_pred, axis=1)
    print(classification_report(np.argmax(skel_test_label,axis=1), y_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(skel_test_label,axis=1), y_pred))

    print("マージ処理")
    cnn_pred = cnn_model.predict(rgb_test_data)
    rnn_pred = rnn_model.predict(skel_test_data)
    print("cnn_pred.shape")
    print(cnn_pred.shape)

    # 空のアレイ
    array_emp = np.empty((120, 5))
    cnn_max=cnn_pred.max(1)
    rnn_max=rnn_pred.max(1)
    print("cnn_max.shape")
    print(cnn_max.shape)

    for k in range(120):
        if rnn_max[k]*1.0 > cnn_max[k]*3.02:
            array_emp[k,:] = rnn_pred[k,:]
        else:
            array_emp[k,:] = cnn_pred[k,:]
    print("array_emp")
    print(array_emp)
    marge_pred = np.argmax(array_emp, axis=1)
    print("marge_pred[0]")
    print(marge_pred[0])

    print(classification_report(np.argmax(skel_test_label,axis=1), marge_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(skel_test_label,axis=1), marge_pred))
'''
    #可視化
    # フォルダの作成
    # make output directory
    folder = 'marge_results/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    # 損失の描画
    # loss
    def plot_history_loss(fit):
        # Plot the loss in the history
        axL.plot(fit.history['loss'],label="loss for training")
        axL.plot(fit.history['val_loss'],label="loss for validation")
        axL.set_title('model loss')
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc='upper right')

    # 正解率の描画
    # acc
    def plot_history_acc(fit):
        # Plot the loss in the history
        axR.plot(fit.history['acc'],label="accuracy for training")
        axR.plot(fit.history['val_acc'],label="accuracy for validation")
        axR.set_title('model accuracy')
        axR.set_xlabel('epoch')
        axR.set_ylabel('accuracy')
        axR.legend(loc='upper right')

    plot_history_loss(fit)
    plot_history_acc(fit)
    fig.savefig(folder + '/c3d_loss_acc.pdf')
'''

if __name__ == '__main__':
    main()
