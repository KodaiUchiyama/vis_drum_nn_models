# coding:utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from models.c3d_model import *
import models.c3d_model
from model_utils import *  # Contains some useful functions for loading datasets
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils.io_utils import HDF5Matrix
from keras import backend as K
#########################
#Set data_name
train_data_name = 'vis_trim_dataX_dataY.h5'  # Set to name of h5py dataset
test_data_name = 'vis_trim_test_dataX_dataY.h5'
"""
Main function call for doing random hyperparameter search
ハイパパラメータ生成、種はランダム
"""
def main():
    # 繰り返し回数
    num_trials = 1


    final_accuracies = []
    learning_rates = []
    weight_scales = []

    for i in range(num_trials):

        # random search over logarithmic space of hyperparameters
        #lr = 10**np.random.uniform(-3.0,-6.0)
        lr = 0.001
        #ws = 10**np.random.uniform(-2.0,-4.0)
        ws = 0.01
        #learning_rates.append(lr)
        output_dim = (30, 42)
        #weight_scales.append(ws)
        optimizer = 'sgd'

        """Read a sample from h5py dataset and return key dimensions
model_utils内の関数
            Example returns:
                frame_h = 90
                frame_w = 160
                channels = 3
                audio_vector_dim = 42 = dataY_sample.shape[0]
        """
        #frame_h, frame_w, channels, audio_vector_dim = returnH5PYDatasetDims(data_name = train_data_name)

        """Load full dataset as an HDF5 matrix object for use in Keras model

                Example returns and corresponding matrix shapes:
                    dataX_train.shape = (26000,90,160,15)
                    dataY_train.shape = (26000,42)
                    dataX_test.shape = (4000,90,160,15)
                    dataY_test.shape = (4000,42)
        """
        dataX_train, dataY_train,dataZ_train = load5hpyTrainData(data_name = train_data_name)
        #dataX_test, dataY_test, dataZ_test = load5hpyTestData(data_name = test_data_name)


        print("dataX_train.shape:", dataX_train.shape)
        print("dataY_train.shape:", dataY_train.shape)
        print("dataZ_train.shape:", dataZ_train.shape)
        dataY_train = np.array(dataY_train)
        dataZ_train = np.array(dataZ_train)

        #np.save('dataY_train.npy', dataY_train)
        #np.save('dataZ_train.npy', dataZ_train)
        #dataY_train = np.load('dataY_train.npy')
        #dataZ_train = np.load('dataZ_train.npy')
        #print("dataX_test.shape:", dataX_test.shape)
        #print("dataY_test.shape:", dataY_test.shape)
        #print("dataZ_test.shape:", dataZ_test.shape)
        #RGBのデータだけは画素値なので正規化しておく
        rgb_train_data /= 255
        rgb_test_data /= 255
        # dataYを主成分分析して42次元から10次元に次元削減する
        # PCA
        # 次元数10まで圧縮
        pca = PCA(n_components=10)
        dataY_train_pca = pca.fit_transform(dataY_train)
        print('dataY_train_pca shape: {}'.format(dataY_train_pca.shape))
        # 寄与率
        print('explained variance ratio train: {}'.format(pca.explained_variance_ratio_))
        '''
        pca_test = PCA(n_components=10)
        dataY_test_pca = pca_test.fit_transform(dataY_test)
        print('dataY_train_pca shape: {}'.format(dataY_test_pca.shape))
        # 寄与率
        print('explained variance ratio of test: {}'.format(pca_test.explained_variance_ratio_))
        '''
        '''
        model = CNN_LSTM_model(image_dim=(frame_h,frame_w,channels),
                               audio_vector_dim=audio_vector_dim,
                               learning_rate=lr,
                               weight_init=ws)
        '''
        '''
        model = create_model(image_dim=(frame_h,frame_w,channels),
                              audio_vector_dim=audio_vector_dim,
                              learning_rate=lr,
                              weight_init=ws,
                              output_dim=output_dim,
                              optimizer=optimizer)
        '''
        '''
        model = AlexNet_model(image_dim=(frame_h,frame_w,channels),
                              audio_vector_dim=audio_vector_dim,
                              learning_rate=lr,
                              weight_init=ws,
                              output_dim=output_dim,
                              optimizer=optimizer)
        '''
        model_dir = './weights'
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
        #モデルの取得
        print("[Info] Reading model architecture...")
        model = c3d_model.get_model()

        print("[Info] Loading model weights...")
        model.load_weights(model_weight_filename)
        print("[Info] Loading model weights -- DONE!")

        #取得したモデルの重みを利用した別のモデルの生成
        int_model = c3d_model.get_int_model(model)
        #modelの最上部にsoftmaxを追加する
        cnn_model = c3d_model.add_LSTM(int_model, output_dim)

        #モデルコンパイル
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd  = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        def custom_loss(y_true, y_pred):
        for i range(30):
            loss += K.log(K.sum(K.square(K.abs(y_pred[i] - y_true[i]))) + 1 / (25*25))
        return loss
        cnn_model.compile(loss=custom_loss, optimizer=optimizer)
        #loss_history = LossHistory()
        #acc_history = AccuracyHistory()
        #callbacks_list = [loss_history, acc_history]

        # train the model #IMPUT space time image & RGB image , OUTPUT decomposed audio vector
        #fit=model.fit([dataX_train , dataZ_train],dataY_train_pca,epochs=50,batch_size=70,shuffle=False,verbose=1,validation_split=0.1)
                  #validation_data=([dataX_test,dataZ_test], dataY_test),
                  #callbacks = callbacks_list)
        fit=cnn_model.fit(dataZ_train,dataY_train_pca,shuffle=True,epochs=50,batch_size=5,verbose=1,validation_split=0.05)

        #fit=model.fit(dataZ_test,dataY_test_pca,shuffle=False,epochs=2,batch_size=10,verbose=1,validation_split=0.1)
        cnn_model.save('rgb_custom_model.h5')
        #model.save('twostream_model.h5')
        #model = load_model('rgb_model_not_drop.h5')
        #print(model.summary())
        #可視化
        # フォルダの作成
        # make output directory
        folder = 'results/'
        if not os.path.exists(folder):
            os.makedirs(folder)


        def plot_history_loss(fit):
            plt.plot(fit.history['loss'],label="loss for training")
            plt.plot(fit.history['val_loss'],label="loss for validation")
            plt.title('model loss')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc='upper right')
            plt.ylim(0, 10)
        plot_history_loss(fit)
        plt.savefig(folder + '/vis_custom_loss.pdf')

        '''
        # Graph training history
        plotAndSaveData(loss_history, "Loss", learning_rate_hp=lr, weight_hp=ws, title="Loss History")
        plotAndSaveData(acc_history, "Accuracy", learning_rate_hp=lr, weight_hp=ws, title="Acc History")

        # Save final accuracy 最後のplotAndSaveSessionで使用
        final_accuracies.append(acc_history.test_acc[-1])
        '''
        '''
        #これは1動画が入っているテストサンプルを用意してから、loadmatとかでいいでしょh5py経由しなくても
        # Make a directory to store the predicted spectrums
        folder_name = '{lr:%s}-{ws:%s}' % (str(lr), str(ws))
        dir_path = makeDir(folder_name)  # dir_path = "../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}"
        # Run the model on some unseen data and save the predicted spectrums in the directory defined previously

        genAndSavePredSpectrum(model,
                               dir_path,
                               window_length = 300,
                               data_name=train_data_name)
        '''
        '''
        #hoge*10次元
        trainPredict = model.predict(dataZ_train)
        print(trainPredict[0])
        print(dataY_train_pca[0])
        y_pred = trainPredict[0]
        y_true = dataY_train_pca[0]
        A = K.variable(y_pred)
        B = K.variable(y_true)
        print(A - B)#10*hoge
        print(K.get_value(A - B))
        print("shape:",K.int_shape(A - B))
        print("絶対値",K.get_value(K.abs(A - B)))
        print("絶対値の二乗",K.get_value(  K.square(K.abs(A - B))  ))
        print("絶対値の二乗の要素の和",K.get_value(  K.sum(K.square(K.abs(A - B)))  ))
        print("e",1/(25*25))
        print("絶対値の二乗の要素の和 + e",K.get_value(  K.sum(K.square(K.abs(A - B))) + 1 / (25*25)  ))
        print("log(絶対値の二乗の要素の和 + e)",K.get_value(  K.log(K.sum(K.square(K.abs(y_pred - y_true))) + 1 / (25*25))  ))
        print("log(絶対値の二乗の要素の和 + e)",K.get_value(  K.log(K.sum(K.square(K.abs(A - B))) + 1 / (25*25))  ))
        #K.log(K.pow(K.l2_normalize(y_pred - y_true, 0),2) + 1/K.pow(25,2))
        '''
        '''
        inversed_dim = pca.inverse_transform(trainPredict)
        print("42")
        print(inversed_dim[0])
        print(dataY_train[0])
        '''

        '''
        #hoge*42次元
        inversed_dim = pca.inverse_transform(trainPredict)
        print(inversed_dim.shape)

        for i in range(10):
            print("input",dataY_train[i])
            print("output",inversed_dim[i])

        ##### PLOT RESULTS
        inversed_dim = inversed_dim[0:500, :]
        print(inversed_dim.shape)
        dataY_train = dataY_train[0:500, :]
        print(dataY_train.shape)

        plt.subplot(3, 1, 1)
        plt.imshow(inversed_dim.T, aspect='auto')# (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
        plt.title('Predicted feature')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')

        plt.subplot(3, 1, 2)
        #注釈を付ける
        #plt.annotate('RMSE: %.3f' % (trainScore), xy=(5, 5), xytext=(5, 33))
        plt.imshow(dataY_train.T, aspect='auto') # (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
        plt.title('Ground Truth')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')

        plt.subplot(3, 1, 3)
        plt.imshow(dataY_train.T, aspect='auto')
        plt.colorbar()
        plt.tight_layout()#図の調整
        plt.draw()
        plt.savefig("predicted_spectrums" + '.png')  # ../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}/1.png
        plt.close()

        for i in range(5):
            print("\n")
            print("input",dataY_train_pca[i,:])
            print("\n")
            print(trainPredict[i,:])
        sys.exit()
        '''
    #Once all trials are complete, make a 3D plot that graphs x: learning rate, y: weight scale and z: final_accuracy
    #ランダムに重みと学習率を生成して、三次元のグラフを作って、どのパラメータの値が効果的かを目視できる
    #plotAndSaveSession(learning_rates,weight_scales,final_accuracies)
    print("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")

if __name__ == '__main__':
    main()
