# coding:utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from models.c3d_model import *
from models.ALEX_models import *
#import models.c3d_model
from model_utils import *  # Contains some useful functions for loading datasets
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.utils.io_utils import HDF5Matrix
from keras import backend as K
from keras.callbacks import EarlyStopping
#########################
#Set data_name
train_data_name = 'vis_trim_RGB_dataX_dataY.h5'  # Set to name of h5py dataset
#train_data_name = 'vis_drum_dataX_dataY.h5'  # Set to name of h5py dataset
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
        lr = 0.01
        #ws = 10**np.random.uniform(-2.0,-4.0)
        ws = 0.01
        #learning_rates.append(lr)
        output_dim = 10
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
        frame_h, frame_w, channels, audio_vector_dim = returnH5PYDatasetDims(data_name = train_data_name)

        #dataY_train,dataZ_train = load5hpyTrainData(data_name = train_data_name)
        #dataZ_train = load5hpyTrainData(data_name = train_data_name)
        #dataX_test, dataY_test, dataZ_test = load5hpyTestData(data_name = test_data_name)

        #dataY_train = np.array(dataY_train)
        #dataZ_train = np.array(dataZ_train)

        #np.save('dataY_train.npy', dataY_train)
        #np.save('dataZ_train.npy', dataZ_train)
        #sys.exit()
        dataY_train = np.load('dataY_train.npy')
        dataZ_train = np.load('dataZ_train.npy')
        
        #print("dataX_train.shape:", dataX_train.shape)
        print("dataY_train.shape:", dataY_train.shape)
        print("dataZ_train.shape:", dataZ_train.shape)
        
        # dataYを主成分分析して42次元から10次元に次元削減する
        # PCA
        # 次元数10まで圧縮
        pca = PCA(n_components=10)
        dataY_train_pca = pca.fit_transform(dataY_train)
        print('dataY_train_pca shape: {}'.format(dataY_train_pca.shape))
        # 寄与率
        print('explained variance ratio train: {}'.format(pca.explained_variance_ratio_))
       
       
        dataZ_train, dataZ_test, dataY_train_pca, dataY_test_pca = train_test_split(dataZ_train, dataY_train_pca,test_size=0.1, shuffle = False)
        print("dataY_train.shape:", dataY_train_pca.shape)
        print("dataZ_train.shape:", dataZ_train.shape)
        print("dataY_test.shape:", dataY_test_pca.shape)
        print("dataZ_test.shape:", dataZ_test.shape)
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
        
        model = InceptionV3_model(image_dim=(frame_h,frame_w,channels),
                              audio_vector_dim=audio_vector_dim,
                              learning_rate=lr,
                              weight_init=ws,
                              output_dim=output_dim,
                              optimizer=optimizer)
        
        ####################################
        ##c3d用
        '''
        model_dir = './weights'
        model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
        #モデルの取得
        print("[Info] Reading model architecture...")
        model = get_model()

        print("[Info] Loading model weights...")
        model.load_weights(model_weight_filename)
        print("[Info] Loading model weights -- DONE!")

        #取得したモデルの重みを利用した別のモデルの生成
        int_model = get_int_model(model)
        #modelの最上部にsoftmaxを追加する
        cnn_model = add_LSTM(int_model, output_dim)

        #モデルコンパイル
        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        sgd  = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
        def custom_loss(y_true, y_pred):
            for i in range(30):
                loss += K.log(K.sum(K.square(K.abs(y_pred[i] - y_true[i]))) + 1 / (25*25))
            return loss
        #cnn_model.compile(loss=custom_loss, optimizer=optimizer)
        cnn_model.compile(loss='mean_squared_error', optimizer=optimizer)
        #fit=cnn_model.fit(dataZ_train,dataY_train,shuffle=True,epochs=50,batch_size=5,verbose=1,validation_split=0.05)
        '''
        ###################################

        #loss_history = LossHistory()
        #acc_history = AccuracyHistory()
        #callbacks_list = [loss_history, acc_history]

        # train the model #IMPUT space time image & RGB image , OUTPUT decomposed audio vector
        #fit=model.fit([dataX_train , dataZ_train],dataY_train_pca,epochs=50,batch_size=70,shuffle=False,verbose=1,validation_split=0.1)
                  #validation_data=([dataX_test,dataZ_test], dataY_test),
                  #callbacks = callbacks_list)
        es_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
        #fit=model.fit(dataZ_train,dataY_train_pca,shuffle=False,epochs=50, batch_size=30,verbose=1,validation_data=(dataZ_test, dataY_test_pca), callbacks=[es_cb])
        #model.save('rgb_inception_model.h5')
        


        '''
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
        plt.savefig(folder + '/vis_inception_epoch50_loss.pdf')
        '''



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
        

        
        model = load_model('rgb_inception_model.h5',custom_objects={'custom_loss': custom_loss})
        print("loading model done..")
        #hoge*10次元
        Predicted_feature = model.predict(dataZ_test)
        #hoge*42次元
        inversed_feature = pca.inverse_transform(Predicted_feature)
        
        #shape check
        print("predicted_feature_10dim",Predicted_feature.shape)
        print("inversed_feature_42dim",inversed_feature.shape)

        ##### PLOT RESULTS
        plt.subplot(3, 1, 1)
        plt.imshow(Predicted_feature.T, aspect='auto')# (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
        plt.title('Predicted feature')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')

        plt.subplot(3, 1, 2)
        #注釈を付ける
        #plt.annotate('RMSE: %.3f' % (trainScore), xy=(5, 5), xytext=(5, 33))
        plt.imshow(dataY_test_pca.T, aspect='auto') # (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
        plt.title('Ground Truth')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')

        plt.subplot(3, 1, 3)
        plt.imshow(dataY_test_pca.T, aspect='auto')
        plt.colorbar()
        plt.tight_layout()#図の調整
        plt.draw()
        plt.savefig("predicted_spectrums" + '.png')  # ../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}/1.png
        plt.close()
        
    #Once all trials are complete, make a 3D plot that graphs x: learning rate, y: weight scale and z: final_accuracy
    #ランダムに重みと学習率を生成して、三次元のグラフを作って、どのパラメータの値が効果的かを目視できる
    #plotAndSaveSession(learning_rates,weight_scales,final_accuracies)
    print("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")

if __name__ == '__main__':
    main()
