# coding:utf-8
import matplotlib
from matplotlib import cm
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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils.io_utils import HDF5Matrix
from keras import backend as K
from keras.callbacks import EarlyStopping
#########################
#Set data_name
detection_data_name = 'train_SPACE_detection_dataX_dataY.h5'  # Set to name of h5py dataset
train_data_name = '3d_RGB_dataX_dataY.h5'  # Set to name of h5py dataset
test_data_name = '3d_test_SPACE_dataX_dataY.h5'
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
        output_dim = 42
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
        frame_h=299
        frame_w=299
        channels=3
        audio_vector_dim=42

        #load5hpyDetectionData(data_name = detection_data_name)
        #load5hpyTrainData(data_name = train_data_name)
        #load5hpyTestData(data_name = test_data_name)
        #sys.exit()
        
        #保存したnumpyデータを読み込み
        #dataY_train = np.load('./old_dataset/3d_RGB_dataY.npy')
        #dataZ_train = np.load('./old_dataset/3d_RGB_dataZ.npy')
        dataS_train = np.load('train_space_detection_dataS.npy')
        dataY_train = np.load('train_space_detection_dataY.npy')
        dataZ_train = np.load('train_space_detection_dataX.npy')
        print("dataS_train.shape:", dataS_train.shape)
        print("dataY_train.shape:", dataY_train.shape)
        print("dataZ_train.shape:", dataZ_train.shape)
        
        
        '''
        #配列をシャッフル
        dataZ_train, dataY_train = shuffle(dataZ_train, dataY_train)
        
        new_dim_len = len(dataY_train)*30
        dataY_train = dataY_train.reshape((new_dim_len,audio_vector_dim))
        dataZ_train = dataZ_train.reshape((new_dim_len,frame_h,frame_w,channels))
        print("reshaped dataY_train.shape:", dataY_train.shape)
        print("reshaped dataZ_train.shape:", dataZ_train.shape)
        '''
        
        dataS_val = np.load('val_space_detection_dataS.npy')
        dataY_val = np.load('val_space_detection_dataY.npy')
        dataZ_val = np.load('val_space_detection_dataX.npy')
        print("dataS_val.shape:", dataS_val.shape)
        print("dataY_val.shape:", dataY_val.shape)
        print("dataZ_val.shape:", dataZ_val.shape)
        '''
        dataY_test = np.load('dataY_test.npy')
        dataZ_test = np.load('dataX_test.npy')
        print("dataY_test.shape:", dataY_test.shape)
        print("dataZ_test.shape:", dataZ_test.shape)
        new_dim_len = len(dataY_test)*30
        dataY_test = dataY_test.reshape((new_dim_len,audio_vector_dim))
        dataZ_test = dataZ_test.reshape((new_dim_len,frame_h,frame_w,channels))
        print("reshaped dataY_train.shape:", dataY_test.shape)
        print("reshaped dataZ_train.shape:", dataZ_test.shape)
        '''
        '''
        # dataYを主成分分析して42次元から10次元に次元削減する
        # PCA
        # 次元数10まで圧縮
        #トレーニング
        pca = PCA(n_components=10)
        dataY_train_pca = pca.fit_transform(dataY_train)
        print('dataY_train_pca shape: {}'.format(dataY_train_pca.shape))
        # 寄与率
        print('explained variance ratio train: {}'.format(pca.explained_variance_ratio_))
        '''
        
        #データをトレーニングとテストに分割       
        dataZ_train, dataZ_test, dataY_train, dataY_test,dataS_train, dataS_test  = train_test_split(dataZ_train, dataY_train,dataS_train,test_size=0.1, shuffle = False)
        #dataZ_train, dataZ_test, dataY_train, dataY_test  = train_test_split(dataZ_train, dataY_train,test_size=0.1, shuffle = False)
        print("dataY_train.shape:", dataY_train.shape)
        print("dataZ_train.shape:", dataZ_train.shape)
        print("dataS_train.shape:", dataS_train.shape)
        print("dataY_test.shape:", dataY_test.shape)
        print("dataZ_test.shape:", dataZ_test.shape)
        print("dataS_test.shape:", dataS_test.shape)
        
        '''
        #テスト
        pca = PCA(n_components=10)
        dataY_test_pca = pca.fit_transform(dataY_test)
        print('dataY_train_pca shape: {}'.format(dataY_test_pca.shape))
        # 寄与率
        print('explained variance ratio train: {}'.format(pca.explained_variance_ratio_))
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
        model = InceptionV3_model(image_dim=(frame_h,frame_w,channels),
                              audio_vector_dim=audio_vector_dim,
                              learning_rate=lr,
                              weight_init=ws,
                              output_dim=output_dim,
                              optimizer=optimizer)
        '''
        
        model = InceptionV3_LSTM_model(image_dim=(frame_h,frame_w,channels),
                              audio_vector_dim=audio_vector_dim,
                              learning_rate=lr,
                              weight_init=ws,
                              output_dim=output_dim,
                              optimizer=optimizer)
        
        '''
        model = AlexNet_model(image_dim=(frame_h,frame_w,channels),
                              audio_vector_dim=audio_vector_dim,
                              learning_rate=lr,
                              weight_init=ws,
                              output_dim=output_dim,
                              optimizer=optimizer)
        '''
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

        print("learning rate :",lr)
        print("optimizer :",optimizer)
        # train the model #IMPUT space time image & RGB image , OUTPUT decomposed audio vector
        es_cb = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
        fit=model.fit([dataZ_train, dataS_train], dataY_train,shuffle=False,epochs=50, batch_size=30,verbose=1,validation_data=([dataZ_test,dataS_test], dataY_test), callbacks=[es_cb])
        model.save('detection_skel_mul_model.h5')
        
        
        #print("learning rate :",lr)
        #print("optimizer :",optimizer)
        #es_cb = EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')
        #fit=model.fit(dataZ_train,dataY_train,shuffle=False,epochs=50, batch_size=30,verbose=1,validation_data=(dataZ_test, dataY_test), callbacks=[es_cb])
        #model.save('RGB_inception_model.h5')
        
        
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
            plt.ylim(0, 1.5)
        plot_history_loss(fit)
        plt.savefig(folder + '/detection_skel_mul_loss.pdf')
        

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
        #model = load_model('space_inception_model.h5',custom_objects={'custom_loss': custom_loss})
        #model = load_model('./saved_models/space_inception_model.h5')
        #print("loading model done..")
        #hoge*10次元
        #Predicted_feature = model.predict([dataZ_val,dataS_val])
        Predicted_feature = model.predict(dataZ_test)
        #hoge*42次元
        #inversed_feature = pca.inverse_transform(Predicted_feature)
        #shape check
        print("predicted_feature_42dim",Predicted_feature.shape)
        #print("inversed_feature_42dim",inversed_feature.shape)
        print("max",np.max(Predicted_feature))
        print("min",np.min(Predicted_feature))
        print("maxg",np.max(dataY_test))
        print("ming",np.min(dataY_test))
        ##### PLOT RESULTS
        plt.subplot(2, 1, 1)
        plt.imshow(Predicted_feature[0:400].T, aspect='auto',vmin=-1.0,vmax=20.0)# (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
        plt.title('42dim Predicted feature')
        plt.ylabel('Frequency bands')
        plt.xlabel('Time (frames)')
        plt.colorbar()
        '''
        '''
        plt.subplot(4, 1, 2)
        #注釈を付ける
        #plt.annotate('RMSE: %.3f' % (trainScore), xy=(5, 5), xytext=(5, 33))
        plt.imshow(dataY_test_pca.T, aspect='auto') # (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
        plt.title('10dim Ground Truth')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')

        plt.subplot(4, 1, 3)
        plt.imshow(inversed_feature.T, aspect='auto')
        plt.title('42dim Predicted feature')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')
        '''

        '''
        plt.subplot(2, 1, 2)
        plt.imshow(dataY_test[0:400].T, aspect='auto', vmin=-1.0, vmax=20.0)
        plt.title('42dim Ground Truth')
        plt.ylabel('Frequency bands')
        plt.xlabel('Time (frames)')
        plt.colorbar()

        plt.tight_layout()#図の調整
        plt.draw()
        plt.savefig("predicted_spectrums3" + '.png')  # ../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}/1.png
        plt.close()
        '''
    #Once all trials are complete, make a 3D plot that graphs x: learning rate, y: weight scale and z: final_accuracy
    #ランダムに重みと学習率を生成して、三次元のグラフを作って、どのパラメータの値が効果的かを目視できる
    #plotAndSaveSession(learning_rates,weight_scales,final_accuracies)
    print("--- {EVERYTHING COMPLETE HOMIEEEEEEEEE} ---")

if __name__ == '__main__':
    main()
