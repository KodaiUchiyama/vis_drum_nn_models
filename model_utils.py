# coding:utf-8
#from exp_models.CNN_LSTM_models import *
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import Callback
from keras.models import save_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, errno
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.metrics import mean_squared_error
import gc
import math
import sys
from tqdm import tqdm
import h5py
####################
# Data related Utils
#データをセットをロードしたのち返す関数
def load5hpyTrainData(data_name):
    """Load h5py data and return HDF5 object corresponding to X_train, Y_train
        Returns:
            dataX_train (HDF5Matrix object): keras object for loading h5py datasets
            dataY_train (HDF5Matrix object): keras object for loading h5py datasets
    """
    #data_dir = '/home/KODAI/MATLAB_vis_master/'
    data_dir = '/home/KODAI/vis_drum_master/'
    data_file = data_dir + data_name  # data_name = 'TopAngle100_dataX_dataY.h5' by default

    # Load first element of data to extract information on video
    with h5py.File(data_file, 'r') as hf:
        print("Reading train data from file..")
        #dataX_train = hf['dataX_train']  # Adding the [:] actually loads it into memory
        dataY_train = hf['dataY_train']
        dataZ_train = hf['dataZ_train']
        #print("dataX_train.shape:", dataX_train.shape)
        print("dataY_train.shape:", dataY_train.shape)
        print("dataZ_train.shape:", dataZ_train.shape)

    # Load data into HDF5Matrix object, which reads the file from disk and does not put it into RAM
    #dataX_train = HDF5Matrix(data_file, 'dataX_train')
    dataY_train = HDF5Matrix(data_file, 'dataY_train')
    dataZ_train = HDF5Matrix(data_file, 'dataZ_train')
    #dataX_train = HDF5Matrix(data_file, 'dataX_train',start=0,end=60000)
    #dataY_train = HDF5Matrix(data_file, 'dataY_train',start=0,end=6000)
    #dataZ_train = HDF5Matrix(data_file, 'dataZ_train',start=0,end=6000)
    #データをnumpy形式の変換
    print("convert h5 to numpy ...")
    #dataX_train = np.array(dataX_train)
    dataY_train = np.array(dataY_train)
    dataZ_train = np.array(dataZ_train)
    #np.save('3d_RGB_dataX.npy', dataX_train)
    np.save('3d_RGB_dataY.npy', dataY_train)
    np.save('3d_RGB_dataZ.npy', dataZ_train)


def load5hpyTestData(data_name):
    """Load h5py data and return HDF5 object corresponding to X_test, Y_test
        Returns:
            dataX_test (HDF5Matrix object): keras object for loading h5py datasets
            dataY_test (HDF5Matrix object): keras object for loading h5py datasets
    """
    data_dir = '/home/KODAI/vis_drum_master/'
    data_file = data_dir + data_name  # data_name = 'TopAngle100_dataX_dataY.h5' by default

    # Load first element of data to extract information on video
    with h5py.File(data_file, 'r') as hf:
        print("Reading test data from file..")
        dataX_test = hf['dataX_test']
        dataY_test = hf['dataY_test']
        #dataZ_test = hf['dataZ_train']
        print("dataX_test.shape:", dataX_test.shape)
        print("dataY_test.shape:", dataY_test.shape)
        #print("dataZ_test.shape:", dataZ_test.shape)

    # Load data into HDF5Matrix object, which reads the file from disk and does not put it into RAM
    dataX_test = HDF5Matrix(data_file, 'dataX_test')
    dataY_test = HDF5Matrix(data_file, 'dataY_test')
    #dataZ_test = HDF5Matrix(data_file, 'dataZ_train')

    #データをnumpy形式の変換
    print("convert h5 to numpy ...")
    dataX_test = np.array(dataX_test)
    dataY_test = np.array(dataY_test)
    #dataZ_test = np.array(dataZ_test)
    np.save('dataX_test.npy', dataX_test)
    np.save('dataY_test.npy', dataY_test)
    #np.save('dataZ_test.npy', dataZ_test)

def load5hpyDetectionData(data_name):
    """Load h5py data and return HDF5 object corresponding to X_train, Y_train
        Returns:
            dataX_train (HDF5Matrix object): keras object for loading h5py datasets
            dataY_train (HDF5Matrix object): keras object for loading h5py datasets
    """
    #data_dir = '/home/KODAI/MATLAB_vis_master/'
    data_dir = '/home/KODAI/vis_drum_master/'
    data_file = data_dir + data_name  # data_name = 'TopAngle100_dataX_dataY.h5' by default

    # Load first element of data to extract information on video
    with h5py.File(data_file, 'r') as hf:
        print("Reading detection data from file..")
        dataX_train = hf['dataX_train']  # Adding the [:] actually loads it into memory
        dataS_train = hf['dataS_train']
        dataY_train = hf['dataY_train']
        #dataZ_train = hf['dataZ_train']
        print("dataS_train.shape:", dataS_train.shape)
        print("dataX_train.shape:", dataX_train.shape)
        print("dataY_train.shape:", dataY_train.shape)
        #print("dataZ_train.shape:", dataZ_train.shape)

    # Load data into HDF5Matrix object, which reads the file from disk and does not put it into RAM
    dataS_train = HDF5Matrix(data_file, 'dataS_train')
    dataX_train = HDF5Matrix(data_file, 'dataX_train')
    dataY_train = HDF5Matrix(data_file, 'dataY_train')
    #dataZ_train = HDF5Matrix(data_file, 'dataZ_train')
    #データをnumpy形式の変換
    print("convert h5 to numpy ...")
    dataX_train = np.array(dataX_train)
    dataS_train = np.array(dataS_train)
    dataY_train = np.array(dataY_train)
    #dataZ_train = np.array(dataZ_train)
    np.save('train_space_detection_dataS.npy', dataS_train)
    np.save('trian_space_detection_dataX.npy', dataX_train)
    np.save('train_space_detection_dataY.npy', dataY_train)
    #np.save('space_detection_dataZ.npy', dataZ_train)


#データベースから一つサンプルを取り出してデータセットの型を返す関数
def returnH5PYDatasetDims(data_name):
    """Load h5py data and return the dimensions of data in the dataet
            Returns:
                frame_h (int): image height
                frame_w (int): image width
                channels (int): number of channels in image
                audio_vector_dim (int): number of dimensions (or features) in audio vector

    """

    data_dir = '/home/KODAI/vis_drum_master/'
    data_file = data_dir + data_name  # data_name = 'vis_dataX_dataY.h5' by default

    with h5py.File(data_file, 'r') as hf:
        print("Reading data sample from file..")
        dataZ_sample = hf['dataZ_train'][0]  # select one sample from (7233,244,244,3)
        dataY_sample = hf['dataY_train'][0]
        print("dataZ_sample.shape:", dataZ_sample.shape)
        print("dataY_sample.shape:", dataY_sample.shape)
#
    (frame_h, frame_w, channels) = dataZ_sample.shape  # (90,160,15)
    audio_vector_dim = dataY_sample.shape[0]
#inputデータ（image）のshapeとoutputのgroundtruthのオーティオデータの型の次元を返す
    return frame_h, frame_w, channels, audio_vector_dim

########################
# Custom Keras Callbacks

class saveModelOnEpochEnd(Callback):
    """Custom callback for Keras which saves model on epoch end"""
    def on_epoch_end(self, epoch, logs={}):
        # Save the model at every epoch end
        print("Saving trained model...")
        model_prefix = 'CNN_LSTM'
        model_path = "../trained_models/" + model_prefix + ".h5"
        save_model(self.model, model_path,
                   overwrite=True)  # saves weights, network topology and optimizer state (if any)
        return

class LossHistory(Callback):
    """Custom callback for Keras which saves loss history of training and testing data"""
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.test_losses = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.test_losses.append(logs.get('val_loss'))

class AccuracyHistory(Callback):
    """Custom callback for Keras which saves accuracy history of training and testing data"""
    def __init__(self):
        super().__init__()
        self.train_acc = []
        self.test_acc = []

    def on_batch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))
        self.test_acc.append(logs.get('val_acc'))

##################################
# Plotting and saving figure utils

def plotAndSaveData(param_history,y_label,learning_rate_hp,weight_hp,title="Parameter History"):
    """Saves a matplotlib plot that graphs parameter history

            Args:
                param_history (Callback object): Keras object that contains information on the parameter history
                y_label (string): label for y axis e.g "Loss"
                learning_rate_hp (float): learning rate hyperparameter e.g 4e-7
                weight_hp (float): weight scale hyperparameter e.g 0.005
                title (string): title for saved graph - default="Parameter History"

            Returns:
                None

            """

    param_history = (vars(param_history))

    colors = ['r','b']

    for idx, param in enumerate(param_history):
        param_values = param_history[param]
        epochs = np.arange(len(param_values))  # epochs is 1,2,3...[num items in param values]
        plt.plot(epochs, param_values, colors[idx])

    # generate legend
    red_patch = mpatches.Patch(color='red', label='train')
    blue_patch = mpatches.Patch(color='blue', label='test')
    plt.legend(handles=[blue_patch, red_patch], prop={'size': 10})

    # Plot the graph
    plt.xlabel('epochs')
    plt.ylabel(y_label)
    title = '%s-{lr:%6f}-{ws:%6f}' % (title, learning_rate_hp, weight_hp)
    plt.title(title)
    # plt.show()
    plt.draw()
    file_name = '{lr:%8f}-{ws:%6f}.png' % (learning_rate_hp, weight_hp)
    plt.savefig('../graphs/training_history/' + file_name)
    plt.close()

def makeDir(dir_name):
    """Checks if a directory exists and creates one if it doesn't exist

              Args:
                  dir_name (string): folder name e.g "{lr:0.000597}-{ws:0.000759}"

              Returns:
                  dir_name (string): complete relative directory name e.g "../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}"
    """

    dir_path = 'results/' + dir_name
    print("Make directory for save predicted spectrums...")
    print(os.path.exists(dir_path))
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return dir_path


#テストデータセット(１動画についての生成かも！)に対してモデルが適応されたとき予測されたスペクトラムを生成して保存する関数
def genAndSavePredSpectrum(model,save_img_path, window_length = 300, data_name='vis_trim_dataX_dataY.h5'):
    """Generates and saves predicted spectrums when using the model on an unseen test set
テストデータセットに対してモデルが適応されたとき予測されたスペクトラムを生成して保存する
              Args:
                  model (Keras model object): model created during training
                  save_img_path (string): path where to save images e.g "../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}"
                  window_length (int): length of predicted window. Default = 300
                  data_name (string): name of the dataset e.g 'TopAngle100_dataX_dataY.h5'

              Returns:
                  None
    """
    # Define the external SSD where the dataset resides in
    data_dir = '/Users/KODAI/Documents/MATLAB_vis_master/'
    #data_dir = '/home/KODAI/MATLAB_vis_master/'
    file_name = data_dir + data_name
    print("Generate predicted spectrums...")
    # Open the h5py file
    with h5py.File(file_name, 'r') as hf:
        print("Reading test data from file..")
        dataX_test = hf['dataX_train'][:]
        dataY_test = hf['dataY_train'][:]
    print("dataX_test.shape:", dataX_test.shape)
    print("dataY_test.shape:", dataY_test.shape)
    print("np.max(dataY)", np.max(dataX_test))
    print("np.min(dataY)", np.min(dataY_test))

    (num_frames, frame_h, frame_w, channels) = dataX_test.shape
    num_windows = math.floor(num_frames / window_length)

#もしnum_framesが１２００だったら、window_length＝３００よりnum_windows＝４
#以下のforの様子は
#0--300
#300--600
#600--900
#900--1200
    for i in tqdm(range(num_windows)):#tpdmはfor文の進捗表示
        pred_idx = i * window_length
        end_idx = pred_idx + window_length

        #それ来たテストデータpredict
        trainPredict = model.predict(dataX_test)
        trainScore = math.sqrt(mean_squared_error(dataY_test[pred_idx:end_idx, :], trainPredict[pred_idx:end_idx, :]))
        print('Train score: %.3f RMSE' % (trainScore))#平均平方二乗誤差

        ##### PLOT RESULTS
        trainPlot = model.predict(dataX_test[pred_idx:end_idx, :])
        print(trainPlot.shape)
        plt.subplot(3, 1, 1)
        plt.imshow(trainPlot.T, aspect='auto')# (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)
        plt.title('Predicted feature')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')

        plt.subplot(3, 1, 2)
        plt.title('Ground Truth')
        plt.ylabel('Note bins')
        plt.xlabel('Time (frames)')
        #注釈を付ける
        plt.annotate('RMSE: %.3f' % (trainScore), xy=(5, 5), xytext=(5, 33))
        plt.imshow(dataY_test[pred_idx:end_idx, :].T, aspect='auto')# (pred_idx:end_idx , 42) > (42 , pred_idx:end_idx)

        plt.subplot(3, 1, 3)
        plt.imshow(dataY_test[pred_idx:end_idx, :].T, aspect='auto')
        plt.colorbar()
        plt.tight_layout()#図の調整
        plt.draw()
        plt.savefig(save_img_path + str(i) + '.png')  # ../graphs/predicted_spectrums/{lr:0.000597}-{ws:0.000759}/1.png
        plt.close()
#Once all trials are complete, make a 3D plot that graphs x: learning rate, y: weight scale and z: final_accuracy
def plotAndSaveSession(learning_rates,weight_scales,final_accuracies):
    learning_rates = np.array(learning_rates)
    weight_scales = np.array(weight_scales)
    final_accuracies = np.array(weight_scales)

    print (final_accuracies)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.scatter(learning_rates, weight_scales, final_accuracies)

    ax.set_xlabel('Learning Rates')
    ax.set_ylabel('Weight init')
    ax.set_zlabel('Final accuracy')

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
