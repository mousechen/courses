# 自己定义一个Vgg16的类，封装起来。

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers import Input,Dense,Activation,ZeroPadding2D,Dropout,Conv2D,MaxPool2D,Flatten,Lambda,BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

# In case we are going to use the TensorFlow backend we need to explicitly set the Theano image ordering
from keras import backend as K
K.set_image_dim_ordering('th')



vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr


class Vgg16():
    
    """
    Vgg 16模型实现
    """
    
    def __init__(self):
        self.FILES_PATH = 'http://files.fast.ai/models/'
        self.create()
        self.get_classes()
       
        
    def get_classes(self):
        """
        下载Imagenet的图片分类，存在缓冲中目录为.keras
        """
        # 暂时使用fast.ai的目录
        fname='imagenet_class_index.json'

        fpath = get_file(fname,self.FILES_PATH+fname,cache_dir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]
        
    def predict(self,imgs,detail=False):
        """
            Predict the labels of a set of images using the VGG16 model.

            Args:
                imgs (ndarray)    : An array of N images (size: N x width x height x channels).
                details : ??
            
            Returns:
                preds (np.array) : Highest confidence value of the predictions for each image.
                idxs (np.ndarray): Class index of the predictions with the max confidence.
                classes (list)   : Class labels of the predictions with the max confidence.
        """
        
        all_preds = self.model.predict(imgs)
        print(all_preds)
        idxs = np.argmax(all_preds,axis=1)
        preds = [all_preds[i,idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds),idxs,classes
    
    def ConvBlock(self,n_layers,n_filters):
        """
            Adds a specified number of ZeroPadding and Covolution layers
            to the model, and a MaxPooling layer at the very end.

            Args:
                layers (int):   The number of zero padded convolution layers
                                to be added to the model.
                filters (int):  The number of convolution filters to be 
                                created for each layer.
        """
        model = self.model
        for layer in range(n_layers):
                model.add(ZeroPadding2D(padding=(1,1)))
                model.add(Conv2D(filters=n_filters,kernel_size=(3,3),strides=(1,1),activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        
        
        
    def FcBlock(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a
            Dropout of 0.5

            Args:   None
            Returns:   None
        """
        model = self.model
        model.add(Dense(4096,activation='relu')) # FC1
        model.add(Dropout(rate=0.5))
        
    def create(self):
        """
            Creates the VGG16 network achitecture and loads the pretrained weights.

            Args:   None
            Returns:   None
        """
        model = self.model = Sequential()
        #  顺序 (Sequential) 模型写法第一add 必须加input_shape
        model.add(Lambda(vgg_preprocess, input_shape=(3,224,224), output_shape=(3,224,224)))

        self.ConvBlock(2,64) # 最开始的2层用64个3x3的过滤器，然后pool
        self.ConvBlock(2,128)
        self.ConvBlock(3,256)
        self.ConvBlock(3,512)
        self.ConvBlock(3,512)

        model.add(Flatten()) # 拉平，展开
        self.FcBlock()
        self.FcBlock()
        model.add(Dense(1000,activation='softmax')) # 1000个分类
        
        # 读取预训练好的模型权重
        fpath = get_file('vgg16.h5', self.FILES_PATH+'vgg16.h5', cache_subdir='models') # 读取训练好的权重
        
        model.load_weights(fpath)
        
            
    def get_batches(self,path,gen = image.ImageDataGenerator(),class_mode='categorical',batch_size=4,shuffle=True):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data. Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path,target_size=(224,224),
                                   class_mode=class_mode,batch_size=batch_size,shuffle=shuffle)
    
            
    def ft(self,n_neurons):
        """
            冻结vgg16出去最后一层softmax 1000的全连接层，改为 传递进去的神经元个数
            Replace the last layer of the model with a Dense (fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn
            weights for the last layer in subsequent training.

            Args:
                num (int) : Number of neurons in the Dense layer
            Returns:
                None
        """
        model = self.model
        model.pop()
        for layer in model.layers:
            layer.trainable = False # 这些层不训练
        model.add(Dense(n_neurons,activation='softmax'))
        self.compile()
        
        
    
    def finetune(self,batches):
        """
            # 微调模型,更新self.classes 不同的数据需要变动这里。
            Modifies the original VGG16 network architecture and updates self.classes for new training data.
            
            Args:
                batches : A keras.preprocessing.image.ImageDataGenerator object.
                          See definition for get_batches().
        """
        self.ft(batches.num_classes) # 获得数据的类别个数
        classes = list(iter(batches.class_indices)) # get a list of all the class labels
        
        # batches.class_indices is a dict with the class name as key and an index as value
        # eg. {'cats': 0, 'dogs': 1}

        # sort the class labels by index according to batches.class_indices and update model.classes
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes
        
            
    def compile(self,lr=0.01):
        """
            用于配置训练模型。
            Configures the model for training.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.compile(optimizer=Adam(lr=lr),loss='categorical_crossentropy',metrics=['accuracy'])
        
        
        
    def fit_data(self,X,y,val,val_lables,batch_size=64,n_epochs=3):
        """
            # 训练模型
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.fit(X,y,validation_data=(val,val_labels),epochs=n_epochs,batch_size=batch_size)
    
    
    def fit(self,batches,val_batches,n_epochs=3):
        """
            使用 Python 生成器逐批生成的数据，按批次训练模型。
            Fits the model on data yielded batch-by-batch by a Python generator.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.fit_generator(batches,validation_data=val_batches,epochs=n_epochs,verbose=1)
        
        
    def test(self,path,batch_size=8):
        """
            测试
            Predicts the classes using the trained model on data yielded batch-by-batch.

            Args:
                path (string):  Path to the target directory. It should contain one subdirectory 
                                per class.
                batch_size (int): The number of images to be considered in each batch.
            
            Returns:
                test_batches, numpy array(s) of predictions for the test_batches.
    
        """
        test_batches = self.get_batches(path,shuffle=False,batch_size=batch_size,class_mode=None)
        return test_batches,self.model.predict_generator(test_batches)