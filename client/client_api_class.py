import json
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers import dp_optimizer_keras
import flwr as fl
import cv2
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.models import Model
from keras.layers import *
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet import ResNet50, ResNet101, ResNet152
from keras.applications.inception_v3 import InceptionV3 
from keras.applications.mobilenet import MobileNet 
from keras.applications.mobilenet_v2 import MobileNetV2 
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge 
from keras.applications.densenet import DenseNet169 
from client import Client

class ClientApi():
    def loadConfig(self):
        print("config json is importing ------")
        ##  Load config json
        with open('config_training.json','r') as file:
            json_data = file.read()
        data = json.loads(json_data)
        self.data_categories = data["data_categories"]
        self.img_width = data["img_width"]
        self.img_height = data["img_height"]
        self.img_dim = data["img_dim"]
        self.imbalanced_type = data["imbalanced_type"]
        self.l2_norm_clip = data['df_l2_norm_clip']
        self.noise_multiplier = data['df_noise_multiplier']
        self.num_microbatches = data['df_num_microbatches']
        self.df_optimizer_type = data["df_optimizer_type"]
        self.fl_num_rounds = data['fl_num_rounds']
        self.fl_min_fit_clients = data['fl_min_fit_clients']
        self.fl_min_evaluate_clients = data['fl_min_evaluate_clients']
        self.fl_min_available_clients = data['fl_min_available_clients']
        self.fl_aggregate_type = data['fl_aggregate_type']
        self.fl_server_address = data['fl_server_address']
        self.batch_size = data['batch_size']
        self.learning_rate = data['learning_rate']
        self.clt_local_epochs = data['clt_local_epochs']
        self.clt_data_path = data['clt_data_path']
        self.model_name = data['model_name']
        print("config json is imported ------")


    def dataImblanced(self, X_train, y_train):

        # Define balanced data generator
        def getSMOTEData(X, y, random = 42):
            smote = SMOTE(random_state= random)
            return smote.fit_resample(X, y)


        def getOSData(X, y, random = 42):
            os = RandomOverSampler(random_state= random)
            return os.fit_resample(X, y)


        def getUSData(X, y, random = 42):
            us = RandomUnderSampler(random_state= random)
            return us.fit_resample(X, y)
        
       
        batch_size, width, height  = X_train.shape
        
        print("imbalanced ",self.imbalanced_type)
        
        match self.imbalanced_type:
            case 0:
                return X_train, y_train
            case 1:
                 return getSMOTEData(X_train.reshape(batch_size, width * height), y_train)
            case 2:
                return getOSData(X_train.reshape(batch_size, width * height), y_train)
            case 3:
                return getUSData(X_train.reshape(batch_size, width * height), y_train)

    def get_base_model(self, model_name, img_dim = 1):
        if img_dim == 1:
            img_input = Input(shape=(self.img_width, self.img_height, 1))
            img_conc = Concatenate()([img_input, img_input, img_input])  

            if model_name == 'Xception' :
                base_model = Xception(input_tensor=img_conc, weights='imagenet', include_top=False)
                
            elif model_name =='ResNet50':
                base_model = ResNet50(input_tensor=img_conc, weights='imagenet', include_top=False)
            
            elif model_name =='ResNet101':
                base_model = ResNet101(input_tensor=img_conc, weights='imagenet', include_top=False)

            elif model_name =='ResNet152':
                base_model = ResNet152(input_tensor=img_conc, weights='imagenet', include_top=False)

            elif model_name =='Inceptionv3':
                base_model = InceptionV3(input_tensor=img_conc, weights='imagenet', include_top=False)
        
            elif model_name =='InceptionResNetV2':
                base_model = InceptionResNetV2(input_tensor=img_conc, weights='imagenet', include_top=False)
        
            elif model_name =='MobileNet':
                base_model = MobileNet(input_tensor=img_conc, weights='imagenet', include_top=False)
            
            elif model_name =='MobileNetV2':
                base_model = MobileNetV2(input_tensor=img_conc, weights='imagenet', include_top=False)

            elif model_name =='VGG16':
                base_model = VGG16(input_tensor=img_conc, weights='imagenet', include_top=False)

            elif model_name =='VGG19':
                base_model = VGG19(input_tensor=img_conc, weights='imagenet', include_top=False)
            
            elif model_name =='NASNetLarge':
                base_model = NASNetLarge(input_tensor=img_conc, weights='imagenet', include_top=False)
        
            elif model_name =='DenseNet169':
                base_model = DenseNet169(input_tensor=img_conc, weights='imagenet', include_top=False)
            
        return base_model



    def generate_cnn_model(self):
        match self.df_optimizer_type :
            case 0:
                optimizer = "adam"
            case 1:
                optimizer=dp_optimizer_keras.DPKerasAdamOptimizer(
                l2_norm_clip= float(self.l2_norm_clip),
                noise_multiplier= float(self.noise_multiplier),
                num_microbatches= self.num_microbatches)      
            case 2:
                optimizer=dp_optimizer_keras.DPKerasSGDOptimizer(
                l2_norm_clip= float(self.l2_norm_clip),
                noise_multiplier= float(self.noise_multiplier),
                num_microbatches= self.num_microbatches)      
            case 3:
                optimizer=dp_optimizer_keras.DPKerasAdagradOptimizer(
                l2_norm_clip= float(self.l2_norm_clip),
                noise_multiplier= float(self.noise_multiplier),
                num_microbatches= self.num_microbatches) 

        base_model = self.get_base_model(self.model_name)

        #  get the output of the second last dense layer 
        base_model_output = base_model.layers[-1].output
        x = Flatten(name = "flate1")(base_model_output)

        # add new layers 
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        # x = Dense(256, activation='relu')(x)
        # x = Dropout(0.4,name='drop2')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3,name='drop3')(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(len(self.data_categories), activation='sigmoid', name='fc3')(x)

        # define a new model 
        self.model_architecture = Model(base_model.input, output)

        # Freeze all the base model layers 
        for layer in base_model.layers:
            layer.trainable=False

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        #optimizer = Adam(lr=0.001, decay=1e-6, beta_1=0.9, epsilon=None, amsgrad=True)
        #optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        self.model_architecture.compile(loss='sparse_categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
        return self.model_architecture

    def load_img(self, data_type, datadir):
        img_arr = []
        target_arr = []
        datadir = datadir + data_type
        Categories = self.data_categories
        
        for i in Categories:
            print(f'loading... category : {i}')
            path = datadir + "/" + i
            
            for img_file in os.listdir(path):
                try: 
                    # Đọc ảnh với OpenCV
                    img = cv2.imread(os.path.join(path, img_file),cv2.IMREAD_GRAYSCALE)
                    
                    # Resize ảnh về kích thước 64x64
                    img = cv2.resize(img, (int(self.img_width), int(self.img_height)))
                    
                    # Thêm ảnh vào mảng img_arr
                    img_arr.append(img)
                    
                    # Thêm nhãn tương ứng vào mảng target_arr
                    target_arr.append(Categories.index(i))
                except Exception as e:
                    print(str(e))
            print(f'loaded category: {i} successfully')
        
        # Chuyển đổi các mảng thành mảng NumPy
        img_arr = np.array(img_arr)
        target_arr = np.array(target_arr)
        return img_arr, target_arr

    def launch_fl_session(self, NoClient: str, path: str, isAttacker = False):
        X_train,y_train= self.load_img(path +'/train/', self.clt_data_path)

        X_train,y_train=self.dataImblanced( X_train,y_train)

        X_train = X_train.reshape(X_train.shape[0], self.img_width, self.img_height)

        X_test,y_test=self.load_img(path +'/test/', self.clt_data_path)

        fl.client.start_numpy_client(server_address=self.fl_server_address, client=Client(self.model_architecture  ,X_train, y_train, X_test, y_test, NoClient, isAttacker))

    def __init__(self) -> None:
        self.loadConfig()
        self.generate_cnn_model()



