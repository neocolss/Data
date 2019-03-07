#If You encounter a problem to read the file 'data.scv' please the first line(the header) from 'data.scv'

#Exemple d'utilisation du code citÃ© dans l'article
#Exemple de deep learning CNN()
#Importation des package python
# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Importing the libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pprint import pprint
from time import time
from sklearn.model_selection import StratifiedKFold
from keras.layers import Dropout,BatchNormalization
from keras import regularizers
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.layers import Flatten
from sklearn.model_selection import cross_val_score
from keras.initializers import random_uniform
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
SEED = 2019
np.random.seed(SEED)

# Importing the dataset
# load dataset
names = ['is_ip','lexical_length_url','using_shortening_service','has_at','double_slash_redirect','has_minus','dots_number_in_domaine_name','ssl','Domain_registeration_length','Favico','port','https_in_domain_name','bad_request','Anchor','links_of_tag','server_handle_form','to_email','abnormal_structure','is_redirect','mouse_on_over','right_click_disabling','pop_up','iframe_use','age_of_domain','dns','popularity','Rank','Google_Index','page_links','Statistics_report','Result']

dataframe = pd.read_csv("data.csv", names=names)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:-1]
Y = dataset[:,-1]


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Reshape data in 3 dimensions (height width, canal = 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#print("X_train shape:", X_train.shape[0:3])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
#print("X_test shape:", X_test.shape[0:3])
#hyperparameters
input_dimension = 30
learning_rate = 0.0025
momentum = 0.85
hidden_initializer = random_uniform(seed=SEED)
dropout_rate = 0.15
kernel_initializer='uniform'
kernel_regularizer=regularizers.l2(0.0001)


# define path to save model
model_path = 'fm_cnn_BN.h5'

#CNN modele defini dans l'article
def create_model():
    conv = Sequential()
    
    conv.add(Conv1D(nb_filter=15, filter_length=3, input_shape = X_train.shape[1:3], activation = 'relu', kernel_regularizer=kernel_regularizer))
    conv.add(Conv1D(nb_filter=15, filter_length=1, activation='relu',kernel_regularizer=kernel_regularizer))
    conv.add(BatchNormalization())
    conv.add(MaxPooling1D(2))
    conv.add(Flatten())
    conv.add(Dropout(dropout_rate))
    
    conv.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu', kernel_regularizer=kernel_regularizer))
    conv.add(Dropout(dropout_rate))
    
    conv.add(Dense(256, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu', kernel_regularizer=kernel_regularizer))
    conv.add(Dropout(dropout_rate))
    
    conv.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu', kernel_regularizer=kernel_regularizer))
    conv.add(Dropout(dropout_rate))
    
    conv.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu',kernel_regularizer=kernel_regularizer))
    
    conv.add(Dense(1, activation = 'sigmoid', kernel_initializer = hidden_initializer))
    #sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    conv.compile(loss = 'binary_crossentropy', optimizer = "adam", metrics = ['accuracy'])
    return conv

#Visualisation du model
from keras.utils.vis_utils import plot_model
model = create_model()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

if __name__ == '__main__':
    #Callback nous permet d'arrÃªter l'execution du programme au cas oÃ¹ l'entrainement des donnÃ©es n'offre aucun approuvement
    callbacks=[EarlyStopping(monitor='val_acc', patience=0, mode='max'), 
               ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, mode='max',verbose=0)]
    #CrÃ©er le modÃ¨le
    conv = KerasClassifier(build_fn=create_model, epochs=100, batch_size=128, verbose=2)
    #Pipeline nous permet d'exÃ©cuter plusieurs traitements par ordre
    estimators = []
    estimators.append(('mlp',conv))
    pipeline = Pipeline(estimators)
    #Cross-Validation meilleur moyen pour entraÃ®ner un modÃ¨le
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    fit_params={'mlp__callbacks':callbacks}
    #Afficher les rÃ©sultats
    results = cross_val_score(pipeline, X_train, y_train, cv=kfold, n_jobs=6, fit_params=fit_params)
    print("Standardized: Accuracy:%.2f%% ; error rate:(%.2f%%)" % (results.mean()*100, results.std()*100))