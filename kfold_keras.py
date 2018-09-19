import os
import time
import copy

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import average,Input
from keras.models import Model,clone_model
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def k_fold_keras_early_stop(model,X,y,k=10,epochs=200,batch_size=128,datagen=None,
                        random_state=None,name='k-fold-CV',train_at_end=False, patience=3):
    """perform k-fold crossvaildation on keras models, with early stopping

    this works with generator and normal datasets
    
    Parameters
    ----------
    model : keras.model
        the model to be trained
    X : array_like
        the training data
    y : array_like
        the training labels
    k : int, optional
        number of validation sets (the default is 10, which [default_description])
    epochs : int, optional
        maximum number of epochs (the default is 200, which [default_description])
    batch_size : int, optional
        batch size during training (the default is 128)
    datagen : generator, optional
        if not None a generator for data augmentation is applied to the dataset before training (the default is None)
    random_state : int, optional
        random state (the default is None)
    name : str, optional
        name (the default is 'k-fold-CV')
    train_at_end : bool, optional
        return an ensemble of the models trained on the k-fold (the default is True)
    patience : int
        Patience setting for the early stopping,(the default is 3)
    Returns
    -------
    tuple
        the scores (cvscores, roc_aucs,episodes)
    """
    # define k-fold cross validation test 
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    # init metrics
    cvscores = []
    episodes = []
    roc_aucs = []
    val_loss = []  
    # save the initialised weights
    tmp_file = 't'+str(time.time()*1000)+'.h5'
    model.save_weights(tmp_file)

    ensemble_list =  []
    with tqdm(total=k,desc=name.ljust(20)) as pbar:
        for train, val in kfold.split(X, y):
            # Fit the initialised model
            model.load_weights(tmp_file)
            earlyStopping = EarlyStopping(monitor='val_acc',mode='auto',verbose=0, patience=patience)
            if datagen:
                train_generator=datagen.flow(X[train], y[train], batch_size=batch_size)
                training = model.fit_generator(train_generator, validation_data=(X[val], y[val]),
                    epochs=epochs,  callbacks=[earlyStopping], verbose=0) 
            else:
                training = model.fit(X[train], y[train], validation_data=(X[val], y[val]),
                    epochs=epochs, batch_size=batch_size, callbacks=[earlyStopping], verbose=0) 

            # evaluate the model
            scores = model.evaluate(X[val], y[val], verbose=0)
            cvscores.append(scores[1] * 100)
            y_predict = model.predict(X[val])
            roc_aucs.append(roc_auc_score(y[val],y_predict))
            episodes.append(len(training.history['val_loss']))
            y_predict = y_predict.astype('float64') # apparently without this log_loss sometimes creates NANs
            val_loss.append(log_loss(y[val],y_predict,eps=1e-15))
            pbar.update(1)
            pbar.set_postfix(Acc=round(np.mean(cvscores),4),vloss=np.mean(val_loss),
                        ROC_AUC=np.mean(roc_aucs),Epi=np.mean(episodes))
            if train_at_end:
                model_copy = clone_model(model)
                model_copy.set_weights(model.get_weights())
                ensemble_list.append(model_copy)
    pbar.close()
    os.remove(tmp_file)

    if train_at_end:
        ensemble_model = ensemble(ensemble_list)
        print('Created ensemble.')
        return (cvscores, roc_aucs,episodes,val_loss),ensemble_model

    else:
        ensemble_model = None
        return (cvscores, roc_aucs,episodes,val_loss)

def ensemble(models):
    """create an ensemble out of various keras models with the same input
    
    Parameters
    ----------
    models : list
        a list of the models to ensemble
    Returns
    -------
    keras.Model
        an ensemble of Keras models
    """
    input_layer= Input(shape=models[0].input_shape[1:])
    for idx,m in enumerate(models):
        m.name= f'model_{idx}'
    outputs = [m(input_layer) for m in models]
    yAvg= average(outputs)

    model = Model(inputs=input_layer, outputs=yAvg, name='ensemble')
    
    return model