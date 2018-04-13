import numpy as np
import pandas as pd
import os
from keras import layers, models, optimizers, losses, regularizers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from shared.logger import get_logger, get_start_time, get_filename, get_curr_time
from shared.plot_history import plot_all
from shared.utility import open_plot
from shared.metrics import f1_score

OUTPUT_DIR="output/titanic"
DRO = 0.0
L2R = 0.0001
ILAYER = 1024
HLAYER = 512
EPOCHS = 75
BATCH_SIZE = 256
K_FOLDS = 5
READY = False

histories = []
logger = get_logger()

logger.info("running on {}".format(os.name))
logger.info("loading data")
cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
dtypes = {
    'PassengerId': 'int64',
    'Pclass': 'int64',
    'Age': 'float32',
    'SibSp':'int64',
    'Parch':'int64'
}
cols_tr = ['Survived'] + cols
dtypes_tr = dtypes.copy()
dtypes_tr['Survived'] = 'float32'
train_df = pd.read_csv("data/titanic/train.csv", usecols=cols_tr, index_col = 'PassengerId', dtype=dtypes)
test_df = pd.read_csv("data/titanic/test.csv", usecols=cols, index_col = 'PassengerId', dtype=dtypes_tr)
train_df = train_df.sample(frac=1)

## PREPROCESS ##

# Impute
logger.info("imputing data")
train_df.Embarked.fillna('S', inplace=True)
test_df.Embarked.fillna('S', inplace=True)
train_df.Age.fillna(train_df.Age.mean(), inplace=True)
test_df.Age.fillna(train_df.Age.mean(), inplace=True)

# Labels to Categories
logger.info("categorizing data")
le_sex = LabelEncoder()
le_sex.fit(train_df.Sex)
le_emb = LabelEncoder()
le_emb.fit(train_df.Embarked)

train_df.Sex = le_sex.transform(train_df.Sex)
test_df.Sex = le_sex.transform(test_df.Sex)

train_df.Embarked = le_emb.transform(train_df.Embarked)
test_df.Embarked = le_emb.transform(test_df.Embarked)

# Scale
logger.info("scaling data")
ss = StandardScaler()
cont_cols = ['Age', 'SibSp', 'Parch']
ss.fit(train_df[cont_cols])
train_df[cont_cols] = ss.transform(train_df[cont_cols])
test_df[cont_cols] = ss.transform(test_df[cont_cols])

y_train = np.asarray(train_df.Survived, dtype='float32')
X_train = train_df.drop('Survived', axis=1).as_matrix()
X_test = test_df.as_matrix()

# One Hot
logger.info("encoding data")
cats = [True, False, False, False, False, True]
encoder = OneHotEncoder(sparse=False, categorical_features = cats, handle_unknown='ignore')
encoder.fit(X_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

logger.info("building model")
model = models.Sequential()
model.add(layers.Dense(ILAYER,
                       activation='relu',
                       input_shape=(X_train.shape[1],),
                       kernel_regularizer=regularizers.l2(L2R)))
model.add(layers.Dropout(DRO))
model.add(layers.Dense(HLAYER, activation='relu', kernel_regularizer=regularizers.l2(L2R)))
model.add(layers.Dropout(DRO))
model.add(layers.Dense(HLAYER, activation='relu', kernel_regularizer=regularizers.l2(L2R)))
model.add(layers.Dropout(DRO))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy', f1_score])

num_val_samples = len(train_df) // K_FOLDS

logger.info(model.summary())

logger.info("validating model")
for i in range(K_FOLDS):
    logger.info("cross validation fold {}".format(i))
    val_data = X_train[i * num_val_samples : (i+1) * num_val_samples,:]
    val_labels = y_train[i * num_val_samples : (i+1) * num_val_samples]

    partial_train_data = np.concatenate([X_train[:(i * num_val_samples)],
                                        X_train[((i+1) * num_val_samples):]],
                                        axis=0)

    partial_train_label = np.concatenate([y_train[:(i * num_val_samples)],
                                        y_train[((i+1) * num_val_samples):]],
                                         axis=0)

    history = model.fit(partial_train_data,
                        partial_train_label,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(val_data, val_labels),
                        shuffle=True,
                        verbose=0)
    histories.append(history)

logger.info("saving plot of loss and accuracy")
plots = plot_all(histories, metrics = {'acc': 'Accuracy', 'f1_score': 'F1'})
pname = "{}/{}-{}.png".format(OUTPUT_DIR, get_filename(), get_start_time())
plots.savefig(pname)
logger.info("created {}".format(pname))
logger.info("finished at {}".format(get_curr_time()))

open_plot(pname)

if READY:
    logger.info("Ready is True, so training on all data and writing submission file")
    model.fit(X_train,
              y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE)

    logger.info("saving test results")
    test_df['Survived'] = np.round(model.predict(X_test)).astype('int')
    fname = "{}/{}-{}.csv".format(OUTPUT_DIR, get_filename(), get_start_time())
    test_df.to_csv(fname, index=True, columns = ['Survived'])
    logger.info("created {}".format(fname))
