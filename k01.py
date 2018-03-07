import numpy as np
import pandas as pd
import os
from keras import layers, models, optimizers, losses, regularizers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from shared.logger import get_logger, get_start_time, get_filename, get_curr_time
from shared.plot_history import plot_all
from shared.utility import open_plot

OUTPUT_DIR="output/titanic"
DRO = 0.5
L2R = 0.001
EPOCHS = 100
BATCH_SIZE = 256

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
encoder = OneHotEncoder(categorical_features = cats, handle_unknown='ignore')
encoder.fit(X_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

logger.info("building model")
model = models.Sequential()
model.add(layers.Dense(512,
                       activation='relu',
                       input_shape=(X_train.shape[1],),
                       kernel_regularizer=regularizers.l2(L2R)))
model.add(layers.Dropout(DRO))
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2R)))
model.add(layers.Dropout(DRO))
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(L2R)))
model.add(layers.Dropout(DRO))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['accuracy'])

logger.info("fitting model")
history = model.fit(X_train.todense(),
                   y_train,
                   epochs=EPOCHS,
                   batch_size=BATCH_SIZE,
                   validation_split=.2,
                   shuffle=True)

logger.info("saving results")
test_df['Survived'] = np.round(model.predict(X_test.todense())).astype('int')
fname = "{}/{}-{}.csv".format(OUTPUT_DIR, get_filename(), get_start_time())
test_df.to_csv(fname, index=True, columns = ['Survived'])
logger.info("created {}".format(fname))

logger.info("saving plot of loss and accuracy")
plots = plot_all(history)
pname = "{}/{}-{}.png".format(OUTPUT_DIR, get_filename(), get_start_time())
plots.savefig(pname)
logger.info("created {}".format(pname))
logger.info("finished at {}".format(get_curr_time()))

open_plot(pname)
