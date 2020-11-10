# -*-coding:utf-8 -*-
from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
# from keras import optimizers
# from keras import metrics
import csv
# import sys

# the parth need to reset
BASE_DIR = '/Corpus'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/offtopic/Technology/'

MAX_SEQUENCE_LENGTH = 1000  # the length of the essay
# MAX_NB_WORDS = 20000
MAX_NB_WORDS = 8000     # max length of the whole words
# MAX_NB_WORDS = 2500
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# def precision(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision1 = true_positives / (predicted_positives + K.epsilon())
#     return precision1
#
# def recall(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall1 = true_positives / (possible_positives + K.epsilon())
#     return recall1
#
# def fbeta_score(y_true, y_pred, beta=1):
#     if beta < 0:
#         raise ValueError('The lowest choosable beta is zero (only precision).')
#
#     # If there are no true positives, fix the F score at 0 like sklearn.
#     if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
#         return 0
#
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     bb = beta ** 2
#     fbeta_score1 = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
#     return fbeta_score1
#
# def fmeasure(y_true, y_pred):
#     return fbeta_score(y_true, y_pred, beta=1)

"""read wordembeddings shape like {word:embeddings}"""
def IndexVectors():
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR,'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

"""rea csv files"""
def read_csv():
    print('Processing text dataset')
    texts = []
    labels = []
    labels_index = {'false':0, 'true':1}
    with open('the_effects_computers_have_on_people.csv', "rb") as f:
        reader = csv.reader(f)
        for line in reader:
            if int(line[1]) > 0:
                texts.append(line[0])
                labels.append(1)
            else:
                texts.append(line[0])
                labels.append(0)

    print('Found %s texts.' % len(texts))
    print('Found %s lables' % len(labels))
    print('Found %s kinds of label' % len(labels_index))
    return texts, labels, labels_index

"""test tfidf"""
def texts_word_tfidf(texts):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
    tv = TfidfVectorizer(decode_error="ignore",norm=None,sublinear_tf=True)
    data = tv.fit_transform(texts)
    print (data.shape)      # (2054, 7600) 7600 is the whole number of the words
    print (data.shape[0],"---",data.shape[1],"\n")


    print (data[0].data)      # <type 'numpy.ndarray'>  last data includings all essays; next data is one of essay; data[0].data[0]means the first word's TFIDF of the first essay.
    print (data[0])           # <class 'scipy.sparse.csr.csr_matrix'>

"""preprocessing of the text"""
def tokenizer(texts, labels):
    # num_words=MAX_NB_WORDS
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # print("word_counts:{}".format(tokenizer.word_counts))
    # print("word_docs:{}".format(tokenizer.word_docs))
    # print("num_words:{}, length of word_index is {}".format(tokenizer.num_words, len(tokenizer.word_docs)))
    # print("document_count:{}".format(tokenizer.document_count))

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding="post")

    labels_result = np.asarray(labels, dtype='int32')

    labels = to_categorical(np.asarray(labels))

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    train_x = data[:-nb_validation_samples]
    train_y = labels[:-nb_validation_samples]
    test_x = data[-nb_validation_samples:]
    test_y = labels[-nb_validation_samples:]

    labels_result = labels_result[indices]
    test_labels_result = labels_result[-nb_validation_samples:]
    return word_index,train_x,train_y,test_x,test_y,test_labels_result

"""preprocessing+tfidf"""
def tokenizer_tfidf(texts, labels):
    # num_words=MAX_NB_WORDS
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    matrixs = tokenizer.sequences_to_matrix(sequences,mode='tfidf')
    matrixs = pad_sequences(matrixs,maxlen=MAX_SEQUENCE_LENGTH,dtype='float32',padding="post",truncating="post")
    # for matrix in matrixs:
    #     print (type(matrix))
    # print (len(matrixs))
    # print (len(np.asarray(matrixs[0])))

    # word_index
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    # main input pad
    data = pad_sequences(sequences,maxlen=MAX_SEQUENCE_LENGTH,padding="post")
    labels_result = np.asarray(labels,dtype='int32')
    labels = to_categorical(np.asarray(labels),2)
    # labels = to_categorical(np.asarray(labels))
    # labels = np.asarray(labels,dtype='int')
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    matrixs = matrixs[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
    train_x = data[:-nb_validation_samples]
    train_y= labels[:-nb_validation_samples]
    test_x = data[-nb_validation_samples:]
    test_y = labels[-nb_validation_samples:]

    train_matrixs = matrixs[:-nb_validation_samples]
    test_matrixs = matrixs[-nb_validation_samples:]

    labels_result = labels_result[indices]
    test_labels_result = labels_result[-nb_validation_samples:]
    return word_index,train_x,train_y,test_x,test_y,train_matrixs,test_matrixs,test_labels_result

"""embedding layer"""
def embeddingmatrix(word_index,embeddings_index):
    print('Preparing embedding matrix.')

    nb_words = min(MAX_NB_WORDS, len(word_index))     # note: here nb_words from embedding layer should be same as nb_words from tokenizer
    # nb_words = len(word_index)
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:    # MAX_NB_WORDS nb_words
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(input_dim=nb_words + 1,
                                output_dim=EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    # learn from add_weight of layer
    # length of embedding_matrix equals to input_dim
    # width of embedding_matrix equals to output_dim
    # that is the most common words assigned before(MAX_NB_WORDS)
    return embedding_layer

"""CNN+tfidf model"""
def modeltrain_tfidf(embedding_layer, labels_index, train_x, train_y, test_x, test_y, train_matrixs, test_matrixs, y_true):
    print('Training model.')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="sequence_input")
    tfidf_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='float32', name="tfidf_input")      # predefine a new input layer

    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    x = concatenate([x, tfidf_input])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    # model = Model(sequence_input, preds)
    model = Model(inputs=[sequence_input, tfidf_input], outputs=preds)
    # categorical_crossentropy
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])  # 'binary_accuracy','accuracy'

    model.fit({'sequence_input':train_x, 'tfidf_input':train_matrixs}, train_y,
              validation_data=([test_x, test_matrixs], test_y), epochs=5, batch_size=32)

    # model.save('test.h5')

    score = model.evaluate([train_x,train_matrixs], train_y, verbose=0)
    print('Train score:', score[0])
    print('Train accuracy:', score[1])

    score = model.evaluate([test_x,test_matrixs], test_y, verbose=0)
    print('Test score:', score[0])
    print ('Test accuracy', score[1])

    #create a picture
    # from keras.utils import plot_model
    # plot_model(model, to_file='cnn_offtopic_papaer2.png',show_shapes=False)

    # output the results
    result = model.predict([test_x, test_matrixs], batch_size=32, verbose=0)
    if result.shape[-1] > 1:
        ypreds = (result.argmax(axis=-1))
    else:
        ypreds = ((result > 0.5).astype('int32'))
    getResult(y_true,ypreds)

"""three layers of cnn model"""
def modeltrain(embedding_layer, labels_index, train_x, train_y, test_x, test_y, y_true):
    print('Training model.')
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="sequence_input")
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    # sgd = optimizers.SGD(lr=0.001)
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])     # rmsprop

    model.fit({'sequence_input':train_x}, train_y,  epochs=3, batch_size=32)    # validation_data=(test_x, test_y),

    score = model.evaluate(train_x, train_y, verbose=0)
    print('Train score:', score[0])
    print('Train accuracy:', score[1])

    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # output results
    result = model.predict(test_x, batch_size=32, verbose=0)
    if result.shape[-1] > 1:
        ypreds = (result.argmax(axis=-1))
    else:
        ypreds = ((result > 0.5).astype('int32'))
    getResult(y_true, ypreds)


def getResult(y_true, y_pred):
    from sklearn import metrics
    print("accuracy %s" % metrics.accuracy_score(y_true, y_pred))
    print("precision is {}".format(metrics.precision_score(y_true, y_pred, average=None)[0]))
    print("recall is {}".format(metrics.recall_score(y_true, y_pred, average=None)[0]))
    print("f1 is {}".format(metrics.f1_score(y_true, y_pred, average=None)[0]))
    print(metrics.classification_report(y_true, y_pred))
    print(metrics.precision_recall_fscore_support(y_true, y_pred))

"""max,min and average number of words in each prompt"""
def text_len(texts,labels):
    tt = []
    num_1 = 0
    num_0 = 0
    for text in texts:
        tt.append(len(text))
    for i in labels:
        if i == 1:
            num_1 += 1
        if i == 0:
            num_0 += 1
    print ("The longest text is {}".format(max(tt)))
    print ("The shortest text is {}".format(min(tt)))
    print ("The average text is {}".format(sum(tt)/len(tt)))
    new_tt = sorted(tt,reverse=True)
    print (new_tt[0],"---",new_tt[1])
    print ("1 : {}".format(num_1))
    print ("0: {}".format(num_0))

if __name__ == '__main__':
        embeddings_index = IndexVectors()
        texts, labels, labels_index =read_csv()
        # word_index, train_x, train_y, test_x, test_y, test_labels_result = tokenizer(texts, labels)
        # word_index, train_x, train_y, test_x, test_y, train_matrixs, test_matrixs, test_labels_result = tokenizer_tfidf(texts, labels)
        # embedding_layer = embeddingmatrix(word_index,embeddings_index)
        # modeltrain_tfidf(embedding_layer, labels_index, train_x, train_y, test_x, test_y, train_matrixs, test_matrixs, test_labels_result)
        # modeltrain(embedding_layer, labels_index, train_x, train_y, test_x, test_y, test_labels_result)

