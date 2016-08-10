import theano
from theano import tensor as T
import numpy as np
import scipy.io as sio
import gzip

DATA_PATH = "data/seq_data.mat"
IMAGE_SZ = 28
np.random.seed(777)


def load_dna_dataset():
    d = sio.loadmat(DATA_PATH)
    data = d['data']
    labels = d['labels']
    return (data, labels)


def prepare_dna_dataset(data, labels, ratio=75):
    vec_data = [np.reshape(x, (1, 4000)) for x in data]

    # shuffle
    XY = zip(vec_data, labels)
    np.random.shuffle(XY)

    # split the data
    n = len(vec_data)
    sz_train = n * ratio // 100
    training = XY[:sz_train]
    validation = XY[sz_train:]

    # transform to list
    train_X = [d[0] for d in training]
    train_Y = [d[1].reshape(1, 4) for d in training]

    val_X = [d[0] for d in validation]
    val_Y = [d[1].reshape(1, 4) for d in validation]
    return (train_X, train_Y, val_X, val_Y)


def load_mnist_dataset():
    images_path = "data/train-images-idx3-ubyte.gz"
    labels_path = "data/train-labels-idx1-ubyte.gz"
    # load data
    with gzip.open(str(images_path)) as handle:
        X = (np.frombuffer(handle.read(), np.uint8, offset=16))
    # normalize to make [0, 1)
    X_trans = X / floatX(255)

    # load labels
    with gzip.open(str(labels_path)) as handle:
        y = (np.frombuffer(handle.read(), np.uint8, offset=16))
    return (X_trans, y)


def prepare_mnist_dataset(data, labs, ratio=75):
    data = data.reshape(-1, 28 * 28)
    X_train = [np.reshape(k, (1, 28 * 28)) for k in data]
    y_train = [label_to_vector(k).reshape(1, 10) for k in labs]

    # split the data
    n = len(data)
    set_sz = n * ratio // 100

    return X_train[:set_sz], y_train[:set_sz], X_train[set_sz:], y_train[set_sz:]


def label_to_vector(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)


def init_weghts(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


def model(X, w):
    return T.nnet.softmax(T.dot(X, w))


# data, labels = load_data()
# train_X, train_Y, val_X, val_Y = prepare_data(data, labels)

data, labs = load_dna_dataset()
train_X, train_Y, val_X, val_Y = prepare_dna_dataset(data, labs)

X = T.matrix()
y = T.matrix()

ndim = train_X[0].shape[1]
nclass = train_Y[0].shape[1]
w = init_weghts((ndim, nclass))

py_x = model(X, w)  # P(y|X, w)
y_pred = T.argmax(py_x, axis=1)

# objective function
cost = T.mean(T.nnet.categorical_crossentropy(py_x, y))
gradient = T.grad(cost=cost, wrt=w)
update = [(w, w - gradient * 0.05)]

train = theano.function(inputs=[X, y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred)

# training the model
batch = 200
epoch = 30
error_stat = []
for i in xrange(epoch):
    now_error = []
    for (start, end) in zip(range(0, len(train_X), batch), range(batch, len(train_X), batch)):
        now_error.append(train(np.vstack(train_X[start:end]), np.vstack(train_Y[start:end])))
    error_stat.append(np.mean(now_error))

print error_stat


# evaluate the model
def evaluate(data, labs):
    test_results = [predict(x)[0] for x in data]
    labs_scalar = [np.argmax(y) for y in labs]
    return sum(int(x == y) for (x, y) in zip(test_results, labs_scalar))


num_acc = evaluate(val_X, val_Y)
print "Reconized: %d/%d, Accuracy = %f" % (num_acc, len(val_X), float(num_acc) / len(val_X))
