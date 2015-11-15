from get_from_file_function import get_from_file_train, get_from_file_res, SIZE

TRAIN_SIZE = 500
TEST_SIZE = 300



train_img = get_from_file_train('files/train-images.idx3-ubyte', TRAIN_SIZE)
train_res = get_from_file_res('files/train-labels.idx1-ubyte', TRAIN_SIZE)
test_img = get_from_file_train('files/t10k-images.idx3-ubyte', TEST_SIZE)
test_res = get_from_file_res('files/t10k-labels.idx1-ubyte', TEST_SIZE)