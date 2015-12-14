import time
startt = time.time()
from load import SOUND_MAP, loadMel, loadRes
from neural import Neural
from hmmlearn.hmm import GaussianHMM


SAVE_PATH_SER = '/dev/Deep-learning/'
WEIGHTS_FILE = SAVE_PATH_SER + 'weights_filr.db'
ARR_SHAPE = [\
    (117, 800),\
    (800, 800),\
    (800, 800),\
    (800, 800),\
    (800, 53)\
]
TIMES = 10
SHORTCARD = 256
SPEED = 0.01
DROP_INPUT = 0.2
DROP_HIDDEN = 0.6
PRINT_FROM = 10
PRINT_TO = 50
STEP_SHOW = 3
RHO = 0.9
EPSILON = 1e-4
STEP = 30000
print "test"

trX, teX = loadMel()
trY, teY = loadRes()


print len(trY), len(teY) 

neur = Neural()

neur.load_from_file(WEIGHTS_FILE)
train_arr = neur.result(trX).tolist()
# train_arr = map(lambda x: SOUND_MAP[x], train_arr)
# print train_arr[:5]
model = GaussianHMM(n_components=53, covariance_type="diag", n_iter=10)

for start, end in zip(range(0, len(trY), STEP), range(STEP, len(trY), STEP)):
    model.fit(train_arr[start:end])
    endt = time.time()
    print endt - startt

print model.predict(train_arr[:50]).tolist()
print map(lambda x: SOUND_MAP[x.index(1)],trY[:50].tolist())
# neur.train(trX, teX, trY, teY, plot=True, epochs=TIMES, shortcard=SHORTCARD, speed=SPEED, drop_input=DROP_INPUT, drop_hidden=DROP_HIDDEN, step_show=STEP_SHOW, rho=RHO, epsilon=EPSILON)

# neur.save_to_file()

# neur.print_res(teX, teY, SOUND_MAP)
endt = time.time()
print endt - startt