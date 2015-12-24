import time
startt = time.time()
import numpy as np
import load
from neural import Neural
from open_file import procces_file
from mel_from_mlf.transform import get_neighboring_mel

SAVE_PATH_SER = '/dev/Deep-learning/'
DIRECTORY_PATH = '/home/ura/Documents/Deep-learning/'
FILE = "test_wav/Vocaroo_s0JVyDMnSJYu"
WEIGHTS_FILE = 'weights_file.db'
ARR_SHAPE = [\
    (100, 800),\
    (800, 800),\
    (800, 800),\
    (800, 800),\
    (800, 28)\
]
TIMES = 100
SHORTCARD = 256
SPEED = 0.01
DROP_INPUT = 0.2
DROP_HIDDEN = 0.6
PRINT_FROM = 10
PRINT_TO = 50
STEP_SHOW = 3
RHO = 0.9
EPSILON = 1e-4
STEP = 50
print "test"

trX, teX = load.loadMel("afterLDA100.db")
# trY, teY = load.loadRes()
trZ, teZ = load.loadLeter()
print "loaded"
neur = Neural()
# neur.load_from_file(WEIGHTS_FILE)
neur.init_model(ARR_SHAPE)
neur.train(trX, teX, trZ, teZ, epochs=TIMES)

neur.save_to_file(WEIGHTS_FILE)


# trT = neur.result(trX)
# teT = neur.result(teX)
# print "start train"
# neur2 = Neural()
# neur2.init_model([(55, 300), (300, 28)])
# neur2.train(trT, teT, trZ, teZ, epochs=TIMES)

# neur = Neural()
# neur.load_from_file(DIRECTORY_PATH + WEIGHTS_FILE)
# mel, res = procces_file(FILE)
# print res
# def translate(neur, mel):
#     res = ['sil']
#     for x in range(len(mel)):
#         result = np.concatenate(get_neighboring_mel(mel, x, 4))
#         # resu = SOUND_MAP[neur.predict([result])[0]]
#         resu = neur.result([result])[0]
#         for y in resu:
#             if y > 0.05:
#                 print "{0:} {1:.1f}  ".format(SOUND_MAP[resu.tolist().index(y)], y * 100),
#         print        
#         # print resu, "  ", 
#         # if res[-1] != resu:
#         #     res.append(resu)
#         #     print resu, "  ", 
        
endt = time.time()
# translate(neur, mel)
print endt - startt