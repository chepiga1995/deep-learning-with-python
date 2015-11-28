SIZE = 28

def output(img):
	out = []
	for x in range(SIZE):
		for y in range(SIZE):
			res = ord(img[x*SIZE + y]) #/ 255
			# if ord(img[x*SIZE + y]) < 128:
			# 	res = 0 
			out.append(res)
	return out   


def get_from_file_train(file, number_get):
	with open(file, 'rb') as f:
	    r = f.read(16)
	    code = ord(r[0]) * 2 ** 32 + ord(r[1]) * 2 ** 16 + ord(r[2]) * 2 ** 8 + ord(r[3])
	    number = ord(r[4]) * 2 ** 32 + ord(r[5]) * 2 ** 16 + ord(r[6]) * 2 ** 8 + ord(r[7])
	    rows = ord(r[8]) * 2 ** 32 + ord(r[9]) * 2 ** 16 + ord(r[10]) * 2 ** 8 + ord(r[11])
	    columns = ord(r[12]) * 2 ** 32 + ord(r[13]) * 2 ** 16 + ord(r[14]) * 2 ** 8 + ord(r[15])
	    train = []
	    if number_get > number or SIZE != rows or SIZE !=columns:
	    	raise Exception("Incorect parametr")
	    for i in range(number_get):
	    	img = f.read(SIZE * SIZE)
	    	train.append(output(img))
	    return train	
def get_from_file_res(file, number_get):
	with open(file, 'rb') as f:
	    r = f.read(8)
	    code = ord(r[0]) * 2 ** 32 + ord(r[1]) * 2 ** 16 + ord(r[2]) * 2 ** 8 + ord(r[3])
	    number = ord(r[4]) * 2 ** 32 + ord(r[5]) * 2 ** 16 + ord(r[6]) * 2 ** 8 + ord(r[7])
	    labels = []
	    if number_get > number:
	    	raise Exception("Incorect parametr")
	    for i in range(number_get):
	    	img = f.read(1) 
	    	labels.append(ord(img[0]))
	    train_res = []	
	    for i in range(len(labels)):
			train_res.append([+(labels[i]==j) for j in range(10)])	
	    return train_res