

def EarlyStop(es, ws):
	firstLoss = es[0]
	threshold = firstLoss/100
	if len(es) == ws:
		for i in range(1, len(es)):
			if abs(es[i] - es[i-1]) > threshold:
				return False
		return True
	return False
