import matplotlib.pyplot as plt

def pm(x):
	fig, ax=plt.subplots()
	ax.matshow(abs(x))
	plt.show()
	plt.close()
