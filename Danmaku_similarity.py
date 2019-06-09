import pdb
import numpy as np
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
from pylab import mpl
import subprocess
from adjustText import adjust_text

def calculate_similarity(feature1, feature2):
	# return np.inner(feature1, feature2)
	return cosine_similarity(feature1, feature2)

def return_message_list():
	return ['草', '哈哈哈', '待机', '88888', '耻辱下播', 'kksk', \
	'awsl', '来了来了', '才八点', '晚安', '我好了', \
	'66666', 'neeeee', 'mooooo', '理解理解', \
	'噔噔咚', '谢谢茄子', '犬山哥', '小狐狸', 'fbk', '白上吹雪', '天才天才', '久等了', '光速下播', '上手上手', '危', \
	'mea', 'aqua', 'meaqua', '要来了', '余裕余裕', '粘贴', '哔哩哔哩 (゜-゜)つロ 干杯~', '牙白', '谁？', '233333', \
	'rua', '全部木大', '吵死了', 'kimo', '白给', '清楚清楚', '斯哈斯哈', '不愧是你', '酸', '完全一致', '夏哥', '白等了', \
	'震撼傻夸', '呕', 'debu', 'ksnb', '丈育', 'dd', '张京华', '点名夸奖', '擦盘子', '咩！', '传统艺能', '川剧变脸', \
	'心豚', '心心', '委员长', '杂鱼体力', '擦玻璃', 'yyut', 'homo', '锤子', '夏色祭', 'hololive', '蝙蝠妹', '爱酱', '阿律', \
	'猫宫', '輝夜月', '帕里', '黑白角龙', '阿夸', '彩虹社', '爱小姐', '774', '清楚系', '天狗', 'ai', '草莓牛奶', '回生', \
	'我很可爱，请给我钱', '我爱你', '家主', '织田信姬', '山田赫敏', 'upd8']

def tsne_plot(labels, new_values):
	"Creates and TSNE model and plots it"
	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])

	plt.figure(figsize=(14, 8)) 
	forbid_overlap = []
	for i in range(len(x)):
		plt.scatter(x[i],y[i])
		forbid_overlap.append(plt.text(x[i], y[i], labels[i]))
		# forbid_overlap.append(plt.annotate(labels[i],
		# 		xy=(x[i], y[i]),
		# 		xytext=(5, 2),
		# 		textcoords='offset points',
		# 		ha='right',
		# 		va='bottom'))
	plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
	plt.xlabel('Danmaku Embedding t-SNE plot')
	adjust_text(forbid_overlap)
	plt.show()

if __name__ == '__main__':
	# set_matplot_zh_font()
	bc = BertClient()
	# Generate all embeddings
	Embedding_list = []
	message_list = return_message_list()
	for message in message_list:
		cur_encoding = bc.encode([message])
		Embedding_list.append(cur_encoding[0])
	Embedding_array = np.asarray(Embedding_list)
	dataframe = pd.DataFrame.from_records(Embedding_array)
	# First, deploy PCA:
	pca_50 = PCA(n_components=50)
	vals = dataframe.ix[:, :767].values
	pca_result_50 = pca_50.fit_transform(vals)
	print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
	time_start = time.time()
	tsne = TSNE(n_components=2, verbose=0, learning_rate = 100, perplexity=45, n_iter=2000, random_state=0)
	tsne_pca_results = tsne.fit_transform(pca_result_50)
	print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
	tsne_plot(message_list, tsne_pca_results)










