'''
SVD的应用：
Singular value decompose
1、图像压缩（image compression）： 较小奇异值就可以表达图像的大部分信息，舍弃掉一部分奇异值实现压缩
2、图像的降噪（image denoise）：噪声一般存在于图像的高频部分，也表现在奇异值小的部分，、
3、音频滤波（filter）： Andrew Ng 的机器学习上有一个svd将混杂声音分离的例子
4、求任意矩阵的伪逆（pseudo-inverse）：由于奇异矩阵或非方阵矩阵不可求逆，需要广义求逆的时候可以用svd
5、模式识别中，可以利用svd从较大数据量的特征中提取主要特征，保留特征中的90%的能量信息
6、推荐系统中数据的降维
7、潜在语义索引（latent semantic indexing）：NLP中，文本分类的关键是计算相关性，
这里的关联矩阵 A=USV'，分解的三个矩阵有实际的物理意义，可以同时得到每类文章和每类关键词的相关性

'''

import numpy as np 
import matplotlib.pyplot as plt 
from pylab import mpl 
from PIL import Image

mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def svd_decompose(img, s_num):
	#svd分解
	u,s,vt = np.linalg.svd(img)
	h,w = img.shape[:2]
	#用前s_num 个奇异值来产生新的对角矩阵
	s1 = np.diag(s[:s_num],0) 
	u1 = np.zeros((h,s_num),float)
	vt1 = np.zeros((s_num,w),float)

	u1[:,:] = u[:,:s_num]
	vt1[:,:] = vt[:s_num,:]

	svd_img = u1.dot(s1).dot(vt1)
	return svd_img

def main():
	#打开彩色图片并将其转换为灰度图片
	img = Image.open('house.jpg').convert('L')
	img = np.array(img)

	svd1= svd_decompose(img,1)
	svd5= svd_decompose(img,5)
	svd10= svd_decompose(img,10)
	svd20= svd_decompose(img,20)
	svd50= svd_decompose(img,50)
	svd100= svd_decompose(img,100)
	plt.figure(1);
	plt.subplot(331);plt.imshow(img,cmap='gray');plt.title('原图');plt.xticks([]);plt.yticks([]);
	plt.subplot(332);plt.imshow(svd1,cmap='gray');plt.title('1 Singular value');plt.xticks([]);plt.yticks([]);
	plt.subplot(333);plt.imshow(svd5,cmap='gray');plt.title('5 Singular value');plt.xticks([]);plt.yticks([]);
	plt.subplot(335);plt.imshow(svd10,cmap='gray');plt.title('10 Singular value');plt.xticks([]);plt.yticks([]);
	plt.subplot(336);plt.imshow(svd20,cmap='gray');plt.title('20 Singular value');plt.xticks([]);plt.yticks([]);
	plt.subplot(338);plt.imshow(svd50,cmap='gray');plt.title('50 Singular value');plt.xticks([]);plt.yticks([]);
	plt.subplot(339);plt.imshow(svd100,cmap='gray');plt.title('100 Singular value');plt.xticks([]);plt.yticks([]);
	plt.show()

if __name__ == '__main__':
	main()

