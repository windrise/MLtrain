## SVD的应用：(Singular value decompose)

* 图像压缩（image compression）： 较小奇异值就可以表达图像的大部分信息，舍弃掉一部分奇异值实现压缩
* 图像的降噪（image denoise）：噪声一般存在于图像的高频部分，也表现在奇异值小的部分，、
* 音频滤波（filter）： Andrew Ng 的机器学习上有一个svd将混杂声音分离的例子
* 求任意矩阵的伪逆（pseudo-inverse）：由于奇异矩阵或非方阵矩阵不可求逆，需要广义求逆的时候可以用svd
* 模式识别中，可以利用svd从较大数据量的特征中提取主要特征，保留特征中的90%的能量信息
* 推荐系统中数据的降维
* 潜在语义索引（latent semantic indexing）：NLP中，文本分类的关键是计算相关性，这里的关联矩阵 A=USV'，分解的三个矩阵有实际的物理意义，可以同时得到每类文章和每类关键词的相关性


## svd compression效果
![svdcompression](https://github.com/windrise/MLtrain/blob/master/SVD/svd.png)
