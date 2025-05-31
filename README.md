# Fashion-Mnist Classification
- A machine learning project presented by 6 hfuters
- We use different models to classify the dataset
## Fashion-Mnist数据集
- Fashion-MNIST是Zalando的研究论文中提出的一个数据集，由包含60000个实例的训练集和包含10000个实例的测试集组成
- 每个实例包含一张28x28的灰度服饰图像和对应的类别标记
- 共有10类服饰，分别是：
  
  - T-shirt（T恤）
  - trouser（牛仔裤）
  - pullover（套衫）
  - dress（裙子）
  - coat（外套）
  - sandal（凉鞋）
  - shirt（衬衫）
  - sneaker（运动鞋）
  - bag（包）
  - ankle boot（短靴）
  
![数据集展示](https://i-blog.csdnimg.cn/blog_migrate/e1cc2fc3420f61881757e57ef523479a.png#pic_center)
### 数据集下载地址  
https://www.kaggle.com/datasets/zalando-research/fashionmnist/data  

- 测试集:`fashion-mnist_test.csv`
- 训练集:`fashion-mnist_train.csv`分别作为测试集和训练集
- 你也可以自行再划分验证集
- 下载好数据集后，导入流程见[导入数据集]（/数据导入）
## 分类模型
- 通过编写代码，我们希望能够实现类似于下图的分类功能
![预期功能](https://i-blog.csdnimg.cn/blog_migrate/1acd4c5e5ef73c7b46a7bb45eb4f9ac6.png#pic_center)

基于tensorflow和sscikit-learn等软件包，我们选取了以下6种机器学习中的经典模型:
- [多层感知机（MLP）](/MLP.ipynb)
- [深度神经网络（DNN）](/DNN.ipynb)
- [卷积神经网络（CNN）](/CNN.ipynb)
- [支持向量机（SVM）](/SVM.ipynb)
- [随机森林（Random-Forest）](/Random-Forest.ipynb)
- [XGBoost](/XGBoost.ipynb)  

读者可通过相应的`.ipynb`文件查看具体实现流程  

具体预测结果已放至[预测结果](/预测结果)中，读者可自行查看

[模型对比](/模型对比)中进行了6个模型的性能对比

# **Appreciate Reading!**
