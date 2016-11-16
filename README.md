# Andrew Ng's Machine Learning Course
Tags : Learning

## Lesson 1
###Definition of machine learning
A computer program is said to learn from experience E with respect to some task T and from performance measure P, if its performance on T, as measured by P, improves with experience E.

---
###parts
- 监督学习 （Supervised Learning）
    - 回归问题 （Regression Problem）
    - 分类问题 （Classification Problem）
- 学习理论（Learning Theory）
- 非监督学习（Unsupervised Learning）
    - 聚类问题（Cluster Problem）
- 强化学习*（Reinforcement Learning）

---
## Lesson 2  梯度下降与线性回归
### Definition

mark|meaning|含义
:----:|:----:|:----:
$x$|features|特征
$y$|target|目标
$m$|number of training examples|训练集个数
$n$|number of features|特征数
$h$|hypothesis function|假定函数
$(x^i,y^i)$|No.i training example|第i个训练集
$\theta$|parameters|参数
$\alpha$|step|步长


---
### 梯度下降（Gradient Descent）

　　在监督学习方法中，我们采用最小二乘法对假定函数进行优化，即我们需要最小化评估函数 $J(\theta)$：
$$ J(\theta) = \sum_{i=1}^n (h_\theta(x_i) - y_i)^2 $$

　　其中，$h_\theta(x)$ 表示假定函数， $\theta$表示假定函数中的参数集。

---
　　梯度下降法为我们提供一种获得$J(\theta)$局部最小值的方法：
　　从任意一个参数集$\theta^0$开始，沿$J(\theta)$在$\theta^0$处的梯度的反方向，将$\theta$挪动步长$\alpha$，到达$\theta^1$，以此类推， 有：

$$ \theta^i = \theta^{i-1} - \alpha \nabla_\theta J(\theta) $$

　　直到一组$\theta^i$，满足$\nabla_\theta J(\theta) = 0$，即在$\theta^i$处，取得$J(\theta)$的局部最小值，$h_{\theta^i}(x)$即为最终的假定函数。
　　其中，
$$
\nabla_\theta J(\theta) = 
\left[
\begin{matrix}
\frac{\partial J}{\partial \theta_1}\\
\frac{\partial J}{\partial \theta_2}\\
\vdots\\
\frac{\partial J}{\partial \theta_n}\\
\end{matrix}
\right]
$$

---
　　以上介绍的是批梯度下降法（Batch Gradient Descent），批梯度下降法必须将全部训练样本集放入内存中进行计算，在实际使用中有诸多限制，而随机梯度下降法（Random Gradient Descent）可以每次根据一个训练样本对$\theta$进行优化：
　　我们的目标是最小化$J(\theta)$函数：

$$ J(\theta) = \sum_{i=1}^n (h_\theta(x_i) - y_i)^2 = (h_\theta(x_0) - y_0)^2 + (h_\theta(x_1) - y_1)^2 + ...+ (h_\theta(x_m) - y_m)^2$$

　　随机梯度下降法，是从任意一个参数集$\theta^0$开始，依次根据每个样本，沿$J_\lambda(\theta)$在$\theta^0$处的梯度的反方向，将$\theta$挪动距离$\alpha$，到达$\theta^1$，以此类推， 有：

$$ \theta^i = \theta^{i-1} - \alpha \nabla_\theta J_\lambda(\theta) $$

　　其中：

$$J_\lambda(\theta)=(h_\theta(x_\lambda) - y_\lambda)^2$$

　　随机梯度法在拥有大规模训练数据时，效率会远优于批梯度下降法。但随机梯度下降法很难精确收敛到局部最小值，会在非常接近最小值附近徘徊。

---
### 线性回归（Linear Regression）

　　在线性回归问题中，$h_\theta(x)$可以作如下表示：

$$h_\theta(x) = X_i\Theta , 其中 X_i \in R^{1*n}, \Theta \in R^{n*1}$$

　　$J(\theta)$可以表示如下：

$$J(\theta) = X\Theta  - Y, 其中 X \in R^{n*m}, \Theta 、Y \in R^{n*1}$$

　　其中，$X_i$表示一个训练集的特征构成的行向量，$\Theta$表示参数构成的列向量，$X$表示所有训练集特征构成的$m*n$维矩阵，$Y$表示所有训练集目标构成的列向量，运用线性代数定理，可推出：

$$\nabla_\theta J(\theta) = X^TX\Theta - X^TY \overset{set}{=} \overrightarrow0$$

　　最终获得法方程（Normal Equation）：

$$\Theta \overset{set}{=} (X^TX)^{-1}X^TY$$

　　法方程为线性回归问题提供了简便解法。

---
##Lesson 3
### 欠拟合（Underfitting）与过拟合（Overfitting）

### Locally weighted regression

- Parametric Learning Algorithmn
    - have a fix set of $\Theta$
- Non-parametric Learning Algorithmn
    - number of $\Theta$ grows linearly with $m$
    
