# Neural Network

###### tags: `mlb`

## 1. Logistic Regression

<!--
在談論 neural network 之前，我們必須先回憶一下 logistic regression，**以下所探討的 logistic regression 為二元分類器**
-->

### 1.1 Definition

<!--
假設有一個樣本 $X$ 其包含特徵 (features) 為 $x_1, x_2, ... , x_n$，可以將 logistic regression 用數學表示如下：
-->
Hypothesis of logistic regression :

$$
h(X) = \theta(b + \sum_{i=1}^nx_iw_i)
$$

<!--
其中 $w_1, w_2, ..., w_n$ 與 $b$ 是訓練的參數，而 $\theta(x)$ 被稱作 **sigmoid function** 定義如下：
-->

$\theta(x)$ is sigmoid function which is defined as the following :

$$
\theta(x) = \frac{e^x}{1 + e^x} = \frac{1}{e^{-x} + 1}
$$

<!--
從 sigmoid function 的定義可以知道它的輸出**必定介於 0 與 1 之間**，因此我們可以將其輸出的值直接對應到機率，下圖為其函數曲線：
-->

We can know that the output range of sigmoid function is between 0 and 1.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

:::warning
Q : According to the definition of logistic regression, how many trainable variables in a logistic regression model ?
<!--由上述 logistic regression 的定義，logistic regression 總共有幾個可訓練的參數 ? 假設每筆資料有 $n$ 個 feature-->
:::

### 1.2 Loss Function and Optimization

<!--
接著 logistic regression 的 loss function 為 cross entropy，其定義如下：
-->

**Cross entropy** is used as loss function here and is defined as the following :

$$
loss = \sum\limits_{i=1}^n ln(1 + e ^ {-yh(X_i)})
$$

<!--
其中 $X_i$ 代表第 $i$ 個樣本，而我們要做的事情就是最小化 loss ，在 logistic regression 中便是利用 gradient descent 來進行 loss 最小化的動作
-->

$X_i$ means the $i_{th}$ sample. And all we need is to minimize the output of loss function. We will use gradient descent algorithm to do this task.

<!--
gradient descent 的概念是將 loss 對每個可訓練參數做偏微分得到每個參數目前的梯度，然後利用梯度來更新參數，簡易的步驟如下：
-->

The concept of gradient descent is to get gradients of each variables and update variables by their gradients.

The process will repeat the following two steps : 

1. compute $g_w = \frac{\partial loss}{\partial W}, g_b = \frac{\partial loss}{\partial b}$

2. update $W_{new} = W - \eta \cdot \frac{g_w}{||g_w||}$ , $b_{new} = b - \eta \cdot \frac{g_b}{||g_b||}$

<!--
其中 $\eta$ 又被稱作 learning rate 可以調整每次更新的幅度，可以參照以下示意圖：
-->

We call $\eta$ as **learning rate**. We can adjust its value to control the learning speed of the model.

- Too large : model cannot converge.
- Too small : it may converge to a local minimum if loss function is non-convex.

<!--
![](https://cdn-images-1.medium.com/max/1600/0*QwE8M4MupSdqA3M4.png)

![](http://mathonline.wdfiles.com/local--files/local-maxima-and-minima-and-absolute-maxima-and-minima/Screen%20Shot%202014-08-31%20at%202.33.00%20PM.png)
-->

## 2. Neural Network (神經網路)

### 2.1 Neuron (神經元)

<!--
神經元即一個神經細胞，透過複數個樹突將訊號傳進細胞本體，當細胞本體的膜電位超過一定的閥值時，將訊號透過軸突傳出去：
-->

Neuron is also called "nerve cell" (神經細胞). Its cell body will accept multiple signal from dendrites (樹突). And it will emit its signal through axon (軸突) when its membrane potential (膜電位) exceeds a threshold. 
![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Neuron.svg/1200px-Neuron.svg.png)

<!--
neural network 的雛型便是模擬神經元的行為，所以我們假設一個神經元每個樹突接收到的訊號分別為 $x_1, x_2, ... x_n$，接著每個樹突會有一個自己的權重 $w_1, w_2, ..., w_n$ 代表自己所接收到的訊號的重要程度，所以傳達到細胞本體的訊號就會是：
-->

The prototype of neural network had come from simulating the behaviour of neurons.

We can assume that the signal of each dendrites is $x_1, x_2, ... x_n$. And each dendrites has a weight to evaluate the importance of the input signal. So we can use the following equation to represent the signal of a neuron : 

$$
signal = \sum\limits_{i=1}^n w_ix_i = XW
$$

<!--
上式以矩陣 $X, W$ 來簡化式子，其中 $X$ 為 $1 \times n$ 的矩陣，$W$ 為 $n \times 1$ 的矩陣，接著，神經元的訊號必須超過一個閥值才會向下一個神經元傳遞訊號，所以假設如下：
-->

We can use matrix to simplfiy the equation where $X$ is a $1 \times n$ matrix and $W$ a is $n \times 1$ matrix.

The behaviour of membrane potential and threshold can be interpreted as follows.

$$
output =
\begin{equation}
\begin{cases}
    1, &\text{signal > threshold} \\
    0, &\text{signal <= threshold}
\end{cases}
\end{equation}
$$

<!--
做一點小手腳，將 $threshold$ 移至比較符號的左邊：
-->

Shift $threshold$ to left of compare operator : 

$$
output = 
\begin{equation}
\begin{cases}
    1, &\text{signal - threshold > 0} \\
    0, &\text{signal - threshold <= 0}
\end{cases}
\end{equation}
$$

<!--
將 $XW$ 代入 $signal$，並且設 $b = -threshold$：
-->

Replace $signal$ with $XW$ and let $b = -threshold$ : 

$$
output =
\begin{equation}
\begin{cases}
    1, &\text{XW + b > 0} \\
    0, &\text{XW + b <= 0}
\end{cases}
\end{equation}
$$

<!--
最後我們需要有一個函式可以表示輸入超過 0 時輸出為 1 且輸入不超過 0 時輸出為 0，即：
-->

Finally, we need a threshold function whose output is 1 when the input is larger than 0, and output is 0 else where: 

$$
f(x) = 
\begin{equation}
\begin{cases}
    1, &\text{x > 0} \\
    0, &\text{x <= 0}
\end{cases}
\end{equation}
$$

<!--
首先符合這個描述的是 unit step function：
-->

Unit step function matchs our requirement : 

![](https://www.intmath.com/laplace-transformation/svg/svgphp-unit-step-functions-definition-1a-s0.svg)

<!--
但是由 unit step function 所建立出來的 loss function 為 0/1 error (非 0 即 1)，很難最佳化，在演算法中被歸類為 NP-Hard
-->

But there are some issues towards unit step funcion : 

- The output of unit step function is either 0 or 1. It is a NP-hard problem to optimize the loss function constructed by unit step function.

<!--
同時考慮到現實上神經訊號的變化並非是離散的，所以我們必須將 unit step function 換成分布很接近但是連續可微分的函式：
-->

- In reality, the change of signal is not discrete.

So, we should find other continuously differentiable (連續可微分) functions with similar output to the unit step function : 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/320px-Logistic-curve.svg.png)

<!--
也就是 sigmoid function，將 unit step function 換成 sigmoid function 後我們可以得到如下的數學模型來代表一顆神經元：
-->

Sigmoid function is a continuously differentiable function and has similar output to the unit step function. We can modify the equation as follows : 

$$
output = \theta(XW +b) = \theta(b + \sum_{i=1}^n x_iw_i)\\
where~~\theta(x) = \frac{1}{e^{-x} + 1}
$$

We call the sigmoid function as **activation function** here. Because it determines wheather to **activate** the neuron or not.

![](http://1.bp.blogspot.com/-hACN9Tu7nto/WPS0yKLylxI/AAAAAAAAAnM/5Q84dLVnizAvbQdM_j4HkGofCBskCuJpgCK4B/s1600/%25E7%25A5%259E%25E7%25B6%2593%25E5%2585%2583.PNG)

<!--
沒錯，logistic regression 可以代表一顆神經元，而 sigmoid function 在這裡被稱為 **激活函數 (activation function)**
-->

Here, we've already demonstrated how to simulate a single neuron using logistic regression and activation function.

### 2.2 Multiple Neurons

<!--
有了一顆神經元的數學表示式，我們可以將嘗試建立多顆神經元的數學表示
，考慮兩顆神經元的情況下：
-->

Now we consider two neurons accepting the same input signal : 

![](https://i.imgur.com/meH7O92.png)


$$
output_1 = \theta(XW_1 + b_1) \\
output_2 = \theta(XW_2 + b_2)
$$

<!--
改以矩陣的形式表示：
-->

Use matrix format : 

$$
output = 
\begin{bmatrix}
    output_1, output_2
\end{bmatrix} = 
\begin{bmatrix}
    \theta(XW_1 + b_1), \theta(XW_2 + b_2)
\end{bmatrix}
= \theta(
\begin{bmatrix}
    XW_1 + b_1, XW_2 + b_2
\end{bmatrix}
)
$$

<!--
接著將矩陣拆開：
-->

Then, unfold the matrix : 

$$
\theta(
\begin{bmatrix}
    XW_1 + b_1, XW_2 + b_2
\end{bmatrix}
) =
\theta(
\begin{bmatrix}
    XW_1, XW_2
\end{bmatrix} +
\begin{bmatrix}
    b_1, b_2
\end{bmatrix}
) =
\theta(
X \times
\begin{bmatrix}
    W_1, W_2
\end{bmatrix}
+
\begin{bmatrix}
    b_1, b_2
\end{bmatrix}
)
$$

<!--
這邊設 $W_{total} = [W_1, W_2], b_{total} = [b_1, b_2]$，式子改寫如下：
-->

Substitute the original equation:
$W_{total} = [W_1, W_2], b_{total} = [b_1, b_2]$ : 

$$
output = \theta(XW_{total} + b_{total}) \\
where~~W_{total} = [w_{ij}]_{n \times 2},~b_{total} = [b_{ij}]_{1 \times 2}
$$

<!--
有了以上推導後，我們便可以用以下數學式表示 $m$ 個神經元：
-->

After we derived the equation for two neurons, we can move foward to the equation for $m$ neurons : 

$$
output = \theta(XW + b) \\
where~~W = [w_{ij}]_{n \times m},~b = [b_{ij}]_{1 \times m},~output = [h_{ij}]_{1 \times m}
$$

<!--
在 neural network 中，我們將這樣的一組神經元稱作一個 **layer**，也就是一層，$W$ 稱為 **weight**，$B$ 稱為 **bias**
-->

In neural network, we call the collection of $m$ neurons at the same phase as a **layer**, $W$ as **weight**, and $b$ as **bias**.

### 2.3 Basic Architecture

<!--
神經網路基本架構示意圖如下：
-->


![](https://pic.pimg.tw/darren1231/1483983368-1814844174_n.png?v=1483983372)

<!--
1. 第一層為輸入層 (input layer)，也就是我們要輸入的特徵，以 $[x_1, x_2, ..., x_n]$ 表示
2. 第二層為隱藏層 (hidden layer)，隱藏層會將 $[x_1, x_2, ..., x_n]$ 做 2.2 小節所推導出的運算，最後輸出表示為 $[h_1, h_2, ..., h_k]$，$k$ 表示隱藏層的神經元數量
3. 最後示輸出層 (output layer)，將 $[h_1, h_2, ..., h_n]$ 在做一次 2.2 小節的計算，最後輸出 $[y_1, y_2, ..., y_m]$，$m$ 表示輸出層神經元數量
-->

1. Input layer : features we want to feed in the model. It is represented as $[x_1, x_2, ..., x_n]$.
2. Hidden layer : a **layer** derived from section 2.2. It is represented as $[h_1, h_2, ..., h_k]$.
3. Output layer : prediction of the model. And it is also a layer derived from section 2.2. It is represented as $[y_1, y_2, ..., y_m]$.

:::info
<!--
注意：如果我們的模型跟一開始的 logistic regression 一樣是二元分類問題，那輸出層的神經元數量設為 1 即可，即是將隱藏層的輸出做一次普通的 logistic regression
-->
Note : 
If you want to make a binary classification, you can set total number of neurons of output layer to 1. This is equal to do a logistoc regression with output of hidden layer. 
:::


### 2.4 Loss Function

<!--
在推導 loss function 之前，我們需要先介紹一個數學概念：相對熵 (relative entropy, 又稱 Kullback–Leibler divergence)

相對熵用來計算兩個不同的機率分布之間的差異性，其定義如下：
-->

Kullback–Leibler divergence (also called "relative entropy") : a measure that tells the similarity between two distributions.

$$
D_{KL}(P||Q) = \sum_{i} P(i) ln \frac{P(i)}{Q(i)}
$$

<!--
其中 $P,~Q$ 代表兩種不同的機率分布

有了上述概念後，我們可以將 $P$ 想成樣本標籤的分布，$Q$ 則是分類器預測出來的分布，接著將 $ln$ 裡面拆開：
-->

$P,~Q$ are two different distributions. We can think of $P$ as labels of samples, and $Q$ as prediction of the samples.

According to the laws of logarithms (對數律), Kullback–Leibler divergence can be written as : 

$$
D_{KL}(P||Q) = \sum_{i} P(i) ln P(i) - \sum_{i} P(i) ln Q(i) 
$$

<!--
而 $\sum P(i) ln P(i)$ 是已知的常數，所以必須要最佳化的部分就是：
-->

$\sum P(i) ln P(i)$ is a constant that we already know. So, we only need to optimize the right part of the equation : 

$$
-\sum_{i} P(i) ln Q(i)
$$

<!--
考慮到在二元分類問題，標籤只有兩個可能：$0,~1$，而分類器輸出的數值代表標籤為 $1$ 的機率，可以將式子改寫如下：
-->
In binary classification, the labels are either 0 or 1. And if we consider the prediction result to be the probability of class 1, the equation can be modified as follows. 

$$
-P(i) ln Q(i) = -[y_{true}~ln~y_{pred} + (1 - y_{true})~ln~(1 - y_{pred})]
$$

<!--
以 $y_{true}$ 代表標籤，$y_{pred}$ 代表分類器輸出的值，式子前半部為 $y_{true} = 1$ 的狀況，後半部為 $y_{true} = 0$ 的狀況，而這就是我們在二元分類問題上會使用的 loss function，稱為 **cross entropy**
-->

$y_{true}$ means label, and $y_{pred}$ means predicted probability of class 1.

We call this equation as **cross entropy**.

:::warning
Q：What is the different between cross entropy here and cross entropy of logistic regression ?
<!--
這邊的 cross entropy 跟 logistic regression 的 cross entropy 有什麼關係呢 ?
-->
:::

### 2.5 Backpropagation

#### 2.5.1 Forward pass

Consider a nerual network as following : 

```
                   Wh, bh           Wo, bo
|---------|  x  |----------|  h  |----------|
|  input  | --> |  hidden  | --> |  output  | --> o --> loss
|---------|     |----------|     |----------|
```

We feed a sample as $x$, get $h$ from hidden layer, get $o$ from output layer, and compute its loss.

This process is called as **forward pass**.

#### 2.5.2 Backward pass

<!--
取得 loss 後，就可以計算每個可訓練參數的梯度，首先只考慮離 loss 較近的 $W_o$，其梯度可以如下表示：
-->

After getting the loss, we can compute the gradients of each trainable variables.

First, we look at $W_o$ : 

$$
Gradient(W_o) = \frac{\partial loss}{\partial W_o}
$$

<!--
根據微積分的 [chain rule](https://en.wikipedia.org/wiki/Chain_rule) 可以把式子分解如下：
-->

According to [chain rule](https://en.wikipedia.org/wiki/Chain_rule), we can get the following equation : 

$$
Gradient(W_o) = \frac{\partial o}{\partial W_o} \times \frac{\partial loss}{\partial o} 
$$

<!--
因為我們已經有 loss function 以及 $o$ 的值，可以直接算出 $\frac{\partial loss}{\partial o}$：
-->

We can compute $\frac{\partial loss}{\partial o}$ by $loss$ and $o$ : 

$$
loss = f(o) = - y_{true}\ ln\ o - (1 - y_{true})\ ln\ (1 - o) \\
\frac{\partial loss}{\partial o} = f'(o) = - \frac{y_{true}}{o} ＋ \frac{1 - y_{true}}{1 - o}
$$

<!--
所以我們可以專注在如何計算 $\frac{\partial o}{\partial W_o}$，同樣再用 chain rule 拆解：
-->

To compute $\frac{\partial o}{\partial W_o}$, We can use chain rule again : 

$$
z = hW_o + b_o \\
o = Sigmoid(z) \\
\frac{\partial o}{\partial W_o} = \frac{\partial z}{\partial W_o} \times \frac{\partial o}{\partial z} = \frac{\partial z}{\partial W_o} \times \frac{\partial Sigmoid(z)}{\partial z}
$$

<!--
其中 $\frac{\partial Sigmoid(z)}{\partial z}$ 可以直接套用 sigmoid 的微分公式取得：
-->

Get $\frac{\partial Sigmoid(z)}{\partial z}$ by formula of derivative of sigmoid : 

![](https://i2.wp.com/kawahara.ca/wp-content/uploads/derivative_sigmoid.png?fit=430%2C339)

<!--
而 $\frac{\partial z}{\partial W_o} = \frac{\partial (hW_o + b_o)}{\partial W_o} = h$ 也是已知的值，到此 $\frac{\partial loss}{\partial W_o}$ 已經算出來了：
-->

and $\frac{\partial z}{\partial W_o} = \frac{\partial (hW_o + b_o)}{\partial W_o} = h$

So, we can get the following equation : 


$$
\frac{\partial loss}{\partial W_o} = h \times Sigmoid'(hW_o + b_o) \times (- \frac{y_{true}}{o} ＋ \frac{1 - y_{true}}{1 - o})
$$

<!--
其中 $h$, $o$ 都是已知的數值
-->

<!--
接著可以嘗試計算 $\frac{\partial loss}{\partial W_h}$：
-->

Then, try to compute $\frac{\partial loss}{\partial W_h}$ : 

$$
\frac{\partial loss}{\partial W_h} = \frac{\partial h}{\partial W_h} \times \frac{\partial o}{\partial h} \times \frac{\partial loss}{\partial o}
$$

<!--
$\frac{\partial loss}{\partial o}$ 上面已經算出來了，所以直接來看 $\frac{\partial o}{\partial h}$：
-->

$\frac{\partial loss}{\partial o}$ is known. We should focus on $\frac{\partial o}{\partial h}$

$$
\frac{\partial o}{\partial h} = \frac{\partial (hW_o + b_o)}{\partial h} \times \frac{\partial o}{\partial (hW_o + b_o)} \\
= W_o \times Sigmoid'(hW_o + b_o)
$$

<!--
$\frac{\partial h}{\partial W_h}$ 可以直接參考上面 $\frac{\partial o}{\partial W_o}$，概念一模一樣：
-->

The derivation of $\frac{\partial h}{\partial W_h}$ is the same as the derivation of $\frac{\partial o}{\partial W_o}$ : 

$$
\frac{\partial h}{\partial W_h} = x \times Sigmoid'(xW_h + b_h)
$$

<!--
所以 $\frac{\partial loss}{\partial W_h}$ 式子如下：
-->

We get $\frac{\partial loss}{\partial W_h}$ as following : 

$$
\frac{\partial loss}{\partial W_h} = x \times Sigmoid'(xW_h + b_h) \times W_o \times Sigmoid'(hW_o + b_o) \times (- \frac{y_{true}}{o} ＋ \frac{1 - y_{true}}{1 - o})
$$

#### 2.5.3 Backward pass for multiple hidden layer

<!--
考慮一個 neural network 架構如下：
-->

Consider a neural network as following : 

```
       W1, b1        W2, b2                  Wn-1, bn-1          Wn, bn        Wo, bo
input ---------> h1 ---------> h2 ---> ... --------------> hn-1 ---------> hn ---------> output
```

<!--
Backward pass 可以從 output layer 往前回推：
-->

Backward pass is worked from output layer to input layer : 

$$
\frac{\partial\ loss}{\partial W_o} = \frac{\partial\ output}{\partial W_o} \times \frac{\partial\ loss}{\partial\ output}
$$

$$
\frac{\partial\ loss}{\partial W_n} = \frac{\partial h_n}{\partial W_n} \times \frac{\partial\ output}{\partial h_n} \times \frac{\partial\ loss}{\partial\ output}
$$

$$
\frac{\partial\ loss}{\partial W_{n-1}} = \frac{\partial h_{n-1}}{\partial W_{n-1}} \times 
\frac{\partial h_n}{\partial h_{n-1}} \times \frac{\partial\ output}{\partial h_n} \times \frac{\partial\ loss}{\partial\ output}
$$

$$
\frac{\partial\ loss}{\partial W_{n-2}} = \frac{\partial h_{n-2}}{\partial W_{n-2}} \times 
\frac{\partial h_{n-1}}{\partial h_{n-2}} \times \frac{\partial h_{n}}{\partial h_{n-1}} \times \frac{\partial\ output}{\partial h_n} \times \frac{\partial\ loss}{\partial\ output}
$$

$$
.\\.\\.
$$

<!--
可以觀察到，如果 backward pass 從最後一層往前算每個可訓練參數的梯度的話，能夠重複使用一些已經被計算出來的數值，因此能夠大幅減少計算的時間
-->

We can reuse the intermediate products of the process of previous layer to reduce the cost of computation.


:::warning
Q：The equation of $\frac{\partial loss}{\partial b_o}$ is very similar to $\frac{\partial loss}{\partial W_o}$.

The equation of $\frac{\partial loss}{\partial W_o}$ : 


<!--
$\frac{\partial loss}{\partial b_o}$ 推導出來的公式其實跟 $\frac{\partial loss}{\partial W_o}$ 很像，$\frac{\partial loss}{\partial W_o}$ 公式如下：
-->

$$
\frac{\partial loss}{\partial W_o} = h \times Sigmoid'(hW_o + b_o) \times (- \frac{y_{true}}{o} + \frac{1 - y_{true}}{1 - o})
$$
<!--
而 $\frac{\partial loss}{\partial b_o}$ 公式如下：
-->

And the following is the equation of $\frac{\partial loss}{\partial b_o}$ : 

$$
\frac{\partial loss}{\partial b_o} = ??? \times Sigmoid'(hW_o + b_o) \times (- \frac{y_{true}}{o} + \frac{1 - y_{true}}{1 - o})
$$
<!--
請問上式中 $???$ 該填入什麼 ?
-->

What is the correct value of "$???$" ?
:::

### 2.6 Gradient Descent in Neural Network

#### 2.6.1 Gradient descent in logistic regression

<!--
上一小節推導了 $W$ 跟 $b$ 的梯度，接著可以利用這些梯度去更新 neural network 所有的 $W$ 跟 $b$，但是在這之前，有一個小細節要注意

首先先回想 logistic regression 的 loss function：
-->

Look at the loss function of logistic regression again : 

$$
loss = \sum\limits_{i=1}^n ln(1 + e ^ {-yh(X_i)})
$$

<!--
上式要注意的地方是 $n$ 這個變數的意義，在這裡 $n$ 是指輸入模型的樣本數量，而 logistic regression 在訓練過程中每一次迭代都是將整個 dataset 輸入進模型，所以 $n$ 即代表整個 dataset 的樣本數量
-->

where $n$ is equal to total number of samples in a dataset.

In other words, we will feed the whole dataset into model at every iteration during training. But this way won't work on neural network for some reasons : 

- Too much time to compute gradients of each variables with each samples
- Too much memory during forward pass and backward pass

<!--
也就是說 **logistic regression 的 gradient descent 每次都是用整個 dataset 去完成的**，但是同樣的作法無法在 neural network 實現，其原因在於計算開銷極大，neural network 訓練參數多，計算每筆資料對應每個訓練參數的梯度會很花時間
-->



#### 2.6.2 Stochastic gradient descent (SGD)

<!--
為了解決上述問題，有人提出了 Stochastic Gradient Descent ( SGD, 隨機梯度下降法)，其概念就是每次迭代只從 dataset **隨機選取一筆資料**來計算 loss 以及更新訓練參數
-->

Definition : randomly select one sample from dataset for each iteration

<!--
所以 SGD 在每一次迭代的步驟如下：
-->

Pseudo code of SGD in a iteration: 

<!--
1. 隨機從訓練資料選取一個樣本 $x$

2. 計算 $x$ 的 $loss$

3. 計算 $g_w = \frac{\partial loss}{\partial W}, g_b = \frac{\partial loss}{\partial b}$

4. $W_{new} = W - \eta \cdot \frac{g_w}{||g_w||}$ 且 $b_{new} = b - \eta \cdot \frac{g_b}{||g_b||}$
-->

1. Pick a sample $x$ from training dataset randomly
2. Compute $loss$ of $x$
3. Compute gradients $g_w = \frac{\partial loss}{\partial W}, g_b = \frac{\partial loss}{\partial b}$ of each variable
4. Update all variables $W_{new} = W - \eta \cdot \frac{g_w}{||g_w||}$, $b_{new} = b - \eta \cdot \frac{g_b}{||g_b||}$

<!--
這個方法可以直觀上的解釋，假設所有訓練資料都屬於一個特定的分布 $P(x)$，那從訓練資料中隨機選取的一筆資料所計算出來的梯度方向就會與所有訓練資料所計算出來的梯度方向非常相似
-->

Why it works ?

- Each sample follow similar distribute of training dataset.

#### 2.6.3 Mini-batch gradient descent

<!--
SGD 在 neural network 的表現上確實比 gradient descent 好，但是它也存在一些缺點，其原因與 dataset 的分布有關

上面我們假設理想狀況下訓練資料來自一個特定分布 $P(x)$，但是實際狀況訓練資料的分布會是 $P(x) + e$，這裡 $e$ 代表 noise，即是說 dataset 中的資料分布會與理想狀況有一些差距，甚至會有 outlier (異常資料)

因此，在實際使用 SGD 每次更新的梯度之間的 variance 會因為 $e$ 的影響而變大，導致可能發生收斂不穩定

為了避免這種狀況發生，我們可以適量的增加每次要計算 loss 的樣本數，藉由平均複數樣本產生的梯度來減少每次更新時的 variance
-->

Disadvantage of SGD : high variance between the gradients of each iteration because **noise** exists.
-  Unstable convergence

Solution based on gradient descent : 
- mini-batch gradient descent : pick more samples randomly to reduce variance of each iteration.

<!--
所以 mini-batch gradient descent 每次迭代步驟如下：
-->

Pseudo code of mini-batch gradient descent in a iteration : 

<!--
1. 隨機從訓練資料選取 n 個樣本 $x_1, x_2, ..., x_n$

2. 計算 $x_1, x_2, ..., x_n$ 的平均 $loss$

3. 計算 $g_w = \frac{\partial loss}{\partial W}, g_b = \frac{\partial loss}{\partial b}$

4. $W_{new} = W - \eta \cdot \frac{g_w}{||g_w||}$ 且 $b_{new} = b - \eta \cdot \frac{g_b}{||g_b||}$
-->

1. Pick $n$ samples $x_1, x_2, ..., x_n$ randomly
2. Compute average loss of $x_1, x_2, ..., x_n$
3. Compute gradients $g_w = \frac{\partial loss}{\partial W}, g_b = \frac{\partial loss}{\partial b}$ of each variable
4. Update all variables $W_{new} = W - \eta \cdot \frac{g_w}{||g_w||}$ , $b_{new} = b - \eta \cdot \frac{g_b}{||g_b||}$

<!--
下圖為 gradient descent, mini-batch gradient descent, stochaistic gradient descent 的更新狀況：
-->

The following figure shows the learning status between gradient descent, stochastic gradient descent, and mini-batch gradient descent : 

![](https://cdn-images-1.medium.com/max/1600/1*PV-fcUsNlD9EgTIc61h-Ig.png)



### 2.7 Multi-class Problem

<!--
到上一個小節為止，已經推導完一個 neural network 的架構以及訓練流程，但是有些地方的推導 (ex: backpropagation) 是基於二元分類來推導的，所以接著會針對多元分類問題來微調 neural network 架構以及 loss function
-->

We only consider binary classification in the previous sections. Now, we will discuss multi-class classification.

#### 2.7.1 One-hot encoding

Use a vector to represent label of a sample : 
- Single number 0 or 1 is not enough.

One-hot vector : $n$ bits vector for n classes, each bit specify a class.

For example, There is a dataset contain the following labels : 

- red
- yellow
- green

One-hot vector of each label : 

- $[1, 0, 0]$ = red
- $[0, 1, 0]$ = yellow
- $[0, 0, 1]$ = green

Length of one-hot vector depends on total number of class in the dataset.

#### 2.7.2 Softmax function

The number of neurons of output layer is the same as length of one-hot vector.

- The model will produce a vector that corresponds to the probability of each class.

![](https://www.researchgate.net/profile/Andreas_Holzinger/publication/320687279/figure/fig7/AS:561595223691264@1510906143097/Deep-feed-forward-neural-network-with-two-hidden-layers-blue-balls-In-addition-the.png)

For example, the output vector $[0.1, 0.2, 0.7]$ can be explained that it has 10% probability to be class 0, 20% probability to be class 1, 70% probability to be class 2.


Sigmoid function is not enough : 
- Sum of each probability in a output vector is not always 1.
    - How do you determind the class if output vector is $[0.9, 0.9, 0.9]$ ?

Softmax function : normalize each element in the output vector

$$
\sigma(x)_j = \frac{e^{x_j}}{\sum_{i=1}^n e^{x_i}}
$$

- Normalized elements are between 0 and 1.
- Sum of all normalized elements is 1.

Then, we can **use softmax function as activation function of output layer**.

:::info
Sigmoid function is a specific case of softmax function.

$$
Sigmoid(x) = \frac{e^x}{e^x + 1} = \frac{e^x}{e^x + e^0}
$$

The above condition is the same as the following.

$$
output = [z, 0] \\
Softmax(z) = \frac{e^{z}}{\sum_{i=1}^n e^{x_i}} = \frac{e^z}{e^z + e^0}
$$
:::

#### 2.7.3 Multi-class cross entropy

Come back the part need to be optimized of Kullback–Leibler divergence :

$$
-\sum_{i} P(i) ln Q(i)
$$

In multi-class problem, $P(i)$ is a one-hot vector. And $Q(i)$ is a output vector of a neural network.

So, it can be modified as follows : 
$$
-P(i)lnQ(i) = -\sum_{k=0}^n label_k~ln (output_k)
$$

$label_k$ means $k_{th}$ element in the one-hot vector. $output_k$ means $k_{th}$ element in the output vector.

Take a break : [Tinker With a Neural Network Right Here in Your Browser.](https://playground.tensorflow.org)

## 3. Hyperparameters

There are many places to adjust your neural network : 
1. The way to initialize all variables
2. The number of layers
3. The number of neurons in a layer
4. Activation function of a layer
5. Batch size of gradient descent algorithm
6. Optimization algorithm

We will focus on point 4 and 6.

### 3.1 Activation Function

Exclude sigmoid and softmax, there still have many kinds of activation function : 

1. Tanh

$$
Tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

![](https://www.medcalc.org/manual/_help/functions/tanh.png)

2. Relu

$$
Relu(x) = 
\begin{equation}
\begin{cases}
    x, &\text{x > 0} \\
    0, &\text{x <= 0}
\end{cases}
\end{equation}
$$

![](https://cdn.tinymind.com/static/img/learn/relu.png)

4. Leaky Relu
- $\alpha$ is a constant.

$$
LeakyRelu(x) = 
\begin{equation}
\begin{cases}
    x, &\text{x > 0} \\
    ax, &\text{x <= 0, where 0 <}~\alpha~\text{< 1}
\end{cases}
\end{equation}
$$

![](https://i.stack.imgur.com/LZk6i.png)

5. PRelu

- $\alpha$ is a trainable variable.

$$
PRelu(x) = 
\begin{equation}
\begin{cases}
    x, &\text{x > 0} \\
    ax, &\text{x <= 0, where 0 <}~\alpha~\text{< 1}
\end{cases}
\end{equation}
$$

6. Selu

$$
Selu(x) = \lambda 
\begin{equation}
\begin{cases}
    x, &\text{x > 0} \\
    ae^x-a, &\text{x <= 0}
\end{cases}
\end{equation} \\~\\~\\
\alpha = 1.6732632423543772848170429916717 \\
\lambda = 1.0507009873554804934193349852946
$$

![](https://images2017.cnblogs.com/blog/702365/201801/702365-20180101173712034-346353968.png)


### 3.2 Optimization Algorithm

Mini-batch gradient descent algorithm without any **trick** has some problems : 

1. The training time takes too long.
2. It'll get traped in  local minimum easily.
3. Convergence will be unstable if error surface is "bumpy".

#### 3.2.1 Momentum

Simulate momentum of physics : 

$$
m_t = g_t + \mu \times m_{t-1} \\
\Delta\theta = - \eta \times m_t \\
W_{new} = W + \Delta\theta
$$

- $g_t$ : gradient of the variable at $t_{th}$ iteration
- $m_t$ : momentum at $t_{th}$ iteration
- $\mu$ : a constant
- $\eta$ : learning rate

Advantage : 
1. Speed up when error surface is smooth.
2. Leave the point of local minimum easily.

Extend : 
- Nesterov : A algorithm to improve momentum optimizer

#### 3.2.2 Adagrad

Constraint the gradient : 

$$
n_t = n_{t-1} + g_t ^ 2 \\
\Delta\theta = -\frac{\eta}{\sqrt{n_t + \epsilon}} \times g_t \\
W_{new} = W + \Delta\theta
$$

- $g_t$ : gradient of the variable at $t_{th}$ iteration
- $\epsilon$ : a constant to make sure denominator is not 0.
- $\eta$ : learning rate

Advantage : 
1. Speed up at the early stage.
2. Constraint the gradient at the late stage.
    - There is not necessery to use large gradient when loss is near convergence.

Extend : 
- Adadelta : a algorithm to improve adagrad optimizer

#### 3.2.3 RMSProp

Instead of store all previous squared gradients, RMSProp define a **decay rate** to control the ratio of previous gradients : 

$$
E[g ^ 2]_t = \rho E[g ^ 2]_{t-1} + (1 - \rho)g_t ^ 2\\
\Delta\theta = -\frac{\eta}{\sqrt{E[g ^ 2]_t + \epsilon}} \times g_t
$$

- $\rho$ : decay rate. Default setting is 0.9.

#### 3.2.4 Adam

A Combination of momentum and RMSProp algorithm. And it is **the most common** optimization algorithm in neural network.

First, compute momentum and constraints factor.

$$
m_t = \beta_1m_{t-1} + (1 - \beta_1)g_t \\
v_t = \beta_2v_{t-1} + (1 - \beta_2)g_t^2
$$

Second, fix momentum and constraints factor by $\beta_1$ and $\beta_2$.

$$
\hat m_t = \frac{1}{1 - \beta_1 ^ t} \times m_t \\
\hat v_t = \frac{1}{1 - \beta_2 ^ t} \times v_t
$$

Finally, update variable.

$$
\Delta\theta = - \eta \times \frac{\hat m_t}{\sqrt{\hat v_t + \epsilon}}\\
W_{new} = W + \Delta\theta
$$

- $\beta_1$, $\beta_2$ : decay rate

## 4. More

### 4.1 Dropout

Paper : [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

As the title of its original paper, dropout is a simple way to prevent neural networks from overfitting.

#### Idea

- Model combination nearly always improves the performance of machine learning methods.
    -  With  large  neural  networks, averaging  the  outputs  of many separately trained nets is prohibitively expensive.
- Sampling a “thinned” network by dropping out units.
    - A neural net with $n$ units, can be seen as a collection of $2^n$ possible thinned neural networks.

#### For training : 

- Temporarily removing a neuron from the network, along with all its incoming and outgoing connections : 

![](https://i.imgur.com/IPwWsKv.png)

- The choice of which units to drop is random.
- Each unit is retained with a fixed probability $p$ independent of other units.

#### For testing : 

- Use a single neural net at test time **without** dropout. 
- The weights of this network are scaled-down versions of the trained weights.
    - $W_{train} = pW_{test}$

![](https://i.imgur.com/PbvVHak.png)

### 4.2 Autoencoder

- A unsupervised learning algorithm used to **reduce the dimension**.
- Train a neural network that its output is the same as its input : 

![](https://www.doc.ic.ac.uk/~js4416/163/website/img/autoencoders/autoencoder.png)

- It encodes the input as hidden layer by encoder part and recontructs the input from hidden layer by decoder part. 
- Activation function of output layer depends on numerical range of input layer.
    - Linear or sigmoid.
- Loss function
    - Linear : $\sum (x - y_{pred}) ^ 2$
    - Sigmoid : $\sum [x~ln~y_{pred} - (1 - x) ln (1 - y_{pred})]$
- We will extract the value of hidden layer as a dimension-reduced input.

Autoencoder for MNIST ([more](https://gertjanvandenburg.com/blog/autoencoder/)): 

![](https://gertjanvandenburg.com/figures/autoencoder/scatter.svg)

Stacked Autoencoder : 
![](https://www.researchgate.net/profile/Konrad_Kording/publication/274728436/figure/fig2/AS:271715934666753@1441793535194/A-A-stacked-autoencoder-is-trained-on-high-dimensional-data-im-i-1-finding-a.png)

Extend : 
- [Layerwise Pre-training with Autoencoders](chrome-extension://oemmndcbldboiebfnladdacbdfmadadm/http://www.cbs.dtu.dk/courses/DeepLearning_workshop/VanessaJurtz.pdf)

## 5. Reference
- [Wikipedia : Chain Rule](https://en.wikipedia.org/wiki/Chain_rule)
- [Binary v.s. Multi-Class Logistic Regression](https://chrisyeh96.github.io/2018/06/11/logistic-regression.html)
- [An overview of gradient descent optmization algorithm](http://ruder.io/optimizing-gradient-descent/index.html)
- [Keras source code : keras/activations.py](https://github.com/keras-team/keras/blob/master/keras/activations.py)
- [Dropout : A Simple Way to Prevent Neural Networks from Overfitting](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
- [Introduce to Autoencoders](https://www.doc.ic.ac.uk/~js4416/163/website/autoencoders/)
- [Simple MNIST Autoencoder in TensorFlow](https://gertjanvandenburg.com/blog/autoencoder/)
- [Tinker With a Neural Network Right Here in Your Browser.](https://playground.tensorflow.org)
