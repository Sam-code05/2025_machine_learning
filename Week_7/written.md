# Score Matching

我們從 Score Function 的定義出發，探討經典 Score Matching 的計算挑戰，並說明 Denoising Score Matching (DSM) 如何將問題轉化為一個可解的「降噪」任務。最後，再來描述擴散模型如何利用訓練好的 Score 模型，從純噪聲中逐步反向生成高擬真度的數據。

---

## 一、Concept


### 1. Score Function(分數函數)

在統計學中，數據分佈 $p(x)$ 的 **Score Function** 被定義為：**對數機率密度函數對輸入數據 $x$ 的梯度 (gradient)**。

$$
\text{Score}(x) = \nabla_x \log p(x)
$$

* **$x$**：代表一個數據點（例如一張圖片的像素向量）。
* **$p(x)$**：數據的真實機率密度函數。這是一個我們無法確切知道的函數，但擁有來自這個分佈的樣本（即我們的數據集）。
* **$\log p(x)$**：取對數是為了數學上的便利性。
* **$\nabla_x$**：代表對 $x$ 的所有維度計算偏導數（即梯度）。

Score 函數是一個**向量場 (vector field)**。在數據所在的 $k$ 維空間中，**每一個點 $x$**，Score 函數都會賦予它一個**向量**。

這個向量指向**機率密度 $p(x)$ 增長最快**的方向。

> **直觀比喻：**
> 假設一個 2D 的機率密度「山丘」（高處代表高機率區域，低處代表低機率區域）。
> * 如果隨機站在山坡上某個點 $x$，Score function $\nabla_x \log p(x)$ 會得出一個向量，**指向山頂**。
> * 在機率密度低的地方（山腳），向量的幅度（即「分數」）很大，它會強烈地「推」你走向高機率區域。
> * 在機率密度高的地方（山頂附近），向量的幅度很小，因為已經很接近最可能的數據點了。

### 2. Score Matching

**Score Matching** 是一種訓練神經網路的方法，使其能夠**學習**我們未知的 Score Function。

我們的目標是訓練一個神經網路模型 $s_\theta(x)$（參數 $\theta$ 為權重），使其盡可能**等於**真實的 Score Function $\nabla_x \log p(x)$。

**目標：** $s_\theta(x) \approx \nabla_x \log p(x)$

**如何訓練**
最直觀的方法是最小化這兩者之間的均方誤差（Mean Squared Error），這個損失函數稱為 **Fisher Divergence**：

$$
L(\theta) = \mathbb{E}_{p(x)} \left[ \left\| s_\theta(x) - \nabla_x \log p(x) \right\|^2 \right]
$$

> **問題：** 這個損失函數是**無法計算**的！
> 要計算它，我們需要知道 $\nabla_x \log p(x)$，而這又需要我們知道 $p(x)$。但 $p(x)$ 就是我們一開始不知道的東西。

### 3. 經典 Score Matching 

這就是 Score Matching 第一個特別之處。Aapo Hyvärinen 證明了，在一定條件下，上述的「不可計算」的損失函數，可以等價地轉換為一個**完全可計算**的損失函數：

$$
L(\theta) = \mathbb{E}_{p(x)} \left[ \text{trace}(\nabla_x s_\theta(x)) + \frac{1}{2} \left\| s_\theta(x) \right\|^2 \right]
$$

* $\text{trace}(\nabla_x s_\theta(x))$：這是模型 $s_\theta(x)$ 的雅可比矩陣（Jacobian）的跡數（Trace）。
* $\| s_\theta(x) \|^2$：這是模型輸出的 L2 範數的平方。

**優點：** 這個新的損失函數**不再需要 $\nabla_x \log p(x)$**！我們只需要從數據集 $p(x)$ 中**採樣**數據點 $x$（這我們有），然後把它們餵給模型 $s_\theta(x)$ 並計算模型的輸出和導數即可。

**新問題：** 這個方法雖然在理論上可行，但在實務上很困難。如果 $x$ 是一個高維向量（例如一張圖片，維度 $k$ 非常大），計算雅可比矩陣的跡數 $\text{trace}(\nabla_x s_\theta(x))$ 在計算成本上是非常高的。

### 4. Denoising Score Matching (DSM)

這是 **Denoising Score Matching (降噪分數匹配)**（Pascal Vincent, 2011）的切入點，它是解決上述計算昂貴問題的關鍵，也是現代擴散模型的核心。

DSM 的思想是：**與其學習 $p(x)$（乾淨數據）的 Score，不如去學習一個「加噪後」的數據分佈 $q_\sigma(\tilde{x})$ 的 Score。**

1.  **加噪 (Perturbation)：** 我們先定義一個加噪過程。從真實數據 $x \sim p(x)$ 中採樣，然後給它加上一個已知的高斯噪聲 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$，得到加噪後的數據 $\tilde{x} = x + \epsilon$。這個加噪後的數據 $\tilde{x}$ 來自一個新的、平滑後的分佈 $q_\sigma(\tilde{x})$。

2.  **新目標：** 我們訓練模型 $s_\theta(\tilde{x})$ 去匹配這個**加噪分佈的 Score**，即 $\nabla_{\tilde{x}} \log q_\sigma(\tilde{x})$。

3.  **DSM 的魔法：** DSM 證明了，學習这个加噪分佈的 Score，其損失函數可以被簡化為一個**極其簡單**的形式：

$$
L_{DSM}(\theta) = \mathbb{E}_{p(x), \mathcal{N}(\tilde{x}|x, \sigma^2 I)} \left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(\tilde{x}|x) \right\|^2 \right]
$$


這個 $\nabla_{\tilde{x}} \log p(\tilde{x}|x)$ 是什麼？它只是「給定 $x$ ，得到 $\tilde{x}$」的機率的梯度。
* $p(\tilde{x}|x) = \mathcal{N}(\tilde{x}|x, \sigma^2 I) \propto \exp\left(-\frac{\|\tilde{x}-x\|^2}{2\sigma^2}\right)$
* $\log p(\tilde{x}|x) = C - \frac{\|\tilde{x}-x\|^2}{2\sigma^2}$
* $\nabla_{\tilde{x}} \log p(\tilde{x}|x) = \nabla_{\tilde{x}} \left( C - \frac{\|\tilde{x}-x\|^2}{2\sigma^2} \right) = - \frac{\tilde{x}-x}{\sigma^2}$

4.  **DSM 損失函數：**
    把上面的結果代回去，我們得到 DSM 損失函數：

$$
L_{DSM}(\theta) = \mathbb{E}_{p(x), \epsilon} \left[ \left\| s_\theta(x+\epsilon) - \left( - \frac{\epsilon}{\sigma^2} \right) \right\|^2 \right]
$$

    （註：實作中常會省略 $\sigma^2$ 或調整權重，訓練模型 $s_\theta(x+\epsilon)$ 直接去預測 $\epsilon$，這在形式上等價。）

**DSM 的意義：**
我們把一個「計算 $\nabla_x \log p(x)$」的**不可解問題**，轉變成了一個「計算雅可比矩陣跡數」的**昂貴問題**，最後又轉變成了一個「**預測噪聲**」的**簡單迴歸問題**。

訓練過程變成了：
1.  從數據集拿一張乾淨圖片 $x$。
2.  生成一個隨機高斯噪聲 $\epsilon$。
3.  製造一張加噪圖片 $\tilde{x} = x + \epsilon$。
4.  將 $\tilde{x}$ 餵給神經網路 $s_\theta$。
5.  要求神經網路的輸出 $s_\theta(\tilde{x})$ 盡可能等於 $- \frac{\epsilon}{\sigma^2}$ （或者說，就是去預測 $\epsilon$）。

這就是**降噪自動編碼器（Denoising Autoencoder）**！這就是為什麼 Score Matching 和 Denoising 是緊密相連的。

---

## 二、Score Matching 如何用於生成模型

現在我們有了一個強大的工具：一個神經網路 $s_\theta(x_t, t)$，它學會了**在任意噪聲等級 $t$（或 $\sigma_t$）下，為加噪數據 $x_t$ 提供「去噪指引」（即 Score）**。

Score-Based (Diffusion) Generative Models 利用這個工具來「反轉」一個加噪過程。

### 1. 前向過程 (Forward Process / Diffusion)

* 這是一個固定的、不可學習的過程。
* 我們從一張真實圖片 $x_0$ 開始。
* 我們定義一個「時間表」，在 $T$ 個步驟中逐漸對 $x_0$ 添加高斯噪聲。
* $x_0 \to x_1 \to x_2 \to \dots \to x_T$
* 在每一步 $t$，我們都添加少量噪聲。
* 直到最後， $x_T$ 變成了一張**純粹的高斯噪聲圖片**（服從 $\mathcal{N}(0, I)$）。

### 2. 訓練 (Training)

* 我們訓練一個**單一的**神經網路 $s_\theta(x_t, t)$。
* 這個網路的任務是：給定**任意**時刻 $t$ 的加噪圖片 $x_t$，請估計出該時刻的 Score Function $\nabla_{x_t} \log p(x_t)$。
* 我們如何訓練它？使用我們剛才學到的 **Denoising Score Matching (DSM)**！
* **訓練迴圈：**
    1.  隨機從數據集挑一張 $x_0$。
    2.  隨機挑一個時間步 $t$（即一個噪聲等級 $\sigma_t$）。
    3.  根據 $x_0$ 和 $t$ 計算出加噪後的 $x_t$（這一步有固定的數學公式，例如 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$）。
    4.  將 $x_t$ 和 $t$ 餵給神經網路 $s_\theta(x_t, t)$。
    5.  計算 DSM 損失：要求 $s_\theta(x_t, t)$ 的輸出盡可能等於 $\nabla_{x_t} \log p(x_t|x_0)$（即預測用來生成 $x_t$ 的噪聲 $\epsilon$）。
    6.  反向傳播，更新 $\theta$。

### 3. 生成 (Sampling / Reverse Process)

這就是見證奇蹟的時刻。我們現在要**反轉**加噪過程，從一張噪聲圖片生成一張真實圖片。

* **Langevin Dynamics (朗之萬動力學)** 是實現這一點的核心採樣器。
* Langevin Dynamics 採樣的基本思想是：
    * 從一個隨機點 $x$ 開始。
    * **第一步（Drift / 漂移）：** 計算 $x$ 所在的 Score $\nabla_x \log p(x)$，並沿著這個方向走一小步。這一步會讓 $x$ 向著機率更高的區域移動（即「上山」）。
    * **第二步（Diffusion / 擴散）：** 加上一個微小的隨機噪聲。這一步確保我們不會卡在某個局部的山峰，而是能探索整個機率分佈。

* **生成過程：**
    1.  **起始：** 從 $t=T$ 開始，生成一張純粹的高斯噪聲圖片 $x_T \sim \mathcal{N}(0, I)$。
    2.  **迭代：** 進行 $T$ 次迭代，從 $t=T$ 逐步遞減到 $t=1$。
    3.  在每一步 $t$：

        a.  將當前的噪聲圖片 $x_t$ 和時間 $t$ 餵給訓練好的 Score 模型，得到**估計的 Score**： $\hat{s} = s_\theta(x_t, t)$。

        b.  使用這個估計的 Score $\hat{s}$，執行一步 Langevin Dynamics（或更先進的採樣器，如 DDPM/DDIM 更新規則）。這一步會計算出一個「稍微去噪」的 $x_{t-1}$。

        c.  這個更新步驟的直觀意義是：「**嘿，模型 $s_\theta$ 告訴我『上山』（去噪）的方向在這裡，我先往這個方向走一步，然後再隨機晃動一下。**」
    4.  **結束：** 當 $t=0$ 時，我們得到的 $x_0$ 就是一張全新的、由模型生成的高擬真度圖片。

---

## 總結

1.  **Score Matching** 是一種學習數據 Score Function ( $\nabla_x \log p(x)$ ) 的方法。
2.  **Denoising Score Matching (DSM)** 是一種計算上可行的 Score Matching 變體，它透過將問題轉化為「**預測噪聲**」（即降噪）來規避了原版方法中棘手的數學計算。
3.  在**擴散模型**中，DSM 被用來訓練一個神經網路 $s_\theta(x_t, t)$，使其能夠在**所有噪聲等級 $t$** 下估計 Score。
4.  在**生成**新數據時，模型從純噪聲 $x_T$ 開始，利用 $s_\theta(x_t, t)$ 估計的 Score 作為「**指引**」，逐步反轉加噪過程，一步步「**去噪**」，最終得到一張清晰的樣本 $x_0$。

---

# Question

The **score-based/diffusion** models discussed so far all seem to rely heavily on using Gaussian noise in the forward pass.

Question: Why we choose Gaussian noise? If we use other types of noise in the forward pass (e.g., uniform noise or a blurring process), does the mathematical derivation of the DSM still hold? How would the model's performance be affected?
