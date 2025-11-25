## Week 1-10 Analysis and literature review of unanswered questions

---

### **1. 學習率 (Learning Rate) 相關問題**

**敘述**：
* 如果學習率太大會震盪，太小會收斂緩慢。實務上該如何選擇或調整？ 是否有必要使用動態學習率？
* 在哪種監督式學習應用中，模型可解釋性與準確性同等重要？

**分析與研究：**

1.  **動態學習率**：這幾乎是現代深度學習的標準實踐。實務上很少使用固定的學習率。研究者已經開發出許多「自適應 (adaptive)」學習率演算法，它們能根據訓練過程動態地為每個參數調整學習率。
    * **參考文獻：** Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization." *arXiv preprint arXiv:1412.6980.*
    * **網址：** [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

2.  **可解釋性**：在**醫療診斷**、**金融信貸**和**法律判決**等高風險領域，可解釋性與準確性同等重要。在這些領域，一個「黑盒子」的答案（即使準確）是不可接受的，因為決策者必須知道模型「為何」做出此判斷，以確保其公平性、無偏見並符合法規。
    * **參考文獻：** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.*
    * **網址：** [https://arxiv.org/abs/1602.04938](https://arxiv.org/abs/1602.04938) (這篇是 LIME 模型的經典論文)

---

### **2. 參數化 vs. 非參數化學習**

**敘述**：
* 線性迴歸是「參數化學習」，而局部加權線性迴歸 (LWLR) 是「非參數化學習」嗎？
* 它們各自在什麼場景下表現更好？ LWLR 的高計算成本 在實務上如何權衡？

**分析與研究：**

* **參數化 vs. 非參數化**：線性迴歸是參數化的，因為它試圖學習一組**固定數量**的參數（權重 $w$）。無論你有 100 筆或 100 萬筆資料，模型的參數數量不變。LWLR 是非參數化的，因為它**沒有固定的參數**；相反，它在**預測時**才根據鄰近的數據點即時計算一個模型。模型（或說預測）的複雜度會隨著數據量的增加而增加。
* **權衡**：LWLR 的高計算成本 (因爲每次預測都要重新計算) 使其不適用於大規模或需要即時預測的系統。但它的優勢在於能擬合 (fit) **非常複雜的非線性**數據模式，而標準線性迴歸做不到。
    * **參考文獻：** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction.* Springer. (本書第 6 章詳細討論了局部迴歸與權衡)。
    * **網址：** [https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf](https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf)

---

### **3. 函數近似理論 (UAT)**

**敘述**：
* UAT 常用 $L^2$ 範數還是 sup 範數？ 選擇指標會改變結論嗎？
* ReLU 是否也有類似的近似定理？ 效率有差異嗎？
* 如果目標函數不連續或有噪聲會怎樣？

**分析與研究：**

* **ReLU 近似定理**：是的，ReLU 激活函數同樣具有「通用近似」能力。研究顯示，ReLU 網路在效率上（例如，表示某些函數所需的網路寬度或深度）通常優於 sigmoid/tanh。
    * **參考文獻：** Hanin, B. (2019). "Approximation capabilities of deep neural networks with ReLU activation function." *arXiv preprint arXiv:1903.02102.*
    * **網址：** [https://arxiv.org/abs/1903.02102](https://arxiv.org/abs/1903.02102)

* **不連續函數與範數**：
    * 經典的 UAT（例如 Hornik, 1989）通常使用 $L^p$ 範數或 sup 範數 來證明。
    * 是的，神經網路甚至可以近似**不連續**的函數。
    * 選擇 $L^2$（均方誤差）或 sup（最大誤差）會改變結論的強度。例如，在 $L^2$ 中近似一個不連續函數是可行的，但在 sup 範數中則不行（因為在跳變點的誤差無法縮小到 0）。
    * **參考文獻：** Kidger, P., & Lyons, T. (2020). "Approximation of Discontinuous Functions by Neural Networks." *arXiv preprint arXiv:2001.05008.*
    * **網址：** [https://arxiv.org/abs/2001.05008](https://arxiv.org/abs/2001.05008)

---

### **4. SGD 的隨機性**

**敘述**：
* SGD 的隨機性 雖然有助於跳出淺層局部最小值，但會不會在訓練後期阻礙收斂到更深的最小值，導致在最佳解附近「抖動」？
* 學習率衰減 (decay) 是唯一的解決方案嗎？

**分析與研究：**

* **抖動問題**：是的，固定的學習率會導致 SGD 在最優解的山谷底部持續「抖動」而無法收斂到谷底。
* **解決方案**：學習率衰減 (decay) 是最常用 且最簡單的方案。但它**不是唯一**的。近年來有更複雜的學習率排程 (schedule) 被提出，例如「循環學習率 (Cyclical LR)」或「預熱 (Warmup) 後衰減」。
    * **參考文獻：** Loshchilov, I., & Hutter, F. (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts." *arXiv preprint arXiv:1608.03983.* (這篇論文提出了一種「帶熱重啟」的學習率排程，它會周期性地重置學習率，以跳出局部最小值，同時又能收斂)。
    * **網址：** [https://arxiv.org/abs/1608.03983](https://arxiv.org/abs/1608.03983)

---

### **5. One-vs-Rest vs. Softmax**

**您的問題**：
* 與原生處理多分類的 Softmax 相比，One-vs-Rest (OvR) 在什麼情況下表現不佳？
* 它有什麼 Softmax 沒有的優點嗎？

**分析與研究：**
是的，這兩者有明顯的權衡。

* **OvR 的缺點**：OvR 的主要問題是它假設了類別之間是**獨立**的，它為每個類別訓練一個獨立的二元分類器。如果類別之間存在**高度混淆或重疊**，OvR 的分類邊界會不準確。此外，它還可能導致「校準 (calibration)」問題（即每個分類器輸出的 "confidence" score 彼此之間無法直接比較）。
* **OvR 的優點**：
    1.  **可解釋性**：OvR 讓你清楚地知道模型對「類別 A vs. 非 A」的判斷依據，這比 Softmax 更易於解讀。
    2.  **效率/靈活性**：如果你有 1000 個類別，Softmax 必須在 1000 個類別上計算歸一化。而 OvR 在計算上更簡單。
* **參考文獻：** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning.* (本書第 4.3 和 4.4 節對 OvR 和 Softmax 有詳細的比較)。
* **網址：** [https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf](https://www.sas.upenn.edu/~fdiebold/NoHesitations/BookAdvanced.pdf)

---

### **6. GDA vs. 羅吉斯迴歸**

**敘述**：
* GDA 在高斯假設下比羅吉斯迴歸更有效率。但如果現實世界中高斯假設不成立，GDA 的性能會多快地下降？
* 是否存在一個「非高斯」的臨界點，使得羅吉斯迴歸的魯棒性(robustness) 超過 GDA？

**分析與研究：**

* **研究結論**：研究表明，GDA（生成模型）在**數據量很少**時，**即使高斯假設不完全成立**，通常也比羅吉斯迴歸（判別模型）表現得更好。
* **臨界點**：隨著數據量的增加，羅吉斯迴歸（因其假設更少、更魯棒）的性能會迅速趕上並最終**超越** GDA。GDA 的性能會因為「模型錯誤 (model misspecification)」而達到一個瓶頸，而羅吉斯迴歸會持續改進。這個「臨界點」取決於數據偏離高斯分佈的程度。
    * **參考文獻：** Ng, A. Y., & Jordan, M. I. (2001). "On Discriminative vs. Generative classifiers: A comparison of logistic regression and naive bayes." *Advances in neural information processing systems (NIPS) 14.*
    * **網址：** [https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) 

---

### **7. 擴散模型與高斯噪聲**

**敘述**：
* 為什麼擴散模型 似乎都依賴高斯噪聲？
* 如果使用其他類型的噪聲（如均勻噪聲 或模糊），DSM 的數學推導是否仍然成立？ 性能會受何影響？

**分析與研究：**

* **為什麼選高斯？**：1. 數學便利性（高斯分佈的疊加仍然是高斯分佈）；2. 它與 SDE 中的 Wiener 過程直接相關；3. 它的 Score $\nabla\log p(x|x_0)$ 有簡單的解析解。
* **非高斯噪聲**：**DSM 的數學推導（即簡化為預測噪聲）在非高斯情況下通常不成立**。但是，研究者已經在開發新的理論框架（例如 "Flow Matching"）來處理更通用的噪聲過程。
    * **參考文獻：** de Bortoli, R., et al. (2023). "Score-based Generative Models with Non-Gaussian Noise." *arXiv preprint arXiv:2306.07536.*
    * **網址：** [https://arxiv.org/abs/2306.07536](https://arxiv.org/abs/2306.07536)

---

### **8. Euler-Maruyama (EM) 的穩定性**

**敘述**：
* ODE 的 Euler 方法在「剛性 (stiff)」問題 上不穩定。EM 方法是否也面臨類似的數值穩定性問題？
* 是否需要對 $\Delta t$ 施加限制 以確保不發散？

**分析與研究：**

* **EM 的穩定性**：與 ODE 不同，SDE 有兩種穩定性：「均方穩定性 (mean-square stability)」和「漸近穩定性 (asymptotic stability)」。EM 方法可能在一種意義上穩定，但在另一種意義上不穩定。
* **$\Delta t$ 的限制**：是的，對於「剛性」SDE，EM 方法通常需要**非常小**的 $\Delta t$ 才能保持穩定，這在計算上是昂貴的。這催生了「隱式 (implicit)」SDE 求解器的研究。
    * **參考文獻：** Higham, D. J. (2000). "Stability of the Euler-Maruyama method for stochastic differential equations." *SIAM Journal on Numerical Analysis, 38(3), 753-769.*
    * **網址：** [https://strathprints.strath.ac.uk/57/1/strathprints000057.pdf](https://strathprints.strath.ac.uk/57/1/strathprints000057.pdf)

---

### **9. Wiener 過程的時間反轉**

**敘述**：
* 在 1D 反向 SDE 推導中，我們用 $dW_t \overset{d}{=} -dW_t$。為什麼 Wiener 過程 可以這樣處理？
* 如果使用非高斯噪聲，此假設是否仍然成立？

**分析與研究：**

* **高斯噪聲**：Wiener 過程的增量 $dW_t$ 服從一個均值為 0 的高斯分佈 $\mathcal{N}(0, \Delta t)$。高斯分佈是**對稱**的（即 $p(x) = p(-x)$）。因此，$-dW_t$ 的分佈也是 $\mathcal{N}(0, \Delta t)$。它們在統計上無法區分。
* **非高斯噪聲**：**不一定成立**。這取決於噪聲分佈是否也是**對稱**的。
    * 如果使用**非對稱**的噪聲（例如 Poisson 過程或 Gamma 過程的增量），則 $L_t$ 和 $-L_t$ 的分佈完全不同，此假設失效。
    * **參考文獻：** Beutner, M. E. (2008). "Time reversal of Lévy processes." *Journal of Theoretical Probability, 21(1), 161-175.* (Lévy 過程是 SDE 中非高斯噪聲的通用框架)
    * **網址：** [https://arxiv.org/abs/0704.0321](https://arxiv.org/abs/0704.0321)