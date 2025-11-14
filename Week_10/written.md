# Written assignment 

## 第一題題目

給定一個前向 SDE (Stochastic Differential Equation)：  
$$dx_t = f(x_t, t) dt + g(x_t, t) dW_t$$

證明其對應的 Probability Flow ODE (常微分方程) 可以寫為：  
$$dx_t = \left[f(x_t, t) - \frac{1}{2} \frac{\partial}{\partial x} g^2(x_t, t) - \frac{g^2(x_t, t)}{2} \frac{\partial}{\partial x} \log p(x_t, t) \right] dt$$

---

### 1. 想法

證明的核心是利用機率密度 $p(x_t, t)$ 的演化。SDE 和其對應的 Probability Flow ODE 共享**完全相同**的機率密度演化路徑。

1.  **SDE $\to$ Fokker-Planck 方程 (FPE)**：首先，寫出 SDE 所對應的 Fokker-Planck 方程，它描述了 SDE 粒子群體的機率密度 $p(x_t, t)$ 如何隨時間演進。
2.  **ODE $\to$ 連續性方程**：接著，寫出目標 ODE 所對應的連續性方程，它描述了 ODE 粒子群體的機率密度 $p(x_t, t)$ 如何隨時間演進。
3.  **劃上等號**：由於兩者的機率密度 $p(x_t, t)$ 必須相同，我們令 FPE 和連續性方程相等，然後解出 ODE 的漂移項 (drift term)，即可完成證明。

---

### 2. 過程

(為簡潔，我在推導中省略 $(x_t, t)$ ，將 $p(x_t, t)$ 記為 $p$ ， $f(x_t, t)$ 記為 $f$ ， $g(x_t, t)$ 記為 $g$ )

### Step 1： SDE 的 Fokker-Planck 方程 (FPE)

給定的前向 SDE 為：  
$$dx_t = f dt + g dW_t$$

其機率密度 $p(x, t)$ 的演化由 **Fokker-Planck 方程**描述。對於此 SDE，FPE 的具體形式為：  
$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x} [fp] + \frac{1}{2} \frac{\partial^2}{\partial x^2} [g^2p]$$
**(Equation A)**

### Step 2： ODE 的連續性方程 (Continuity Equation)

假設存在一個確定性的 Probability Flow ODE，其形式為：
$$dx_t = F dt$$
其中 $F = F(x_t, t)$ 是我們想要找到的未知漂移係數。

此 ODE 的機率密度 $p(x, t)$ 的演化由**連續性方程**描述：
$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x} [Fp]$$
**(Equation B)**

### Step 3：令方程相等並求解 $F$

為了讓 SDE 和 ODE 具有相同的機率密度演化 $p(x, t)$，我們必須令 **(Equation A) = (Equation B)**：
$$-\frac{\partial}{\partial x} [F p] = -\frac{\partial}{\partial x} [f p] + \frac{1}{2} \frac{\partial^2}{\partial x^2} [g^2 p]$$

接著對等式兩邊同時關於 $x$ 積分（並假設在 $\pm\infty$ 處的機率通量為 0）：
$$-[F p] = -[f p] + \frac{1}{2} \frac{\partial}{\partial x} [g^2 p]$$

兩邊同乘 -1：
$$Fp = fp - \frac{1}{2} \frac{\partial}{\partial x}[g^2 p]$$

接下來，使用**乘法求導法則** $\frac{\partial}{\partial x}(uv) = (\frac{\partial u}{\partial x})v + u(\frac{\partial v}{\partial x})$ 來展開 $\frac{\partial}{\partial x} [g^2 p]$：

$$\frac{\partial}{\partial x} [g^2 p] = \left( \frac{\partial g^2}{\partial x} \right) p + g^2 \left( \frac{\partial p}{\partial x} \right)$$

將這個展開式代回原方程：
$$F p = f p - \frac{1}{2} \left[ \left( \frac{\partial g^2}{\partial x} \right) p + g^2 \left( \frac{\partial p}{\partial x} \right) \right]$$

然後，在等式兩邊同時除以 $p$（假設 $p \neq 0$），以解出 $F$：
$$F = f - \frac{1}{2} \left( \frac{\partial g^2}{\partial x} \right) - \frac{1}{2} g^2 \left( \frac{1}{p} \frac{\partial p}{\partial x} \right)$$

### Step 4：引入 Score Function

最後注意最後一項 $\left( \frac{1}{p} \frac{\partial p}{\partial x} \right)$。
根據**Score Function (分數函數)** 的定義，  
$$\frac{\partial}{\partial x} \log p = \frac{1}{p} \frac{\partial p}{\partial x}$$

將這個 Score Function 的定義代入我們解出的 $F$ 中：  
$$F = f - \frac{1}{2} \frac{\partial g^2}{\partial x} - \frac{g^2}{2} \frac{\partial}{\partial x} \log p$$

這正是題目要求證明的 ODE 漂移係數 $F$。
因此，我們證明了與前向 SDE 對應的 probability flow ODE 為：  
$$dx_t = \left[ f(x_t, t) - \frac{1}{2} \frac{\partial}{\partial x} g^2(x_t, t) - \frac{g^2(x_t, t)}{2} \frac{\partial}{\partial x} \log p(x_t, t) \right]dt$$

**證明完畢。**

## 第二題題目

### 1. AI 的未來能力

#### 能力描述：AI 視覺法律輔助系統

我認為 AI 在未來 20 年有望實現一項突破性的能力：**從影像或影片中自動分析法律事件並生成具有推理步驟的法律報告**。

* **「做什麼」的事：**
    以交通故為例，AI 系統能直接讀取交通監視器影像，自動辨識出各車輛與行人的行為，判斷如「誰闖紅燈」、「碰撞發生的因果順序」等關鍵事實。接著，AI 能依據內建的交通法規知識庫，自動歸納並劃分肇事責任比例，最終產出一份包含事實依據與法律推理步驟的正式報告。

* **應用場景：**
    1.  **自動化交通事故處理：** 警方接獲報案後，系統自動調閱路口影像，在幾分鐘內產出初步的肇事責任報告，員警僅需複核，大幅縮短現場處理時間。
    2.  **保險快速理賠：** 保險公司能依據 AI 報告，在數小時內完成責任認定並核定理賠，提升客戶滿意度。
    3.  **法庭證據分析：** 在法庭上，此報告可作為客觀的輔助證據，幫助法官理解複雜的事故動態。

* **「為什麼重要」：**
    此能力對人類社會意義重大。
    1.  **效率：** 現階段肇事責任認定需大量人工調查，效率低下。AI 自動化將大幅縮短事故處理時間，減少交通壅塞與警力負擔。
    2.  **公正性：** 人工判斷易受主觀影響，而 AI 透過一致且透明的算法，能確保責任認定的客觀與公正。
    3.  **可行性：** 雖然目前 AI 在精確責任比例劃分上仍有困難，但 GPT-4V 等多模態模型的進展，已展現了強大的影像與文本推理能力。隨著技術成熟，此系統在 20 年後極有可能成為現實。

---

### 2. 機器學習類型

要實現上述能力，預計採用**混合式機器學習方案**，以**監督式學習 (Supervised Learning)** 為主體，輔以**自我監督學習 (Self-supervised Learning)**（例如 LLM 預訓練）和**符號式 AI（Symbolic AI）**。

* **為何需要這類學習：**
    1.  **監督式學習**是必要的，因為系統的核心能力（如辨識車輛、偵測碰撞、判斷紅綠燈）需要從大量「已標註」的資料中學習。
    2.  **自我監督學習**（如大型語言模型）擅長從海量的無標註法律文本中學習語言結構與法律常識，這對於「生成報告」至關重要。
    3.  **符號式 AI**（如法律知識圖譜）是確保 AI 推理「合法合規」的關鍵。它提供一組明確的規則（如交通法規），讓模型的輸出有據可循。

* **任務中的「資料來源」與「目標訊號」：**
    * **資料來源：**
        1.  **（監督式）影像資料：** 大量已標註的交通事故監視器影片。標註內容包括：「車輛/行人/號誌」的邊界框、「碰撞」發生的時間點、「闖紅燈」等違規事件標籤。
        2.  **（監督式）案例資料：** 已有人類專家（如交警、法官）解析好的「事故影像」與對應的「責任歸屬報告」。
        3.  **（自我監督）文本資料：** 海量的法律文書、交通法規、過往判決書。
    * **目標訊號：**
        1.  **（監督式）影像模型：** 預測的標註框、事件分類標籤，應與人工標註的「真實答案」盡可能一致。
        2.  **（監督式）推理模型：** 針對一個事故案例，模型輸出的「肇事責任歸屬」應與人類專家的「標準答案」一致。

* **是否存在學習回饋或環境互動：**
    此任務**主要不是透過與環境即時互動來學習**。它不像玩遊戲需要即時的獎懲（強化學習）。模型主要是透過「離線」的監督式學習，從一個固定的、已標註好的資料集中學習如何正確地「模仿」人類專家的判斷。因此，強化學習在此並非核心途徑，但可在模擬環境中作為輔助優化策略。

---

### 3. 第一步的「模型化」

#### 簡化模型問題：自動判定十字路口雙車碰撞的闖紅燈責任

作為實現 20 年後最終目標的第一步，我設計一個簡化的模型化問題：**系統從一段十字路口的監視器影片中，自動判定一場「雙車碰撞」事故，是否由「闖紅燈」行為造成，並指出違規方。**

* **這個簡化問題如何代表最終能力：**
    這個問題是最終系統的一個核心子集。它聚焦於最常見的事故場景（路口、雙車、紅綠燈），完美地涵蓋了三個關鍵模組：

    1.  **影像理解**（辨識車輛、辨識紅綠燈）
    2.  **事件識別**（判斷碰撞、判斷車輛是否在紅燈時穿越路口）
    3.  **法律規則應用**（將「闖紅燈」與「碰撞」進行因果連結，並依據「闖紅燈應負全責」的法規進行判斷）
    如果我們能解決這個問題，就等於驗證了從影像感知到法律推理這條核心路徑的可行性。

* **它的可測試性（如何知道模型是否成功）：**
    這個簡化模型的成功標準非常明確且易於衡量：
    1.  **辨識準確率 (Accuracy)：** 模型對「車輛」和「紅綠燈狀態」的辨識準確率必須高於 95%。
    2.  **事件偵測 F1-Score：** 針對「闖紅燈」和「碰撞」這兩類關鍵事件，模型的 F1-Score（綜合考量準確率與召回率）必須達到業界可用水準。
    3.  **責任判定準確率：** 這是最重要的指標。在所有包含闖紅燈的事故中，模型「正確揪出」違規方（車 A 或車 B）的準確率。我們可以蒐集 100 段影像，先由 3 位交通專家進行「盲測」並達成一致結論，以此作為「標準答案」，來評估 AI 的判斷準確率。

* **需要哪些數學或機器學習工具來解決：**
    1.  **物件偵測 (YOLOv11 / Faster R-CNN)：** 這是必要的工具，用於從影片中精確定位並追蹤「車輛」的軌跡，以及辨識「交通號誌燈」的當前狀態（紅、黃、綠）。
    2.  **時間序列分析 / 事件辨識 (Transformer / LSTM)：** 僅靠靜態圖片無法判斷，我們需要能處理時序的 Transformer 模型來分析車輛軌跡與號誌時序的關係，以判斷「在紅燈亮起時，車輛A的軌跡是否穿越了停止線」。
    3.  **碰撞偵測 (光流法 / 影像差異)：** 透過分析連續幀之間的像素劇烈變化（例如使用光流法）或物體間距，來自動標記「碰撞」發生的確切時間點。
    4.  **規則引擎 / 知識圖譜：** 一個小型的「符號式AI」模組。它儲存了簡化的法規。
    5.  **最終決策 (簡單邏輯)：** `IF (is_violation(Vehicle_A) AND is_collision(Vehicle_A, Vehicle_B)) THEN responsible_party = Vehicle_A`。

## 第三題題目

In the derivation of 1D inverse SDE, we can replace $ds$ with $-dt$ and obtain the final $dx_{t}$ equation because $dW_t$ and $-dW_t$ are equal in distribution ($dW_t \overset{d}{=} -dW_t$).

Question : Why can this random term (Wiener process) be handled in this way during time reversal? Would this assumption still hold if we used other non-Gaussian noise?