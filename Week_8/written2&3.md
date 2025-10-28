# 一、SDE (隨機微分方程) 

## 1. 核心概念

它是一種用來描述系統如何**隨時間演進**的數學方程式，其特殊之處在於它**同時包含了「確定性」和「隨機性」的變化**。

## 2. 與常微分方程 (ODE) 的比較

我們可以將 SDE 與 ODE (常微分方程) 進行對比：

* **常微分方程 (ODE, Ordinary Differential Equation):**
    * **形式：** $\frac{dx}{dt} = f(x, t)$
    * **特性：** 這是**確定性 (deterministic)** 的。只要給定一個初始狀態 `x(0)`，未來的整個路徑 `x(t)` 都被**唯一確定**了。它就像在一個固定的向量場中移動，每一步的方向都是確定的。

* **隨機微分方程 (SDE, Stochastic Differential Equation):**
    * **形式：** $dX_t = f(X_t, t) \, dt + g(X_t, t) \, dW_t$
    * **特性：** 這是**隨機性 (stochastic)** 的。即使你給定一個初始狀態 $X_0$，未來也會有**無限多種可能的演進路徑**，SDE 描述的是這些路徑的**機率分佈**。

## 3. SDE 的組成部分

$dX_t = f(X_t, t) \, dt + g(X_t, t) \, dW_t$

這個方程式由兩個關鍵部分組成：

1.  **漂移項 (Drift Term): $f(X_t, t) \, dt$**
    * 這部分是**確定性**的，與 ODE 相同。
    * 它代表了系統在沒有隨機干擾時的「平均」或「預期」運動趨勢。就像一艘船的引擎，提供一個基礎的前進動力。

2.  **擴散項 (Diffusion Term): $g(X_t, t) \, dW_t$**
    * 這部分是**隨機性**的，也是 SDE 的靈魂。
    * $g(X_t, t)$ 是一個函數，用來控制隨機噪聲的**強度或幅度**。
    * $dW_t$ 是**維納過程 (Wiener Process)** 或稱**布朗運動 (Brownian Motion)** 的一個無限小增量。你可以將它想像成在每個瞬間都對系統施加的一個極其微小、完全不可預測的「隨機推力」（高斯白噪聲）。就像作用在船上的隨機波浪。

## 4. SDE 在擴散模型 (Diffusion Models) 中的應用

SDE 是連接 Score Matching 和 Diffusion Models 的核心理論橋樑：

* **前向過程 (Forward Process):**
    將一張乾淨的圖片 $x_0$ 逐步加入噪聲，直到它變成純高斯噪聲 $x_T$ 的過程，可以被精確地建模為一個 SDE。這個 SDE 從 $t=0$ 演進到 $t=T$，其漂移項會使分佈趨向原點，而擴散項則不斷注入噪聲。

* **反向過程 (Reverse Process):**
    這是生成模型的關鍵。研究證明，這個「加噪」的 SDE 存在一個對應的「反向時間 SDE」（Reverse-Time SDE）。這個反向 SDE 描述了如何從一個純噪聲分佈 $x_T$ 出發，**逆著時間**（從 $t=T$ 回到 $t=0$）演進，最終得到一個來自真實數據分佈的樣本 $x_0$。

* **與 Score Function 的連結：**
    這個「反向時間 SDE」的漂移項（即確定性的部分）被證明與**數據分佈的 Score Function ( $\nabla_{x_t} \log p(x_t)$ )** 直接相關。

> **核心思想：** 這就是為什麼需要訓練一個神經網路去「學習」Score Function (即 Score Matching)。**因為只要學會了 Score Function，我們就等於知道了反向 SDE 的完整形式**，進而可以透過數值方法求解這個 SDE 來生成全新的數據。

---

# 二、Question

For ODEs, the Euler method can be numerically unstable when dealing with certain **"stiff"** problems if the time step Δt is not small enough.

**Question**: Does the Euler-Maruyama method also face similar numerical stability issues ? Are there any restrictions on the time step Δt to ensure that the numerical solution does not diverge ?