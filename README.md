# DeepONet實現集合：從簡單函數到複雜PDE算子學習

本項目提供了DeepONet（Deep Operator Network）的完整實現集合，包含從簡單數學函數到複雜偏微分方程的各種算子學習範例。項目展示了DeepONet在函數到函數映射學習上的強大能力。

## 🎯 項目概述

DeepONet是基於萬能逼近定理的深度學習架構，專門用於學習**算子**（函數到函數的映射）。與傳統神經網路學習數值到數值的映射不同，DeepONet學習的是：

```
傳統NN: 數值 → 數值    (如：x → f(x))
DeepONet: 函數 → 函數   (如：u(x) → G[u](x))
```

## 📁 項目結構

```
DeepONet/
├── 📁 簡單函數算子例子/
│   ├── simple_deeponet_sin.py      # 正弦函數算子 (微分/積分)
│   ├── simple_deeponet_sigmoid.py   # Sigmoid函數算子 (微分/縮放)
│   └── simple_deeponet_exp.py       # 指數函數算子 (指數/對數)
│
├── 📁 複雜PDE算子例子/
│   ├── elliptic_pde_deeponet.py           # 橢圓PDE算子
│   ├── reaction_diffusion_deeponet.py     # 反應-擴散方程算子
│   └── elliptic_2d_example.py             # 2D橢圓PDE計算複雜度示例
│
├── requirements.txt                 # 依賴包
└── README.md                       # 本文件
```

## 🚀 快速開始

### 安裝依賴

```bash
pip install -r requirements.txt
```

### 運行範例

```bash
# 簡單函數算子（推薦初學者）
python simple_deeponet_sin.py       # 學習sin函數的微分/積分
python simple_deeponet_sigmoid.py   # 學習sigmoid函數變換
python simple_deeponet_exp.py       # 學習指數/對數變換

# 複雜PDE算子（進階）
python elliptic_pde_deeponet.py     # 橢圓PDE求解器學習
python reaction_diffusion_deeponet.py  # 反應-擴散方程學習
```

## 📚 範例詳細說明

### 🔸 簡單函數算子系列

這些例子適合理解DeepONet的基本概念，計算時間短，結果直觀。

#### 1. 正弦函數算子 (`simple_deeponet_sin.py`)

**學習目標：**
- **微分算子**: sin(ax+b) → a×cos(ax+b)
- **積分算子**: sin(ax+b) → -cos(ax+b)/a

**特點：**
- 10,000訓練樣本，5,000訓練輪次
- 清晰的數學關係，易於驗證
- 2x2網格可視化測試結果

#### 2. Sigmoid函數算子 (`simple_deeponet_sigmoid.py`)

**學習目標：**
- **微分算子**: σ(ax+b) → a×σ(ax+b)×(1-σ(ax+b))
- **縮放算子**: σ(ax+b) → k×σ(ax+b)

**特點：**
- 適合機器學習應用場景
- 展示DeepONet在激活函數上的應用
- 函數值範圍 (0,1)，數值穩定

#### 3. 指數函數算子 (`simple_deeponet_exp.py`)

**學習目標：**
- **指數算子**: f(x) → exp(f(x))
- **對數算子**: f(x) → ln(f(x))

**特點：**
- 強非線性變換
- 包含數值穩定性處理
- 互逆算子關係展示

### 🔸 複雜PDE算子系列

這些例子展示DeepONet在實際科學計算中的應用，適合深入研究。

#### 1. 橢圓PDE算子 (`elliptic_pde_deeponet.py`)

**數學模型：**
```
-∇·(a(x)∇u) = f(x)  in Ω = [0,1]
u = 0  on ∂Ω
```

**算子映射：** 擴散係數 a(x) → 解 u(x)

**特點：**
- 使用高斯隨機場生成訓練數據
- 5,000訓練樣本，64網格點
- 有限差分法PDE求解器

#### 2. 反應-擴散算子 (`reaction_diffusion_deeponet.py`)

**數學模型：**
```
∂u/∂t = D∇²u + f(u)  in Ω = [0,1]
u(x,0) = u₀(x)
```

**算子映射：** 初始條件 u₀(x) → 最終解 u(x,T)

**特點：**
- 使用切比雪夫多項式生成初始條件
- 3,000訓練樣本，80網格點
- 時間演化求解器

## 🎯 可視化格式

所有程式採用統一的可視化格式：

### 訓練過程
- **單張訓練損失圖** (8×6英寸)
- 對數尺度顯示收斂過程
- 清晰的標題和軸標籤

### 測試結果
- **2×2網格佈局** (12×10英寸)
- 每個子圖顯示：
  - 🟢 輸入函數 (綠色實線)
  - 🔵 真實結果 (藍色實線) 
  - 🔴 DeepONet預測 (紅色虛線)

## ⚙️ 技術配置

### 通用設置
- **網路架構**: Branch Net (64-64-64) + Trunk Net (64-64-64)
- **優化器**: Adam (學習率 1e-3)
- **損失函數**: 均方誤差 (MSE)
- **激活函數**: ReLU

### 不同程式的特定設置

| 程式 | 訓練樣本 | 訓練輪次 | 網格點數 | 特殊處理 |
|------|----------|----------|----------|----------|
| Sin函數 | 10,000 | 5,000 | 50 | 週期函數 |
| Sigmoid | 10,000 | 5,000 | 50 | 範圍(0,1) |
| 指數/對數 | 10,000 | 5,000 | 50 | 數值裁剪 |
| 橢圓PDE | 5,000 | 5,000 | 64 | 稀疏求解器 |
| 反應-擴散 | 3,000 | 5,000 | 80 | 時間積分 |

## 🧠 理論背景

### DeepONet架構

```
輸入函數 u(x) ──┐
               ├─→ Branch Net ──┐
               │                ├─→ 內積 + 偏置 → G[u](y)
查詢位置 y ────┴─→ Trunk Net ───┘
```

### 萬能逼近定理

對於連續算子 G，存在神經網路逼近：
```
G[u](y) ≈ Σᵢ₌₁ᵖ bᵢ(u) · tᵢ(y) + b₀
```

其中：
- bᵢ(u): Branch網路輸出
- tᵢ(y): Trunk網路輸出

## 🎓 學習路徑建議

### 初學者 (理解基本概念)
1. 先運行 `simple_deeponet_sin.py`
2. 觀察微分算子的學習過程
3. 理解Branch-Trunk架構

### 進階學習 (探索不同算子)
1. 嘗試 `simple_deeponet_sigmoid.py`
2. 比較不同函數族的學習難度
3. 運行 `simple_deeponet_exp.py` 體驗強非線性

### 專業應用 (實際科學計算)
1. 學習 `elliptic_pde_deeponet.py`
2. 理解PDE求解的資料生成
3. 挑戰 `reaction_diffusion_deeponet.py`

## 🔬 應用領域

### 已實現的算子類型
- ✅ 微分算子 (導數計算)
- ✅ 積分算子 (積分變換)
- ✅ 非線性變換 (指數、對數、Sigmoid)
- ✅ PDE求解算子 (橢圓方程、拋物方程)

### 潛在應用領域
- 🌊 流體力學 (Navier-Stokes方程)
- 📡 信號處理 (濾波、去噪、信道等化)
- 🔬 量子力學 (薛定諤方程)
- 💰 金融數學 (期權定價PDE)
- 🌡️ 傳熱學 (熱傳導方程)

## ⚠️ 注意事項

### 計算資源
- **簡單函數算子**: 2-5分鐘 (推薦入門)
- **PDE算子**: 10-30分鐘 (需要耐心)

### 數值穩定性
- 指數函數使用裁剪防止溢出
- 對數函數確保輸入為正值
- PDE求解器使用稀疏矩陣技術

### 參數調整
- 可根據需要調整訓練樣本數
- 網路深度和寬度可自定義
- 學習率需要針對不同問題調優

## 🚀 擴展方向

### 近期可實現
- 🔄 平移算子 f(x) → f(x-c)
- 📐 絕對值算子 f(x) → |f(x)|
- 🎵 傅立葉變換算子
- 🌊 小波變換算子

### 長期研究
- 🌍 多維PDE (2D/3D)
- ⏰ 時空耦合算子
- 🎛️ 參數化PDE族
- 🤖 物理約束神經網路

## 📖 參考文獻

1. Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229.

2. Chen, T., & Chen, H. (1995). Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. *IEEE Transactions on Neural Networks*, 6(4), 911-917.

3. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303-314.

---

**開始你的DeepONet學習之旅吧！** 🎉

從簡單的sin函數開始，逐步探索算子學習的神奇世界！ 