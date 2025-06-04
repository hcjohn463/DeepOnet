import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 設置隨機種子
torch.manual_seed(42)
np.random.seed(42)

class SimpleDeepONet(nn.Module):
    """簡單的DeepONet學習sigmoid函數算子"""
    
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim=64, output_dim=64):
        super(SimpleDeepONet, self).__init__()
        
        # Branch Network (編碼輸入函數)
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Trunk Network (編碼查詢位置)
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 偏置項
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, branch_input, trunk_input):
        # Branch network output
        branch_out = self.branch_net(branch_input)  # [batch_size, output_dim]
        
        # Trunk network output  
        trunk_out = self.trunk_net(trunk_input)      # [n_points, output_dim]
        
        # 計算內積
        if branch_out.dim() == 2 and trunk_out.dim() == 2:
            output = torch.mm(branch_out, trunk_out.T) + self.bias
        else:
            output = torch.sum(branch_out * trunk_out, dim=-1) + self.bias
            
        return output

def sigmoid(x):
    """Sigmoid函數"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 防止數值溢出

def generate_sigmoid_derivative_data(n_samples, x_points):
    """生成sigmoid函數的微分算子訓練數據
    
    算子: 學習微分算子 d/dx
    輸入: sigmoid(a*x + b) = 1/(1+e^(-ax-b))
    輸出: a*sigmoid(a*x + b)*(1-sigmoid(a*x + b))
    """
    
    input_functions = []
    output_functions = []
    
    print(f"生成 {n_samples} 個sigmoid微分算子樣本...")
    
    for i in range(n_samples):
        # 隨機生成參數
        a = np.random.uniform(0.5, 3.0)  # 斜率參數
        b = np.random.uniform(-2.0, 2.0)  # 偏移參數
        
        # 輸入函數: sigmoid(a*x + b)
        input_func = sigmoid(a * x_points + b)
        
        # 輸出函數: 微分結果 a*sigmoid*(1-sigmoid)
        output_func = a * input_func * (1 - input_func)
        
        input_functions.append(input_func)
        output_functions.append(output_func)
        
        if (i + 1) % 1000 == 0:
            print(f"已完成 {i + 1}/{n_samples} 樣本")
    
    return np.array(input_functions), np.array(output_functions)

def generate_sigmoid_scaling_data(n_samples, x_points):
    """生成sigmoid函數的縮放算子訓練數據
    
    算子: 學習縮放算子 k*f(x)
    輸入: sigmoid(a*x + b)
    輸出: k*sigmoid(a*x + b), 其中k是隨機縮放因子
    """
    
    input_functions = []
    output_functions = []
    
    print(f"生成 {n_samples} 個sigmoid縮放算子樣本...")
    
    for i in range(n_samples):
        # 隨機生成參數
        a = np.random.uniform(0.5, 3.0)  # 斜率參數
        b = np.random.uniform(-2.0, 2.0)  # 偏移參數
        k = np.random.uniform(0.5, 3.0)  # 縮放因子
        
        # 輸入函數: sigmoid(a*x + b)
        input_func = sigmoid(a * x_points + b)
        
        # 輸出函數: 縮放結果 k*sigmoid(a*x + b)
        output_func = k * input_func
        
        input_functions.append(input_func)
        output_functions.append(output_func)
        
        if (i + 1) % 1000 == 0:
            print(f"已完成 {i + 1}/{n_samples} 樣本")
    
    return np.array(input_functions), np.array(output_functions)

def train_simple_deeponet(model, input_funcs, output_funcs, x_points, epochs=5000, lr=1e-3):
    """訓練簡單DeepONet"""
    
    # 準備數據
    branch_inputs = torch.FloatTensor(input_funcs)
    trunk_inputs = torch.FloatTensor(x_points.reshape(-1, 1))
    targets = torch.FloatTensor(output_funcs)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    print("開始訓練DeepONet...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向傳播
        outputs = model(branch_inputs, trunk_inputs)
        loss = criterion(outputs, targets)
        
        # 反向傳播
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses

def test_simple_deeponet(model, x_points, operator_type='derivative', n_test=4):
    """測試簡單DeepONet"""
    
    if (operator_type == 'derivative' or operator_type == 'scaling') and n_test == 4:
        # 針對sigmoid算子的2x2佈局
        plt.figure(figsize=(12, 10))
        
        for i in range(n_test):
            # 生成測試樣本
            a_test = np.random.uniform(0.5, 3.0)
            b_test = np.random.uniform(-2.0, 2.0)
            
            # 輸入函數
            input_test = sigmoid(a_test * x_points + b_test)
            
            # 真實輸出和標籤
            if operator_type == 'derivative':
                output_true = a_test * input_test * (1 - input_test)
                true_label = 'True d/dx'
                pred_label = 'DeepONet d/dx'
                title = f'Sample {i+1}: Sigmoid Derivative Operator'
            else:  # scaling
                k_test = np.random.uniform(0.5, 3.0)
                output_true = k_test * input_test
                true_label = f'True {k_test:.2f}×f(x)'
                pred_label = 'DeepONet Scaling'
                title = f'Sample {i+1}: Sigmoid Scaling Operator'
            
            # DeepONet 預測
            branch_input = torch.FloatTensor(input_test).unsqueeze(0)
            trunk_input = torch.FloatTensor(x_points.reshape(-1, 1))
            
            with torch.no_grad():
                output_pred = model(branch_input, trunk_input).squeeze().numpy()
            
            # 2x2佈局
            plt.subplot(2, 2, i+1)
            plt.plot(x_points, input_test, 'g-', linewidth=2, label=f'Input: σ({a_test:.2f}x + {b_test:.2f})')
            plt.plot(x_points, output_true, 'b-', label=true_label, linewidth=2)
            plt.plot(x_points, output_pred, 'r--', label=pred_label, linewidth=2)
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('Function Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def demo_sigmoid_derivative_operator():
    """演示sigmoid微分算子學習"""
    print("=" * 60)
    print("DeepONet學習sigmoid微分算子: d/dx[σ(ax+b)] = a*σ(ax+b)*(1-σ(ax+b))")
    print("=" * 60)
    
    # 設定參數
    n_points = 50
    x_points = np.linspace(-5, 5, n_points)
    
    # 生成訓練數據
    n_train = 10000
    input_funcs, output_funcs = generate_sigmoid_derivative_data(n_train, x_points)
    
    # 初始化DeepONet
    model = SimpleDeepONet(
        branch_input_dim=n_points,
        trunk_input_dim=1,
        hidden_dim=64,
        output_dim=64
    )
    
    # 訓練模型
    losses = train_simple_deeponet(model, input_funcs, output_funcs, x_points, epochs=5000)
    
    # 繪製收斂性 - 只顯示一張圖
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.title('Sigmoid Derivative Operator Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 測試模型 - 4個樣本排成2x2
    test_simple_deeponet(model, x_points, operator_type='derivative', n_test=4)
    
    print(f"sigmoid微分算子學習完成！最終損失: {losses[-1]:.6f}")

def demo_sigmoid_scaling_operator():
    """演示sigmoid縮放算子學習"""
    print("=" * 60)
    print("DeepONet學習sigmoid縮放算子: k*σ(ax+b)")
    print("=" * 60)
    
    # 設定參數
    n_points = 50
    x_points = np.linspace(-5, 5, n_points)
    
    # 生成訓練數據
    n_train = 10000
    input_funcs, output_funcs = generate_sigmoid_scaling_data(n_train, x_points)
    
    # 初始化DeepONet
    model = SimpleDeepONet(
        branch_input_dim=n_points,
        trunk_input_dim=1,
        hidden_dim=64,
        output_dim=64
    )
    
    # 訓練模型
    losses = train_simple_deeponet(model, input_funcs, output_funcs, x_points, epochs=5000)
    
    # 繪製收斂性 - 只顯示一張圖
    plt.figure(figsize=(8, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.title('Sigmoid Scaling Operator Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 測試模型 - 4個樣本排成2x2
    test_simple_deeponet(model, x_points, operator_type='scaling', n_test=4)
    
    print(f"sigmoid縮放算子學習完成！最終損失: {losses[-1]:.6f}")

def show_concept_explanation():
    """解釋sigmoid DeepONet的基本概念"""
    print("=" * 70)
    print("Sigmoid DeepONet 基本概念解釋")
    print("=" * 70)
    print()
    print("1. Sigmoid函數:")
    print("   σ(x) = 1/(1+e^(-x))")
    print("   - 輸出範圍: (0, 1)")
    print("   - 常用於神經網路激活函數")
    print()
    print("2. DeepONet學習的算子:")
    print("   ┌─────────────────────┐    ┌─────────────────────┐")
    print("   │   微分算子 d/dx     │    │    縮放算子 k×f     │")
    print("   │                     │    │                     │")
    print("   │ σ(ax+b) →           │    │ σ(ax+b) →           │")
    print("   │ a*σ(ax+b)*(1-σ(ax+b))│    │ k*σ(ax+b)          │")
    print("   └─────────────────────┘    └─────────────────────┘")
    print()
    print("3. 本例子特點:")
    print("   - 學習sigmoid函數的各種變換")
    print("   - 比sin函數更適合機器學習應用")
    print("   - 展示DeepONet在不同函數族上的泛化能力")
    print("=" * 70)

def main():
    """主函數"""
    show_concept_explanation()
    
    print("\n選擇要演示的算子:")
    print("1. Sigmoid微分算子 d/dx")
    print("2. Sigmoid縮放算子 k×f(x)")
    print("3. 兩個都演示")
    
    choice = input("請輸入選項 (1/2/3, 直接按Enter默認選3): ").strip()
    
    if choice == '1':
        demo_sigmoid_derivative_operator()
    elif choice == '2':
        demo_sigmoid_scaling_operator()
    else:
        print("\n演示兩個算子...")
        demo_sigmoid_derivative_operator()
        print("\n" + "="*50)
        demo_sigmoid_scaling_operator()
    
    print("\nSigmoid DeepONet演示完成!")
    print("這個例子展示了DeepONet如何學習sigmoid函數的不同算子變換。")
    print("相比於sin函數，sigmoid更貼近實際的機器學習應用場景。")

if __name__ == "__main__":
    main() 