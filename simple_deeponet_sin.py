import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 設置隨機種子
torch.manual_seed(42)
np.random.seed(42)

class SimpleDeepONet(nn.Module):
    """簡單的DeepONet學習正弦函數算子"""
    
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

def generate_sine_data(n_samples, x_points):
    """生成正弦函數的訓練數據
    
    算子: 學習微分算子 d/dx
    輸入: sin(a*x + b)
    輸出: a*cos(a*x + b)
    """
    
    input_functions = []
    output_functions = []
    
    print(f"生成 {n_samples} 個正弦函數樣本...")
    
    for i in range(n_samples):
        # 隨機生成參數
        a = np.random.uniform(0.5, 3.0)  # 頻率參數
        b = np.random.uniform(0, 2*np.pi)  # 相位參數
        
        # 輸入函數: sin(a*x + b)
        input_func = np.sin(a * x_points + b)
        
        # 輸出函數: 微分結果 a*cos(a*x + b)
        output_func = a * np.cos(a * x_points + b)
        
        input_functions.append(input_func)
        output_functions.append(output_func)
        
        if (i + 1) % 200 == 0:
            print(f"已完成 {i + 1}/{n_samples} 樣本")
    
    return np.array(input_functions), np.array(output_functions)

def generate_integration_data(n_samples, x_points):
    """生成積分算子的訓練數據
    
    算子: 學習積分算子 ∫f(x)dx
    輸入: sin(a*x + b)
    輸出: -cos(a*x + b)/a + C
    """
    
    input_functions = []
    output_functions = []
    
    print(f"生成 {n_samples} 個積分算子樣本...")
    
    for i in range(n_samples):
        # 隨機生成參數
        a = np.random.uniform(0.5, 3.0)  # 頻率參數
        b = np.random.uniform(0, 2*np.pi)  # 相位參數
        
        # 輸入函數: sin(a*x + b)
        input_func = np.sin(a * x_points + b)
        
        # 輸出函數: 積分結果 -cos(a*x + b)/a (忽略常數C)
        output_func = -np.cos(a * x_points + b) / a
        
        input_functions.append(input_func)
        output_functions.append(output_func)
        
        if (i + 1) % 200 == 0:
            print(f"已完成 {i + 1}/{n_samples} 樣本")
    
    return np.array(input_functions), np.array(output_functions)

def train_simple_deeponet(model, input_funcs, output_funcs, x_points, epochs=2000, lr=1e-3):
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
        
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    return losses

def test_simple_deeponet(model, x_points, operator_type='derivative', n_test=4):
    """測試簡單DeepONet"""
    
    if (operator_type == 'derivative' or operator_type == 'integration') and n_test == 4:
        # 針對微分算子和積分算子的2x2佈局
        plt.figure(figsize=(12, 10))
        
        for i in range(n_test):
            # 生成測試樣本
            a_test = np.random.uniform(0.5, 3.0)
            b_test = np.random.uniform(0, 2*np.pi)
            
            # 輸入函數
            input_test = np.sin(a_test * x_points + b_test)
            
            # 真實輸出和標籤
            if operator_type == 'derivative':
                output_true = a_test * np.cos(a_test * x_points + b_test)
                true_label = 'True d/dx'
                pred_label = 'DeepONet d/dx'
                title = f'Sample {i+1}: Derivative Operator'
            else:  # integration
                output_true = -np.cos(a_test * x_points + b_test) / a_test
                true_label = 'True ∫dx'
                pred_label = 'DeepONet ∫dx'
                title = f'Sample {i+1}: Integration Operator'
            
            # DeepONet 預測
            branch_input = torch.FloatTensor(input_test).unsqueeze(0)
            trunk_input = torch.FloatTensor(x_points.reshape(-1, 1))
            
            with torch.no_grad():
                output_pred = model(branch_input, trunk_input).squeeze().numpy()
            
            # 2x2佈局
            plt.subplot(2, 2, i+1)
            plt.plot(x_points, input_test, 'g-', linewidth=2, label=f'Input: sin({a_test:.2f}x + {b_test:.2f})')
            plt.plot(x_points, output_true, 'b-', label=true_label, linewidth=2)
            plt.plot(x_points, output_pred, 'r--', label=pred_label, linewidth=2)
            plt.title(title)
            plt.xlabel('x')
            plt.ylabel('Function Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    else:
        # 原來的3行佈局（用於其他情況）
        plt.figure(figsize=(15, 12))
        
        for i in range(n_test):
            # 生成測試樣本
            a_test = np.random.uniform(0.5, 3.0)
            b_test = np.random.uniform(0, 2*np.pi)
            
            # 輸入函數
            input_test = np.sin(a_test * x_points + b_test)
            
            # 真實輸出
            if operator_type == 'derivative':
                output_true = a_test * np.cos(a_test * x_points + b_test)
                operator_name = "Derivative Operator d/dx"
            else:  # integration
                output_true = -np.cos(a_test * x_points + b_test) / a_test
                operator_name = "Integration Operator ∫dx"
            
            # DeepONet 預測
            branch_input = torch.FloatTensor(input_test).unsqueeze(0)
            trunk_input = torch.FloatTensor(x_points.reshape(-1, 1))
            
            with torch.no_grad():
                output_pred = model(branch_input, trunk_input).squeeze().numpy()
            
            # 可視化輸入函數
            plt.subplot(3, n_test, i+1)
            plt.plot(x_points, input_test, 'g-', linewidth=2)
            plt.title(f'Input Function {i+1}\nsin({a_test:.2f}x + {b_test:.2f})')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.grid(True)
            
            # 可視化輸出結果
            plt.subplot(3, n_test, n_test + i + 1)
            plt.plot(x_points, output_true, 'b-', label='True Result', linewidth=2)
            plt.plot(x_points, output_pred, 'r--', label='DeepONet Prediction', linewidth=2)
            plt.title(f'{operator_name} Result {i+1}')
            plt.xlabel('x')
            plt.ylabel('G(f)(x)')
            plt.legend()
            plt.grid(True)
            
            # 計算誤差
            error = np.abs(output_true - output_pred)
            plt.subplot(3, n_test, 2*n_test + i + 1)
            plt.plot(x_points, error, 'k-', linewidth=2)
            plt.title(f'Absolute Error {i+1}')
            plt.xlabel('x')
            plt.ylabel('|True - Predicted|')
            plt.grid(True)
            plt.yscale('log')
        
        plt.tight_layout()
        plt.show()

def demo_derivative_operator():
    """演示微分算子學習"""
    print("=" * 60)
    print("DeepONet學習微分算子: d/dx[sin(ax+b)] = a*cos(ax+b)")
    print("=" * 60)
    
    # 設定參數
    n_points = 50
    x_points = np.linspace(0, 2*np.pi, n_points)
    
    # 生成訓練數據
    n_train = 10000
    input_funcs, output_funcs = generate_sine_data(n_train, x_points)
    
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
    plt.title('Derivative Operator Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 測試模型 - 4個樣本排成2x2
    test_simple_deeponet(model, x_points, operator_type='derivative', n_test=4)
    
    print(f"微分算子學習完成！最終損失: {losses[-1]:.6f}")

def demo_integration_operator():
    """演示積分算子學習"""
    print("=" * 60)
    print("DeepONet學習積分算子: ∫sin(ax+b)dx = -cos(ax+b)/a")
    print("=" * 60)
    
    # 設定參數
    n_points = 50
    x_points = np.linspace(0, 2*np.pi, n_points)
    
    # 生成訓練數據
    n_train = 10000
    input_funcs, output_funcs = generate_integration_data(n_train, x_points)
    
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
    plt.title('Integration Operator Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 測試模型 - 4個樣本排成2x2
    test_simple_deeponet(model, x_points, operator_type='integration', n_test=4)
    
    print(f"積分算子學習完成！最終損失: {losses[-1]:.6f}")

def show_concept_explanation():
    """解釋DeepONet的基本概念"""
    print("=" * 70)
    print("DeepONet 基本概念解釋")
    print("=" * 70)
    print()
    print("1. 傳統神經網路:")
    print("   輸入: 數字/向量  →  輸出: 數字/向量")
    print("   例如: [1, 2, 3] → [0.8]")
    print()
    print("2. DeepONet:")
    print("   輸入: 函數      →  輸出: 函數")
    print("   例如: sin(2x) → 2*cos(2x)  (微分算子)")
    print()
    print("3. DeepONet架構:")
    print("   ┌─────────────┐    ┌──────────────┐")
    print("   │ Branch Net  │    │  Trunk Net   │")
    print("   │(編碼輸入函數)│    │(編碼查詢位置)│")
    print("   └─────────────┘    └──────────────┘")
    print("           │                   │")
    print("           └───────┬───────────┘")
    print("                   │ 內積")
    print("                   ▼")
    print("              ┌─────────┐")
    print("              │ 輸出函數 │")
    print("              └─────────┘")
    print()
    print("4. 本例子中:")
    print("   - Branch Net: 接收 sin(ax+b) 在各點的值")
    print("   - Trunk Net: 接收查詢位置 x")
    print("   - 輸出: 對應的微分或積分結果")
    print("=" * 70)

def main():
    """主函數"""
    show_concept_explanation()
    
    print("\n選擇要演示的算子:")
    print("1. 微分算子 d/dx")
    print("2. 積分算子 ∫dx")
    print("3. 兩個都演示")
    
    choice = input("請輸入選項 (1/2/3, 直接按Enter默認選3): ").strip()
    
    if choice == '1':
        demo_derivative_operator()
    elif choice == '2':
        demo_integration_operator()
    else:
        print("\n演示兩個算子...")
        demo_derivative_operator()
        print("\n" + "="*50)
        demo_integration_operator()
    
    print("\n簡單DeepONet演示完成!")
    print("這個例子展示了DeepONet如何學習函數到函數的映射關係。")
    print("相比於復雜的PDE，這個例子更容易理解DeepONet的核心思想。")

if __name__ == "__main__":
    main() 