# GNP (Graph Neural Preconditioner) 优化项目总结

## 📋 项目背景

### 问题描述
- **目标**: 使用Graph Neural Network Preconditioners (GNP) 加速GMRES求解器解决2D Poisson方程
- **核心挑战**: GNP在小问题(n=8)有效，但随着问题尺寸增大到n=64时失效
- **研究方法**: 通过贝叶斯优化进行系统性超参数搜索，识别关键瓶颈并提出改进方案

### 问题设置
- **方程**: 2D Poisson方程 ∇²p = f，混合Dirichlet/Neumann边界条件
- **离散化**: 有限差分法，n×n网格，矩阵尺寸n²×n²
- **求解器**: GMRES with restart
- **预条件子**: ResGCN网络学习A^(-1)的近似

## 🎯 贝叶斯优化关键发现

### 最关键超参数排序（按重要性）

1. **`training_data = 'x_normal'`** ⭐⭐⭐⭐⭐
   - **发现**: 比'x_subspace'和'x_mix'效果显著更好
   - **原因**: 简单的正态分布随机向量最有效覆盖解空间
   - **影响**: 这是成败的决定性因素

2. **`num_layers = 3`** ⭐⭐⭐⭐
   - **发现**: 浅层网络比深层网络(8-12层)效果更好
   - **原因**: 深层网络容易过拟合，浅层网络泛化能力更强
   - **反直觉**: 更复杂的网络反而性能更差

3. **`lr = 0.008861` (约0.009)** ⭐⭐⭐⭐
   - **发现**: 比常用学习率(1e-3)高很多
   - **原因**: GNP需要快速学习逆算子，需要更aggressive的更新
   - **关键**: 学习率对训练收敛至关重要

4. **`embed = 61`** ⭐⭐⭐
   - **发现**: 中等嵌入维度最优，不需要很大的维度
   - **平衡**: 表达能力 vs 过拟合风险
   - **经济性**: 61维提供最佳性价比

5. **`drop_rate = 0.07`** ⭐⭐
   - **发现**: 轻微dropout有帮助，过强dropout(0.2+)有害
   - **作用**: 适度正则化防止过拟合
   - **精确性**: 需要精确控制强度

6. **`m = 62`** ⭐⭐
   - **发现**: 中等Krylov子空间维度
   - **作用**: 影响训练数据质量
   - **需要**: 随问题尺寸适应性调整

### 次要但有用的参数
- `hidden = 122` (embed * 2)
- `epochs = 322` (adaptive: 200 + embed * 2)
- `batch_size = 8`
- `weight_decay = 1e-5`
- `scale_input = True`

## 📊 性能验证结果

### 实测性能表现

| 问题尺寸 | 矩阵大小 | Speedup | 迭代减少 | 训练Loss | 是否有效 |
|---------|---------|---------|----------|----------|----------|
| n=16    | 256×256 | 0.72x   | 55.3%    | 8.32e-02 | ❌ |
| n=24    | 576×576 | 1.40x   | 76.8%    | 7.43e-02 | ✅ |
| n=32    | 1024×1024| 1.76x   | 79.7%    | 6.76e-02 | ✅ |

### 关键观察
- **成功率**: 66.7% (2/3)
- **尺寸效应**: 对中大型问题(n≥24)非常有效
- **训练质量**: Loss从~1.0降到~0.07，收敛良好
- **迭代效率**: 能减少70-80%的GMRES迭代数

## 🔍 n=64失效原因分析

### 主要瓶颈

1. **训练成本 vs 求解收益失衡**
   - 矩阵尺寸: 4096×4096 (比n=32大16倍)
   - 训练时间增长: 可能需要8-15秒
   - 基础GMRES时间: 2-8秒
   - **问题**: 训练时间增长超过求解时间节省

2. **网络容量与问题复杂度不匹配**
   - 当前配置: embed=61, layers=3
   - n=64复杂度: 是n=32的4-16倍
   - **容量不足**: 网络无法充分学习大矩阵逆算子

3. **数值稳定性恶化**
   - 条件数增长: O(n²) ≈ O(4096)
   - 精度问题: float32可能不足
   - 误差累积: 大矩阵上近似误差放大

4. **超参数尺寸依赖性**
   - 当前"最优"参数针对n=16-32优化
   - 大问题需要不同的参数配置
   - 缺乏自适应机制

## 🚀 三层次改进方案

### Level 1: 立即可操作改进

#### 1.1 自适应超参数策略
```python
def get_adaptive_config(n):
    scale_factor = (n / 16) ** 0.5  # 平方根缩放
    
    return {
        'embed': min(200, int(61 * scale_factor)),
        'num_layers': min(8, int(3 + np.log2(n/16))),
        'm': min(150, int(62 * scale_factor)),
        'epochs': min(800, int(322 + (n-16) * 15)),
        'lr': 8.9e-3 * (16/n)**0.25,  # 大问题用稍小学习率
        'training_data': 'x_normal',   # 保持最佳设置
        'drop_rate': 0.07,
        'scale_input': True
    }
```

#### 1.2 混合精度训练
```python
# 网络用float32，关键计算用float64
b_out = (self.A @ x_out.to(torch.float64)).to(self.dtype)
loss = F.mse_loss(x_out, x) if self.training_data != 'no_x' else F.mse_loss(b_out, b)
```

#### 1.3 增强训练数据
```python
training_data_mix = {
    'x_normal': 0.4,      # 40% 正态随机
    'x_lowfreq': 0.3,     # 30% 低频分量  
    'x_subspace': 0.2,    # 20% Krylov子空间
    'x_smooth': 0.1       # 10% 平滑函数
}
```

### Level 2: 中期算法改进

#### 2.1 残差预条件子架构 ⭐ (最推荐)
```python
class ResidualGNP(nn.Module):
    def __init__(self, A):
        self.A = A
        self.jacobi_precond = torch.diag(1.0 / torch.diag(A))  # 雅可比预条件子
        self.neural_correction = ResGCN(...)  # 神经网络修正
        
    def apply(self, r):
        # M = M_jacobi + ΔM_neural
        z_jacobi = self.jacobi_precond @ r
        delta_z = self.neural_correction(r)
        return z_jacobi + delta_z
```

#### 2.2 层次化训练策略
```python
# 渐进式训练: n=16 → n=32 → n=64
# 1. 在小问题上预训练
# 2. 固定浅层，在中等问题上微调深层  
# 3. 全网络在大问题上细调
def hierarchical_training(sizes=[16, 32, 64]):
    for i, n in enumerate(sizes):
        if i == 0:
            # 全网络训练
            train_full_network(n)
        else:
            # 固定前几层，只训练后几层
            freeze_early_layers()
            train_later_layers(n)
```

#### 2.3 注意力增强GCN
```python
class AttentionGCN(nn.Module):
    def __init__(self):
        self.local_gcn = GCNConv(...)
        self.global_attention = MultiHeadAttention(...)
        
    def forward(self, x):
        local_feat = self.local_gcn(x)
        global_feat = self.global_attention(local_feat)
        return local_feat + global_feat  # 残差连接
```

#### 2.4 物理约束损失
```python
def physics_informed_loss(x_pred, b, A):
    # 标准预条件子损失
    precon_loss = F.mse_loss(x_pred, x_true)
    
    # 物理约束: Ax = b
    physics_loss = F.mse_loss(A @ x_pred, b)
    
    # 边界条件约束
    boundary_loss = enforce_boundary_conditions(x_pred)
    
    return precon_loss + λ1 * physics_loss + λ2 * boundary_loss
```

### Level 3: 根本性创新

#### 3.1 多尺度GNP
```python
class MultiScaleGNP:
    def __init__(self, levels=[16, 32, 64]):
        self.gnp_coarse = ResGCN(...)   # 粗网格GNP
        self.gnp_fine = ResGCN(...)     # 细网格GNP
        self.restrict = ...             # 限制算子
        self.interpolate = ...          # 插值算子
        
    def apply(self, r):
        # V-cycle类似多重网格
        r_coarse = self.restrict(r)
        z_coarse = self.gnp_coarse(r_coarse)
        z_fine = self.interpolate(z_coarse)
        
        # 细网格修正
        residual = r - self.A @ z_fine
        correction = self.gnp_fine(residual)
        
        return z_fine + correction
```

#### 3.2 自适应网络生长
```python
class AdaptiveGNP:
    def grow_network_if_needed(self):
        if self.training_loss_plateau():
            if self.current_capacity < self.problem_complexity:
                self.add_layer() or self.increase_width()
                
    def training_loss_plateau(self):
        return np.std(self.loss_history[-50:]) < 1e-6
```

#### 3.3 领域分解+GNP
```python
class DomainDecompositionGNP:
    def __init__(self, num_subdomains=4):
        self.subdomains = self.partition_domain(num_subdomains)
        self.sub_gnps = [ResGCN(...) for _ in range(num_subdomains)]
        
    def apply(self, r):
        # 1. 分解到各子域
        sub_residuals = self.distribute_to_subdomains(r)
        
        # 2. 各子域并行GNP求解
        sub_solutions = [gnp.apply(sub_r) for gnp, sub_r in 
                        zip(self.sub_gnps, sub_residuals)]
        
        # 3. 组合解并处理接口
        return self.combine_subdomain_solutions(sub_solutions)
```

## 📅 推荐实施路径

### 短期目标 (1-2周)
1. **实施Level 1.1和1.2**: 自适应参数 + 混合精度
2. **验证n=64效果**: 测试基础改进是否足够
3. **如果仍不够**: 进入中期改进

### 中期目标 (1个月)  
4. **实施Level 2.1**: 残差预条件子架构 (最优先)
5. **实施Level 2.2**: 层次化训练策略
6. **对比分析**: 找到最有效的改进组合

### 长期目标 (2-3个月)
7. **如果需要**: 实施Level 3的根本创新
8. **扩展测试**: n=128, 256等更大问题
9. **方法总结**: 发表改进方法论文

## 🎯 关键配置记录

### 当前最佳配置
```python
best_config = {
    'training_data': 'x_normal',
    'm': 62,
    'embed': 61, 
    'num_layers': 3,
    'lr': 0.008861577452533074,
    'drop_rate': 0.06983140212909128,
    'hidden': 122,
    'epochs': 322,
    'batch_size': 8,
    'weight_decay': 1e-5,
    'scale_input': True
}
```

### 贝叶斯优化搜索空间
```python
search_space = [
    Categorical(['x_normal', 'x_subspace', 'x_mix'], name='training_data'),
    Integer(20, 100, name='m'),
    Integer(16, 128, name='embed'), 
    Integer(3, 12, name='num_layers'),
    Real(-4.0, -2.0, name='lr_log'),  # 1e-4 to 1e-2
    Real(0.0, 0.3, name='drop_rate'),
]
```

## 💡 核心洞察总结

### 成功因素
1. **训练数据质量决定一切**: x_normal >> x_subspace/x_mix
2. **简单架构更健壮**: 浅层网络胜过深层网络
3. **学习率需要足够aggressive**: ~0.009 >> 0.001
4. **问题尺寸有明显threshold**: n≥24才开始有效

### 失败教训
1. **不要盲目增加网络复杂度**: 更深更宽不等于更好
2. **传统ML经验不完全适用**: GNP有其独特规律
3. **尺寸scaling需要特殊考虑**: 不能简单外推小问题经验

### 未来方向
1. **残差架构最有前景**: 结合传统方法优势
2. **自适应机制是关键**: 参数需要随问题调整
3. **多尺度思想值得探索**: 借鉴多重网格成功经验

## 📁 相关文件
- `bayesian_hyperparam_search.py`: 贝叶斯优化实现
- `analyze_best_config.py`: 最佳配置验证
- `simple.py`: 主要实验脚本
- `GNP/nn/ResGCN.py`: 网络架构定义
- `GNP/preconditioners/GNP.py`: 预条件子实现

---
**项目状态**: 已完成超参数优化，准备实施算法改进
**下一步**: 优先实施残差预条件子架构 (Level 2.1)
**最终目标**: 使GNP在n=64及以上问题上超越传统GMRES 