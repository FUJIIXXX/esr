import torch
import torch.nn as nn
import dhg
from dhg.nn import HGNNConv
import numpy as np
# ==================== 1. 数据生成 ====================
class MobileEdgeSystem:
    """模拟移动边缘计算系统，生成超图数据"""
    
    def __init__(self, n_edge_nodes=2, n_mobile_devices=10, n_tasks=5):
        self.n_edge = n_edge_nodes
        self.n_mobile = n_mobile_devices
        self.n_tasks = n_tasks
        self.n_nodes = n_edge_nodes + n_mobile_devices + n_tasks
        
    def generate_hypergraph(self):
        """
        生成超图结构
        返回: (hypergraph, node_features, node_types)
        """
        # 节点类型: 0=边缘节点, 1=移动设备, 2=任务节点
        node_types = torch.zeros(self.n_nodes, dtype=torch.long)
        node_types[self.n_edge:self.n_edge+self.n_mobile] = 1
        node_types[self.n_edge+self.n_mobile:] = 2
        
        # 节点特征 (延迟、吞吐量、能耗等指标)
        node_features = torch.randn(self.n_nodes, 16)
        
        # 构建超边
        hyperedges = []
        
        # 1. 边缘-设备连接超边 (每个边缘节点连接其管辖的设备)
        for e_id in range(self.n_edge):
            devices = np.random.choice(
                range(self.n_edge, self.n_edge + self.n_mobile),
                size=np.random.randint(3, 6),
                replace=False
            )
            hyperedges.append([e_id] + devices.tolist())
        
        # 2. 任务执行超边 (每个任务连接1个边缘节点 + 若干设备)
        for t_id in range(self.n_edge + self.n_mobile, self.n_nodes):
            edge_node = np.random.randint(0, self.n_edge)
            n_devices = np.random.randint(1, 4)  # 粗粒度1个，细粒度多个
            devices = np.random.choice(
                range(self.n_edge, self.n_edge + self.n_mobile),
                size=n_devices,
                replace=False
            )
            hyperedges.append([edge_node, t_id] + devices.tolist())
        
        # 使用DHG构建超图
        hg = dhg.Hypergraph(self.n_nodes, hyperedges)
        
        return hg, node_features, node_types
# ==================== 2. GNN编码器 ====================
class HypergraphEncoder(nn.Module):
    """超图神经网络编码器"""
    
    def __init__(self, in_dim=16, hidden_dim=64, out_dim=32, n_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # 第一层
        self.convs.append(HGNNConv(in_dim, hidden_dim))
        
        # 中间层
        for _ in range(n_layers - 2):
            self.convs.append(HGNNConv(hidden_dim, hidden_dim))
        
        # 输出层
        self.convs.append(HGNNConv(hidden_dim, out_dim))
        
        self.activation = nn.ReLU()
        
    def forward(self, x, hg):
        """
        x: 节点特征 [n_nodes, in_dim]
        hg: DHG超图对象
        返回: 节点embeddings [n_nodes, out_dim]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, hg)
            if i < len(self.convs) - 1:
                x = self.activation(x)
        return x
# ==================== 3. 任务调度器 (贪心启发式) ====================
class GreedyScheduler:
    """基于交互强度的贪心调度器"""
    
    def __init__(self, alpha_L=0.4, alpha_T=0.3, alpha_E=0.3, threshold=0.6):
        self.alpha_L = alpha_L
        self.alpha_T = alpha_T
        self.alpha_E = alpha_E
        self.threshold = threshold
    
    def compute_interaction_strength(self, task_emb, device_emb):
        """
        计算任务和设备之间的交互强度
        简化版: 使用余弦相似度模拟
        """
        similarity = torch.cosine_similarity(task_emb, device_emb, dim=-1)
        return (similarity + 1) / 2  # 归一化到[0,1]
    
    def schedule_tasks(self, embeddings, node_types, n_tasks, n_devices):
        """
        为每个任务分配设备
        返回: 调度决策 {task_id: [device_ids]}
        """
        # 提取任务和设备的embeddings
        task_mask = node_types == 2
        device_mask = node_types == 1
        
        task_embs = embeddings[task_mask]  # [n_tasks, emb_dim]
        device_embs = embeddings[device_mask]  # [n_devices, emb_dim]
        
        schedule = {}
        
        for t_id in range(n_tasks):
            task_emb = task_embs[t_id].unsqueeze(0)  # [1, emb_dim]
            
            # 计算与所有设备的交互强度
            strengths = self.compute_interaction_strength(
                task_emb.expand(n_devices, -1), 
                device_embs
            )
            
            # 贪心策略: 选择交互强度最大的设备
            max_strength = strengths.max().item()
            
            if max_strength > self.threshold:
                # 粗粒度: 分配给单个最优设备
                best_device = strengths.argmax().item()
                schedule[t_id] = [best_device]
            else:
                # 细粒度: 分配给Top-K设备
                k = min(3, n_devices)
                top_k_devices = torch.topk(strengths, k).indices.tolist()
                schedule[t_id] = top_k_devices
        
        return schedule
# ==================== 4. 主验证流程 ====================
def main():
    print("=== 移动边缘计算超图调度系统 ===\n")
    
    # 1. 生成系统数据
    print("Step 1: 生成超图数据...")
    system = MobileEdgeSystem(n_edge_nodes=2, n_mobile_devices=8, n_tasks=5)
    hg, node_features, node_types = system.generate_hypergraph()
    
    print(f"节点数: {hg.num_v}")
    print(f"超边数: {hg.num_e}")
    print(f"边缘节点: {system.n_edge}, 移动设备: {system.n_mobile}, 任务: {system.n_tasks}\n")
    
    # 2. 构建GNN编码器
    print("Step 2: 构建超图神经网络...")
    encoder = HypergraphEncoder(in_dim=16, hidden_dim=64, out_dim=32, n_layers=2)
    encoder.eval()
    
    # 3. 编码超图
    print("Step 3: 超图前向传播...")
    with torch.no_grad():
        embeddings = encoder(node_features, hg)
    print(f"节点embeddings形状: {embeddings.shape}\n")
    
    # 4. 贪心调度
    print("Step 4: 执行贪心调度...")
    scheduler = GreedyScheduler(threshold=0.6)
    schedule = scheduler.schedule_tasks(
        embeddings, 
        node_types, 
        system.n_tasks, 
        system.n_mobile
    )
    
    # 5. 输出结果
    print("\n=== 调度结果 ===")
    for task_id, device_ids in schedule.items():
        granularity = "粗粒度" if len(device_ids) == 1 else "细粒度"
        print(f"任务 {task_id}: {granularity} -> 设备 {device_ids}")
    
    print("\n✓ 验证完成！超图建模和GNN编码已跑通。")
if __name__ == "__main__":
    main()