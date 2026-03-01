import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def plot_spectrum_heatmap(G, title="Spectrum Utilization"):
    """
    画出全网频谱占用热力图 (Heatmap)
    这是最直观展示“碎片化”和“拥堵”的图表
    """
    edges = list(G.edges())
    num_edges = len(edges)
    # 获取第一条边的频谱长度，假设所有边长度一致 (通常是320)
    num_slots = len(G[edges[0][0]][edges[0][1]]['spectrum'])
    
    # 1. 准备数据矩阵 (行=链路, 列=频隙)
    spectrum_matrix = np.zeros((num_edges, num_slots))
    
    edge_labels = []
    for i, (u, v) in enumerate(edges):
        # 把列表里的 0/1 填入矩阵
        spectrum_matrix[i, :] = G[u][v]['spectrum']
        edge_labels.append(f"{u}-{v}")

    # 2. 开始画图
    plt.figure(figsize=(15, 8)) # 设置画布大小
    
    # cmap='Greys' 让 0显示白色(空闲)，1显示黑色(占用)
    # 你也可以改成 'Reds' 或 'Blues'
    plt.imshow(spectrum_matrix, aspect='auto', cmap='Reds', interpolation='nearest')
    
    # 3. 设置坐标轴和标签
    plt.title(title, fontsize=16)
    plt.xlabel("Frequency Slots (Index)", fontsize=12)
    plt.ylabel("Network Links (Source-Dest)", fontsize=12)
    
    # 在纵轴显示具体的链路名字
    plt.yticks(range(num_edges), edge_labels, fontsize=8)
    
    # 加个颜色条说明 (0=Free, 1=Occupied)
    cbar = plt.colorbar()
    cbar.set_label('Occupancy (1=Used, 0=Free)')
    
    plt.tight_layout()
    plt.show()

def draw_topology_with_path(G, path=None):
    """
    画出拓扑图，如果给了 path，就高亮这条路径
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42) # 固定布局，保证每次画出来形状一样
    
    # 1. 画背景（所有节点和边）
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=600, edge_color='lightgray', width=2)
    
    # 2. 如果有路径，高亮它
    if path:
        # 生成路径上的边列表 [(1,2), (2,5)...]
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=4)
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange', node_size=600)
        
    plt.title(f"Topology View {'(Red = Selected Path)' if path else ''}")
    plt.show()