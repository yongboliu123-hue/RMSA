import pandas as pd
import networkx as nx
import math
import os
from itertools import islice


MODULATIONS = [
    {"name": "DP-16QAM",    "max_length": 500,  "capacity": 400, "slots": 6},  
    {"name": "SC-DP-16QAM", "max_length": 700,  "capacity": 200, "slots": 3},  
    {"name": "SC-DP-QPSK",  "max_length": 2000, "capacity": 100, "slots": 3}   
]

NUM_SLOTS = 320  


def build_topology(file_path):
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None, engine='python', skiprows=1)
    except Exception as e:
        return None

    G = nx.Graph()
    for index, row in df.iterrows():
        source = str(int(row[3]))      
        target = str(int(row[4]))      
        distance = float(row[5])       
        G.add_edge(source, target, weight=distance)
    return G

def initialize_spectrum(G, num_slots=NUM_SLOTS):
    for u, v in G.edges():
        G[u][v]['spectrum'] = [0] * num_slots

def load_traffic_matrix(file_path):
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
    except Exception as e:
        return []

    requests = []
    request_id_counter = 0
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            bitrate = df.iloc[i, j]
            if bitrate > 0 and i != j:
                requests.append({
                    "id": request_id_counter,
                    "source": str(i + 1),
                    "destination": str(j + 1),
                    "bitrate": float(bitrate) * 10 
                })
                request_id_counter += 1
    return requests

def get_k_shortest_paths(G, source, destination, k=5):
    try:
        paths_gen = nx.shortest_simple_paths(G, source=source, target=destination, weight='weight')
        paths = list(islice(paths_gen, k))
        paths_info = []
        for p in paths:
            distance = sum(G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1))
            paths_info.append({'path': p, 'distance': distance})
        return paths_info
    except nx.NetworkXNoPath:
        return []

def select_modulation(distance):
    for modulation in MODULATIONS:
        if distance <= modulation["max_length"]:
            return modulation
    return None  

def calculate_required_slots(bitrate, modulation):
    num_subcarriers = math.ceil(bitrate / modulation['capacity'])
    total_slots = num_subcarriers * modulation['slots']
    return total_slots

def find_first_fit_slot(G, path, slots_needed):
    for start_index in range(NUM_SLOTS - slots_needed + 1):
        is_available = True  
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            if 1 in G[u][v]['spectrum'][start_index : start_index + slots_needed]:
                is_available = False  
                break    
        if is_available:
            return start_index
    return None

def find_all_available_slots(G, path, slots_needed):
    available_slots = []
    for start_index in range(NUM_SLOTS - slots_needed + 1):
        is_available = True  
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            if 1 in G[u][v]['spectrum'][start_index : start_index + slots_needed]:
                is_available = False  
                break    
        if is_available:
            available_slots.append(start_index)
    return available_slots

def allocate_slots(G, path, start_index, slots_needed):
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        for slot in range(start_index, start_index + slots_needed):
            G[u][v]['spectrum'][slot] = 1

def deallocate_slots(G, path, start_index, slots_needed):
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        for slot in range(start_index, start_index + slots_needed):
            G[u][v]['spectrum'][slot] = 0

def calculate_total_noc(G):
    total_noc = 0
    for u, v in G.edges():
        spectrum = G[u][v]['spectrum']
        for i in range(len(spectrum) - 1):
            if spectrum[i] != spectrum[i+1]:
                total_noc += 1
    return total_noc

def serve_request_benchmark(G, paths_info, req):
    if not paths_info: return False
    
    p_info = paths_info[0]
    path = p_info['path']
    mod = select_modulation(p_info['distance'])
    if not mod: return False
    
    remaining_bitrate = req["bitrate"]
    max_capacity = mod['capacity']
    allocated_chunks = []
    
    while remaining_bitrate > 0:
        chunk = min(remaining_bitrate, max_capacity)
        req_slots = calculate_required_slots(chunk, mod)
        start_slot = find_first_fit_slot(G, path, req_slots)
        
        if start_slot is not None:
            allocate_slots(G, path, start_slot, req_slots)
            allocated_chunks.append((path, start_slot, req_slots))
            remaining_bitrate -= chunk
        else:
            for b_path, s_slot, r_slots in allocated_chunks:
                deallocate_slots(G, b_path, s_slot, r_slots)
            return False
            
    return True

def serve_request_custom(G, paths_info, req):
    if not paths_info: return False

    remaining_bitrate = req["bitrate"]
    allocated_chunks = [] 
    
    while remaining_bitrate > 0:
        best_chunk_path_idx = -1
        best_chunk_start_slot = -1
        best_chunk_req_slots = -1
        best_chunk_size = 0
        min_noc = float('inf')
        
        for i, p_info in enumerate(paths_info):
            path = p_info['path']
            mod = select_modulation(p_info['distance'])
            if not mod: continue 
                
            max_capacity = mod['capacity']
            chunk_size = min(remaining_bitrate, max_capacity)
            req_slots = calculate_required_slots(chunk_size, mod)
            
            valid_slots = find_all_available_slots(G, path, req_slots)
            
            for start_slot in valid_slots:
                allocate_slots(G, path, start_slot, req_slots)
                current_noc = calculate_total_noc(G)
                
                if current_noc < min_noc:
                    min_noc = current_noc
                    best_chunk_path_idx = i
                    best_chunk_start_slot = start_slot
                    best_chunk_req_slots = req_slots
                    best_chunk_size = chunk_size
                    
                deallocate_slots(G, path, start_slot, req_slots)
                
        if best_chunk_path_idx != -1:
            best_path = paths_info[best_chunk_path_idx]['path']
            allocate_slots(G, best_path, best_chunk_start_slot, best_chunk_req_slots)
            allocated_chunks.append((best_path, best_chunk_start_slot, best_chunk_req_slots))
            remaining_bitrate -= best_chunk_size
        else:
            for b_path, s_slot, r_slots in allocated_chunks:
                deallocate_slots(G, b_path, s_slot, r_slots)
            return False
            
    return True


def run_simulation(G_clean, traffic_file, algorithm_type, order_descending):
    G = G_clean.copy()
    initialize_spectrum(G)
    
    requests = load_traffic_matrix(traffic_file)
    requests = sorted(requests, key=lambda x: x['bitrate'], reverse=order_descending)
    
    total_reqs = len(requests)
    allocated_count = 0
    
    for req in requests:
        paths_info = get_k_shortest_paths(G, req["source"], req["destination"], k=5)
        
        if algorithm_type == 'Benchmark':
            success = serve_request_benchmark(G, paths_info, req)
        else:
            success = serve_request_custom(G, paths_info, req)
            
        if success: allocated_count += 1
            
    blocked_count = total_reqs - allocated_count
    blocking_ratio = (blocked_count / total_reqs) * 100 if total_reqs > 0 else 0
    final_noc = calculate_total_noc(G)
    
    return total_reqs, allocated_count, blocked_count, blocking_ratio, final_noc

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    networks_to_run = [
        {"name": "G17",  "topo": "G17-topology.txt",  "matrix_prefix": "G17-matrix-{}.txt"},
        {"name": "IT10", "topo": "IT10-topology.txt", "matrix_prefix": "IT10-matrix-{}.txt"}
    ]

    algorithms = ['Benchmark', 'Custom(NoC-aware Best-Fit)']
    orders = [('Asc', False), ('Desc', True)]

    print("="*115)
    print(f"{'Network':<8} | {'Matrix':<8} | {'Algorithm':<28} | {'Order':<6} | {'Reqs':<5} | {'Alloc/Blk':<10} | {'BP(%)':<7} | {'Total NoC':<10}")
    print("-" * 115)

    for network in networks_to_run:
        topology_file = os.path.join(current_dir, "data", network["topo"])
        base_graph = build_topology(topology_file)
        
        if not base_graph:
            print(f"Failed to load topology for {network['name']}. Skipping...")
            continue

        for i in range(1, 6):
            traffic_file = os.path.join(current_dir, "data", network["matrix_prefix"].format(i))
            if not os.path.exists(traffic_file): 
                continue
                
            for algo in algorithms:
                for order_name, is_desc in orders:
                    total, alloc, blocked, bp, noc = run_simulation(base_graph, traffic_file, algo, is_desc)
                    print(f"{network['name']:<8} | M{i:<7} | {algo:<28} | {order_name:<6} | {total:<5} | {alloc}/{blocked:<8} | {bp:<7.2f} | {noc:<10}")
                    
        if network != networks_to_run[-1]:
            print("-" * 115)

    print("="*115)
