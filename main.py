import pandas as pd
import networkx as nx
import math
import os
from itertools import islice

# ==========================================
# 1. Physical Layer Parameters (Table 1 & 2)
# ==========================================
# Assuming each Frequency Slot Unit (FSU) = 12.5 GHz
MODULATIONS = [
    {"name": "DP-16QAM",    "max_length": 500,  "capacity": 400, "slots": 6},  # 75GHz / 12.5 = 6 slots
    {"name": "SC-DP-16QAM", "max_length": 700,  "capacity": 200, "slots": 3},  # 37.5GHz / 12.5 = 3 slots
    {"name": "SC-DP-QPSK",  "max_length": 2000, "capacity": 100, "slots": 3}   # 37.5GHz / 12.5 = 3 slots
]

NUM_SLOTS = 320  # The number of available slots is 320

# ==========================================
# 2. Topology and Traffic Functions
# ==========================================
def build_topology(file_path):
    print(f"Reading topology file: {file_path}")
    try:
        # Read the file skipping the header
        df = pd.read_csv(file_path, sep='\s+', header=None, engine='python', skiprows=1)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None

    G = nx.Graph()
    for index, row in df.iterrows():
        source = str(int(row[3]))      # Fourth column is source
        target = str(int(row[4]))      # Fifth column is destination
        distance = float(row[5])       # Sixth column is link length (km)
        G.add_edge(source, target, weight=distance)
    return G

def initialize_spectrum(G, num_slots=NUM_SLOTS):
    """Initializes the spectrum array for each link. 0 = free, 1 = occupied."""
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
                    "bitrate": float(bitrate) * 10  # Convert to Gbps based on matrix units
                })
                request_id_counter += 1
    return requests

# ==========================================
# 3. Routing and Modulation Logic
# ==========================================
def get_k_shortest_paths(G, source, destination, k=5):
    """
    Finds up to K shortest paths using Yen's algorithm (via shortest_simple_paths).
    """
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
    """Selects the highest capacity modulation format that supports the path distance."""
    for modulation in MODULATIONS:
        if distance <= modulation["max_length"]:
            return modulation
    return None  

def calculate_required_slots(bitrate, modulation):
    """Calculates required FSUs based on chunk bitrate and modulation capacity."""
    num_subcarriers = math.ceil(bitrate / modulation['capacity'])
    total_slots = num_subcarriers * modulation['slots']
    return total_slots

# ==========================================
# 4. Spectrum Allocation & Fragmentation (NoC)
# ==========================================
def find_first_fit_slot(G, path, slots_needed):
    """Finds the first available block of spectrum satisfying continuity and contiguity constraints."""
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

def allocate_slots(G, path, start_index, slots_needed):
    """Marks spectrum slots as occupied (1) along the path."""
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        for slot in range(start_index, start_index + slots_needed):
            G[u][v]['spectrum'][slot] = 1

def deallocate_slots(G, path, start_index, slots_needed):
    """Frees spectrum slots (0) along the path (Used for simulation tracking)."""
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        for slot in range(start_index, start_index + slots_needed):
            G[u][v]['spectrum'][slot] = 0

def calculate_total_noc(G):
    """
    Calculates the Number of Cuts (NoC) across the entire network.
    Group 5 objective: Minimize spectrum fragmentation (NoC).
    """
    total_noc = 0
    for u, v in G.edges():
        spectrum = G[u][v]['spectrum']
        for i in range(len(spectrum) - 1):
            if spectrum[i] != spectrum[i+1]:
                total_noc += 1
    return total_noc

# ==========================================
# 5. RMSA Algorithms
# ==========================================
def serve_request_benchmark(G, paths_info, req):
    """
    Benchmarking Algorithm: 
    Uses fixed shortest path for routing and First Fit Spectrum Assignment.
    """
    if not paths_info:
        return False
    
    # Strictly use the first path (Fixed Shortest Path)
    p_info = paths_info[0]
    path = p_info['path']
    mod = select_modulation(p_info['distance'])
    if not mod:
        return False
    
    remaining_bitrate = req["bitrate"]
    max_capacity = mod['capacity']
    allocated_chunks = []
    
    # Divide traffic requests exceeding max capacity
    while remaining_bitrate > 0:
        chunk = min(remaining_bitrate, max_capacity)
        req_slots = calculate_required_slots(chunk, mod)
        start_slot = find_first_fit_slot(G, path, req_slots)
        
        if start_slot is not None:
            allocate_slots(G, path, start_slot, req_slots)
            allocated_chunks.append((start_slot, req_slots))
            remaining_bitrate -= chunk
        else:
            # Rollback if any chunk fails
            for s_slot, r_slots in allocated_chunks:
                deallocate_slots(G, path, s_slot, r_slots)
            return False
            
    return True

def serve_request_custom(G, paths_info, req):
    """
    Group 5 Custom Algorithm (NoC-aware Heuristic):
    Evaluates up to K=5 paths. Simulates First-Fit allocation on each, 
    calculates the resulting network-wide NoC, and selects the path that 
    minimizes spectrum fragmentation.
    """
    if not paths_info:
        return False

    best_path_idx = -1
    best_allocation_plan = []
    min_noc = float('inf')
    
    for i, p_info in enumerate(paths_info):
        path = p_info['path']
        mod = select_modulation(p_info['distance'])
        if not mod:
            continue # Path is too long for any modulation format
            
        remaining_bitrate = req["bitrate"]
        max_capacity = mod['capacity']
        
        temp_allocated = []
        success = True
        
        # Simulate chunk allocation on the current candidate path
        while remaining_bitrate > 0:
            chunk = min(remaining_bitrate, max_capacity)
            req_slots = calculate_required_slots(chunk, mod)
            start_slot = find_first_fit_slot(G, path, req_slots)
            
            if start_slot is not None:
                allocate_slots(G, path, start_slot, req_slots)
                temp_allocated.append((start_slot, req_slots, chunk))
                remaining_bitrate -= chunk
            else:
                success = False
                break
                
        if success:
            # Calculate NoC if this path was chosen
            current_noc = calculate_total_noc(G)
            if current_noc < min_noc:
                min_noc = current_noc
                best_path_idx = i
                best_allocation_plan = temp_allocated.copy()
                
        # Rollback the simulation to evaluate the next candidate path
        for start_slot, req_slots, chunk in temp_allocated:
            deallocate_slots(G, path, start_slot, req_slots)
            
    # Formally allocate slots using the path that yielded the lowest NoC
    if best_path_idx != -1 and best_allocation_plan:
        best_path = paths_info[best_path_idx]['path']
        for start_slot, req_slots, chunk in best_allocation_plan:
            allocate_slots(G, best_path, start_slot, req_slots)
        return True
        
    return False

# ==========================================
# 6. Main Simulation Runner
# ==========================================
def run_simulation(G_clean, traffic_file, algorithm_type, order_descending):
    """Executes a full simulation for a specific matrix, algorithm, and sorting order."""
    G = G_clean.copy()
    initialize_spectrum(G)
    
    requests = load_traffic_matrix(traffic_file)
    # Sort traffic demands based on size
    requests = sorted(requests, key=lambda x: x['bitrate'], reverse=order_descending)
    
    total_reqs = len(requests)
    allocated_count = 0
    
    for req in requests:
        paths_info = get_k_shortest_paths(G, req["source"], req["destination"], k=5)
        
        if algorithm_type == 'Benchmark':
            success = serve_request_benchmark(G, paths_info, req)
        else:
            success = serve_request_custom(G, paths_info, req)
            
        if success:
            allocated_count += 1
            
    blocked_count = total_reqs - allocated_count
    blocking_ratio = (blocked_count / total_reqs) * 100 if total_reqs > 0 else 0
    final_noc = calculate_total_noc(G)
    
    return total_reqs, allocated_count, blocked_count, blocking_ratio, final_noc

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    topology_file = os.path.join(current_dir, "data", "Network Italian 10-node", "IT10-topology.txt")
    traffic_file_template = os.path.join(current_dir, "data", "Network Italian 10-node", "IT10-matrix-{}.txt")
    
    base_graph = build_topology(topology_file)
    if not base_graph:
        print("Topology generation failed. Please check file paths.")
        exit()

    algorithms = ['Benchmark', 'Custom(NoC-aware)']
    orders = [('Asc', False), ('Desc', True)]

    print("="*105)
    print(f"{'Matrix':<8} | {'Algorithm':<20} | {'Order':<6} | {'Reqs':<5} | {'Alloc/Blk':<10} | {'BP(%)':<7} | {'Total NoC':<10}")
    print("-" * 105)

    for i in range(1, 6):
        traffic_file = traffic_file_template.format(i)
        if not os.path.exists(traffic_file):
            continue
            
        for algo in algorithms:
            for order_name, is_desc in orders:
                total, alloc, blocked, bp, noc = run_simulation(base_graph, traffic_file, algo, is_desc)
                print(f"M{i:<7} | {algo:<20} | {order_name:<6} | {total:<5} | {alloc}/{blocked:<8} | {bp:<7.2f} | {noc:<10}")
    print("="*105)