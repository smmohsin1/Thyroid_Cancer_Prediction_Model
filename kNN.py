from crud import insert 

def squared_euclidean_distance(p1, p2):
    """
    Calculates the squared Euclidean distance between two points
    Using squared distance during search avoids expensive square root calculations
    
    """
    return sum((x - y) ** 2 for x, y in zip(p1, p2))


def get_max_distance(node):
    """
    Finds the worst (largest) distance by traveling down the right branch

    """

    if node is None:
        return float('inf')
        
    while node['right'] is not None:
        node = node['right']
        
    return node['point'][0] # Distance is stored as a 1D point


def remove_max(node):
    """
    Removes the rightmost node (worst distance) to keep size at k
    """

    if node is None:
        return None
        
    # If there is no right child, this node is the maximum. 
    if node['right'] is None:
        return node['left']
        
    node['right'] = remove_max(node['right'])
    return node


def flatten_neighbors(node, result_list):
    """
    In-order traversal to extract neighbors sorted from closest to furthest
    """

    if node is not None:
        flatten_neighbors(node['left'], result_list)
        
        # The neighbor dictionary is stored in the 'label'
        neighbor_data = node['label']
        neighbor_data['distance'] = neighbor_data['distance'] ** 0.5
        result_list.append(neighbor_data)
        
        flatten_neighbors(node['right'], result_list)


def search_knn(current_node, target_point, k, tracker):
    """
    Recursive helper function to traverse the main 9D KD-Tree.
    Updates the 1D Neighbor KD-Tree in place.
    Average time complexity: O(log n)
   """

    if current_node is None:
        return

    dist_sq = squared_euclidean_distance(current_node['point'], target_point)

    neighbor_data = {
        'point': current_node['point'],
        'label': current_node['label'],
        'distance': dist_sq
    }

    # Retrieve the worst distance currently in our 1D Neighbor Tree
    worst_dist = get_max_distance(tracker['root'])

    # Tree updating logic using your UNIVERSAL insert function!
    # We pass the distance as a 1D point: [dist_sq]
    # We pass the dictionary as the label: neighbor_data
    if tracker['count'] < k:
        tracker['root'] = insert(tracker['root'], [dist_sq], neighbor_data)
        tracker['count'] += 1
    elif dist_sq < worst_dist:
        tracker['root'] = insert(tracker['root'], [dist_sq], neighbor_data)
        tracker['root'] = remove_max(tracker['root'])

    axis = current_node['axis']
    diff = target_point[axis] - current_node['point'][axis]

    if diff < 0:
        good_side = current_node['left']
        bad_side = current_node['right']
    else:
        good_side = current_node['right']
        bad_side = current_node['left']

    search_knn(good_side, target_point, k, tracker) # Search the half of the tree where the target point is located

    current_worst = get_max_distance(tracker['root']) # Pruning step
    
    if tracker['count'] < k or (diff ** 2) <= current_worst:
        search_knn(bad_side, target_point, k, tracker)


def get_knn(node, target_point, k):
    """
    Main function to initialize the K-Nearest Neighbors search.
    Returns a sorted list of dictionaries.
    """
    tracker = {'root': None, 'count': 0} 
    
    search_knn(node, target_point, k, tracker)

    k_nearest = []
    flatten_neighbors(tracker['root'], k_nearest)
        
    return k_nearest