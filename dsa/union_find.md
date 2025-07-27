# Union Find Pattern

## Pattern Overview

The **Union Find** (also known as Disjoint Set Union) pattern is a data structure that efficiently manages and tracks disjoint sets of elements. It provides near-constant time operations for combining sets (union) and determining which set an element belongs to (find). This pattern is particularly powerful for problems involving connectivity, grouping, and dynamic set operations.

## When to Use This Pattern

Your problem matches this pattern if **any** of these conditions is fulfilled:

### ✅ Use Union Find When:

1. **Property-based grouping**: The problem requires arranging elements with a certain property into groups or, to use graph terminology, into connected components.

2. **Set combination**: We have been given a problem that contains elements represented as separate sets initially where we have to combine pairs of sets or find whether two elements belong to the same set or not.

3. **Graph data organization**: The problem data is best organized in the form of a graph, yet the data has not been provided in the form of an adjacency list/matrix.

### ❌ Don't Use Union Find When:

1. **Single connectivity query**: Only one connectivity check is needed (simple DFS/BFS is sufficient)
2. **Complex graph algorithms**: Need shortest paths, topological sorting, or other advanced graph operations
3. **Dynamic edge removal**: Union Find doesn't efficiently support disconnecting elements
4. **Small static datasets**: Simple array operations might be more straightforward

## Core Concepts

### Fundamental Operations

**Find**: Determine which set an element belongs to
- Returns the representative (root) of the set
- Path compression optimization flattens tree structure

**Union**: Combine two sets into one
- Connects the roots of two different sets
- Union by rank/size optimization balances tree height

**Connected**: Check if two elements are in the same set
- Equivalent to checking if find(x) == find(y)

### Key Optimizations

**Path Compression**: During find operations, make nodes point directly to root
**Union by Rank**: Attach smaller tree under root of larger tree
**Union by Size**: Attach tree with fewer nodes under tree with more nodes

### Data Structure Properties
- Near O(1) amortized time for both union and find operations
- Space complexity: O(n) where n is number of elements
- Supports only union operations, not efficient for splits

## Essential Implementation Templates

### Basic Union Find Implementation
```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))  # Each element is its own parent initially
        self.rank = [0] * n           # Track tree heights for union by rank
        self.size = [1] * n           # Track set sizes
        self.components = n           # Number of disjoint components
    
    def find(self, x):
        """Find root with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """Union two sets with union by rank"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x, y):
        """Check if two elements are in same set"""
        return self.find(x) == self.find(y)
    
    def get_size(self, x):
        """Get size of set containing element x"""
        return self.size[self.find(x)]
    
    def get_components(self):
        """Get number of disjoint components"""
        return self.components
```

### Union Find with Custom Elements
```python
class UnionFindDict:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.size = {}
    
    def make_set(self, x):
        """Create a new set with single element"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.size[x] = 1
    
    def find(self, x):
        """Find with path compression"""
        if x not in self.parent:
            self.make_set(x)
        
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union with rank optimization"""
        self.make_set(x)
        self.make_set(y)
        
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
            self.rank[root_x] += 1
        
        return True
```

## Problem Categories

### 1. **Connectivity Problems**
- Network connectivity analysis
- Social network friend circles
- Component counting in graphs
- Path existence queries

### 2. **Grouping and Clustering**
- Similar element grouping
- Region segmentation
- Cluster formation
- Classification problems

### 3. **Dynamic Graph Problems**
- Adding edges dynamically
- Minimum spanning tree construction
- Cycle detection in graph building
- Component merging

### 4. **Grid-Based Problems**
- Island counting variations
- Percolation analysis
- Region growing algorithms
- Connected area calculations

### 5. **Game and Simulation**
- Winning condition detection
- Territory control
- Resource allocation
- State transitions

## Real-World Applications

### 1. **Image Segmentation through Region Agglomeration**

**Business Problem**: Divide a digital image into regions of similar colors for computer vision applications, medical image analysis, or automated image editing.

**Technical Challenge**: Traditional pixel-by-pixel comparison is inefficient. Need to group adjacent pixels with similar properties and merge regions dynamically as similarity criteria evolve.

**Union Find Solution**:
```python
class ImageSegmentation:
    def __init__(self, image_width, image_height):
        self.width = image_width
        self.height = image_height
        self.total_pixels = image_width * image_height
        self.uf = UnionFind(self.total_pixels)
        self.pixel_colors = {}
        self.similarity_threshold = 30  # Color difference threshold
    
    def pixel_to_index(self, row, col):
        """Convert 2D pixel coordinates to 1D index"""
        return row * self.width + col
    
    def index_to_pixel(self, index):
        """Convert 1D index to 2D pixel coordinates"""
        return index // self.width, index % self.width
    
    def add_pixel(self, row, col, color):
        """Add pixel with color information"""
        index = self.pixel_to_index(row, col)
        self.pixel_colors[index] = color
    
    def color_similarity(self, color1, color2):
        """Calculate color similarity (RGB distance)"""
        r_diff = abs(color1[0] - color2[0])
        g_diff = abs(color1[1] - color2[1])
        b_diff = abs(color1[2] - color2[2])
        return (r_diff + g_diff + b_diff) / 3
    
    def segment_image(self):
        """Perform region agglomeration using Union Find"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connectivity
        
        for row in range(self.height):
            for col in range(self.width):
                current_index = self.pixel_to_index(row, col)
                current_color = self.pixel_colors.get(current_index)
                
                if current_color is None:
                    continue
                
                # Check adjacent pixels
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    
                    if (0 <= new_row < self.height and 0 <= new_col < self.width):
                        neighbor_index = self.pixel_to_index(new_row, new_col)
                        neighbor_color = self.pixel_colors.get(neighbor_index)
                        
                        if neighbor_color is not None:
                            # Check color similarity
                            similarity = self.color_similarity(current_color, neighbor_color)
                            
                            if similarity <= self.similarity_threshold:
                                self.uf.union(current_index, neighbor_index)
    
    def get_regions(self):
        """Get all distinct regions after segmentation"""
        regions = {}
        
        for pixel_index in self.pixel_colors:
            root = self.uf.find(pixel_index)
            if root not in regions:
                regions[root] = []
            regions[root].append(self.index_to_pixel(pixel_index))
        
        return regions
    
    def get_region_info(self):
        """Get detailed information about each region"""
        regions = self.get_regions()
        region_info = []
        
        for root, pixels in regions.items():
            avg_color = self.calculate_average_color(pixels)
            region_info.append({
                'region_id': root,
                'pixel_count': len(pixels),
                'average_color': avg_color,
                'pixels': pixels
            })
        
        return region_info
    
    def calculate_average_color(self, pixels):
        """Calculate average color of pixels in region"""
        if not pixels:
            return (0, 0, 0)
        
        total_r = total_g = total_b = 0
        for row, col in pixels:
            pixel_index = self.pixel_to_index(row, col)
            color = self.pixel_colors[pixel_index]
            total_r += color[0]
            total_g += color[1]
            total_b += color[2]
        
        count = len(pixels)
        return (total_r // count, total_g // count, total_b // count)
```

**Business Impact**: Enables automated image analysis, improves medical diagnosis accuracy, facilitates content-based image retrieval, and supports advanced computer vision applications.

### 2. **Network Connectivity and Infrastructure Management**

**Business Problem**: In telecommunications or computer networks, determine connectivity between devices, identify network partitions, and plan infrastructure expansions.

**Technical Challenge**: Networks are dynamic with devices joining/leaving frequently. Need efficient way to track connectivity and identify isolated components for troubleshooting and optimization.

**Union Find Solution**:
```python
class NetworkConnectivityManager:
    def __init__(self):
        self.uf = UnionFindDict()
        self.device_connections = {}
        self.connection_history = []
        self.network_metrics = {}
    
    def add_device(self, device_id, device_info=None):
        """Add new device to network"""
        self.uf.make_set(device_id)
        self.device_connections[device_id] = set()
        if device_info:
            self.network_metrics[device_id] = device_info
    
    def connect_devices(self, device1, device2, connection_quality=1.0):
        """Establish connection between two devices"""
        # Add devices if they don't exist
        self.add_device(device1)
        self.add_device(device2)
        
        # Record connection
        self.device_connections[device1].add(device2)
        self.device_connections[device2].add(device1)
        
        # Union in connectivity structure
        self.uf.union(device1, device2)
        
        # Log connection
        self.connection_history.append({
            'device1': device1,
            'device2': device2,
            'quality': connection_quality,
            'timestamp': time.time()
        })
        
        return True
    
    def are_devices_connected(self, device1, device2):
        """Check if two devices can communicate"""
        return self.uf.find(device1) == self.uf.find(device2)
    
    def get_network_components(self):
        """Get all isolated network components"""
        components = {}
        
        for device in self.device_connections:
            root = self.uf.find(device)
            if root not in components:
                components[root] = []
            components[root].append(device)
        
        return components
    
    def find_isolated_devices(self):
        """Find devices with no connections"""
        isolated = []
        for device, connections in self.device_connections.items():
            if len(connections) == 0:
                isolated.append(device)
        return isolated
    
    def get_largest_network_component(self):
        """Find the largest connected component"""
        components = self.get_network_components()
        if not components:
            return []
        
        largest_component = max(components.values(), key=len)
        return largest_component
    
    def network_resilience_analysis(self):
        """Analyze network resilience and connectivity"""
        components = self.get_network_components()
        total_devices = len(self.device_connections)
        
        analysis = {
            'total_devices': total_devices,
            'connected_components': len(components),
            'largest_component_size': len(self.get_largest_network_component()),
            'connectivity_ratio': len(self.get_largest_network_component()) / total_devices if total_devices > 0 else 0,
            'isolated_devices': len(self.find_isolated_devices()),
            'component_distribution': [len(comp) for comp in components.values()]
        }
        
        return analysis
    
    def suggest_connections_for_resilience(self, target_components=1):
        """Suggest connections to improve network resilience"""
        components = self.get_network_components()
        suggestions = []
        
        if len(components) <= target_components:
            return suggestions
        
        # Sort components by size (largest first)
        sorted_components = sorted(components.values(), key=len, reverse=True)
        
        # Suggest connecting smaller components to largest
        largest_component = sorted_components[0]
        
        for i in range(1, len(sorted_components)):
            if len(suggestions) >= len(components) - target_components:
                break
            
            smaller_component = sorted_components[i]
            # Suggest connecting representative devices
            suggestions.append({
                'connect': (largest_component[0], smaller_component[0]),
                'reason': f'Merge component of size {len(smaller_component)} with largest component',
                'impact': f'Reduces isolated components from {len(components)} to {len(components) - 1}'
            })
        
        return suggestions
```

**Business Impact**: Improves network reliability, reduces downtime, optimizes infrastructure investment, and enables proactive network management and troubleshooting.

### 3. **Percolation Analysis and Material Science**

**Business Problem**: Identify the percolation threshold of a liquid through a filter, analyze material conductivity, or study phase transitions in physical systems.

**Technical Challenge**: Determine if there's a connected path from one side of a grid to another through open sites. Critical for understanding material properties and system behavior at scale.

**Union Find Solution**:
```python
class PercolationAnalyzer:
    def __init__(self, grid_size):
        self.n = grid_size
        self.grid = [[False] * self.n for _ in range(self.n)]  # False = blocked, True = open
        
        # Union Find with virtual top and bottom nodes
        # Nodes: 0 to n*n-1 are grid positions, n*n is virtual top, n*n+1 is virtual bottom
        self.uf = UnionFind(self.n * self.n + 2)
        self.virtual_top = self.n * self.n
        self.virtual_bottom = self.n * self.n + 1
        self.open_sites = 0
    
    def position_to_index(self, row, col):
        """Convert 2D grid position to 1D index"""
        return row * self.n + col
    
    def is_valid_position(self, row, col):
        """Check if position is within grid bounds"""
        return 0 <= row < self.n and 0 <= col < self.n
    
    def open_site(self, row, col):
        """Open a site and connect to adjacent open sites"""
        if not self.is_valid_position(row, col) or self.grid[row][col]:
            return  # Already open or invalid position
        
        self.grid[row][col] = True
        self.open_sites += 1
        current_index = self.position_to_index(row, col)
        
        # Connect to virtual top if in top row
        if row == 0:
            self.uf.union(current_index, self.virtual_top)
        
        # Connect to virtual bottom if in bottom row
        if row == self.n - 1:
            self.uf.union(current_index, self.virtual_bottom)
        
        # Connect to adjacent open sites
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (self.is_valid_position(new_row, new_col) and 
                self.grid[new_row][new_col]):
                neighbor_index = self.position_to_index(new_row, new_col)
                self.uf.union(current_index, neighbor_index)
    
    def percolates(self):
        """Check if system percolates (top connected to bottom)"""
        return self.uf.connected(self.virtual_top, self.virtual_bottom)
    
    def is_site_open(self, row, col):
        """Check if site is open"""
        if not self.is_valid_position(row, col):
            return False
        return self.grid[row][col]
    
    def is_site_full(self, row, col):
        """Check if site is connected to top (can be filled with liquid)"""
        if not self.is_site_open(row, col):
            return False
        
        site_index = self.position_to_index(row, col)
        return self.uf.connected(site_index, self.virtual_top)
    
    def get_percolation_statistics(self):
        """Get detailed percolation statistics"""
        total_sites = self.n * self.n
        open_fraction = self.open_sites / total_sites
        
        # Count full sites (connected to top)
        full_sites = 0
        for row in range(self.n):
            for col in range(self.n):
                if self.is_site_full(row, col):
                    full_sites += 1
        
        # Analyze connected components
        components = {}
        for row in range(self.n):
            for col in range(self.n):
                if self.is_site_open(row, col):
                    site_index = self.position_to_index(row, col)
                    root = self.uf.find(site_index)
                    if root not in components:
                        components[root] = 0
                    components[root] += 1
        
        return {
            'total_sites': total_sites,
            'open_sites': self.open_sites,
            'open_fraction': open_fraction,
            'full_sites': full_sites,
            'percolates': self.percolates(),
            'connected_components': len(components),
            'largest_component_size': max(components.values()) if components else 0
        }
    
    def find_percolation_threshold(self, trials=1000):
        """Monte Carlo simulation to find percolation threshold"""
        threshold_sum = 0
        
        for trial in range(trials):
            # Reset grid
            self.__init__(self.n)
            sites = [(i, j) for i in range(self.n) for j in range(self.n)]
            random.shuffle(sites)
            
            # Open sites until percolation occurs
            for row, col in sites:
                self.open_site(row, col)
                if self.percolates():
                    threshold_sum += self.open_sites / (self.n * self.n)
                    break
        
        return threshold_sum / trials
```

**Business Impact**: Enables material design optimization, improves manufacturing processes, supports quality control in porous materials, and advances research in phase transitions and critical phenomena.

### 4. **Social Network Analysis and Community Detection**

**Business Problem**: Analyze social networks to identify communities, friend circles, and influence patterns for targeted marketing, content recommendation, and network growth strategies.

**Union Find Solution**:
```python
class SocialNetworkAnalyzer:
    def __init__(self):
        self.uf = UnionFindDict()
        self.friendships = {}
        self.user_profiles = {}
        self.interaction_strength = {}
    
    def add_user(self, user_id, profile=None):
        """Add user to social network"""
        self.uf.make_set(user_id)
        self.friendships[user_id] = set()
        if profile:
            self.user_profiles[user_id] = profile
    
    def add_friendship(self, user1, user2, strength=1.0):
        """Create friendship connection between users"""
        self.add_user(user1)
        self.add_user(user2)
        
        # Add bidirectional friendship
        self.friendships[user1].add(user2)
        self.friendships[user2].add(user1)
        
        # Track interaction strength
        self.interaction_strength[(user1, user2)] = strength
        self.interaction_strength[(user2, user1)] = strength
        
        # Union in same community if strong connection
        if strength >= 0.5:  # Threshold for community membership
            self.uf.union(user1, user2)
    
    def get_communities(self):
        """Get all distinct communities"""
        communities = {}
        
        for user in self.friendships:
            root = self.uf.find(user)
            if root not in communities:
                communities[root] = []
            communities[root].append(user)
        
        return communities
    
    def analyze_community_structure(self):
        """Analyze community structure and properties"""
        communities = self.get_communities()
        
        analysis = {
            'total_users': len(self.friendships),
            'total_communities': len(communities),
            'largest_community_size': max(len(comm) for comm in communities.values()) if communities else 0,
            'smallest_community_size': min(len(comm) for comm in communities.values()) if communities else 0,
            'average_community_size': sum(len(comm) for comm in communities.values()) / len(communities) if communities else 0,
            'community_size_distribution': sorted([len(comm) for comm in communities.values()], reverse=True)
        }
        
        return analysis
    
    def find_community_influencers(self):
        """Find most connected users in each community"""
        communities = self.get_communities()
        influencers = {}
        
        for root, members in communities.items():
            if len(members) < 2:
                continue
            
            # Calculate connection count for each member
            member_connections = {}
            for user in members:
                # Count connections within community
                internal_connections = len([friend for friend in self.friendships[user] if friend in members])
                member_connections[user] = internal_connections
            
            # Find user with most connections
            top_influencer = max(member_connections, key=member_connections.get)
            influencers[root] = {
                'user_id': top_influencer,
                'internal_connections': member_connections[top_influencer],
                'community_size': len(members)
            }
        
        return influencers
    
    def suggest_friend_recommendations(self, user_id, max_recommendations=5):
        """Suggest friends based on community membership and mutual connections"""
        if user_id not in self.friendships:
            return []
        
        user_community_root = self.uf.find(user_id)
        communities = self.get_communities()
        user_community = communities.get(user_community_root, [])
        
        recommendations = []
        current_friends = self.friendships[user_id]
        
        # Recommend users from same community who aren't already friends
        for potential_friend in user_community:
            if (potential_friend != user_id and 
                potential_friend not in current_friends):
                
                # Calculate mutual friends
                mutual_friends = len(current_friends.intersection(self.friendships[potential_friend]))
                
                recommendations.append({
                    'user_id': potential_friend,
                    'reason': 'same_community',
                    'mutual_friends': mutual_friends,
                    'community_size': len(user_community)
                })
        
        # Sort by mutual friends and limit results
        recommendations.sort(key=lambda x: x['mutual_friends'], reverse=True)
        return recommendations[:max_recommendations]
```

**Business Impact**: Improves user engagement, enhances content targeting, facilitates viral marketing campaigns, and provides insights for product development and user experience optimization.

## Advanced Techniques

### 1. **Weighted Union Find**
Track additional properties like weights or distances during union operations for specialized applications.

### 2. **Rollback Union Find**
Support undo operations by maintaining operation history, useful for what-if scenarios and backtracking algorithms.

### 3. **Persistent Union Find**
Maintain multiple versions of the data structure for temporal queries and version control applications.

### 4. **Union Find with Custom Equivalence**
Implement custom equality functions for complex objects or multi-dimensional data.

### 5. **Parallel Union Find**
Implement concurrent versions for multi-threaded applications and distributed systems.

## Performance Characteristics

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Find | O(α(n)) | O(1) | α is inverse Ackermann function |
| Union | O(α(n)) | O(1) | Amortized with path compression |
| Connected | O(α(n)) | O(1) | Two find operations |
| Initialize | O(n) | O(n) | Create n disjoint sets |
| Space Total | - | O(n) | Parent and rank arrays |

## Practical Problem Examples

### Beginner Level
1. **Number of Islands** - Basic connected components
2. **Friend Circles** - Social network connectivity
3. **Union Find Basic Operations** - Fundamental understanding
4. **Redundant Connection** - Cycle detection

### Intermediate Level
5. **Accounts Merge** - String-based grouping
6. **Most Stones Removed** - Optimization with Union Find
7. **Satisfiability of Equality Equations** - Constraint satisfaction
8. **Number of Operations to Make Network Connected** - Minimum connections
9. **Smallest String with Swaps** - Permutation groups

### Advanced Level
10. **Optimize Water Distribution** - Minimum spanning tree concepts
11. **Checking Existence of Edge Length Limited Paths** - Offline query processing
12. **Remove Max Number of Edges to Keep Graph Fully Traversable** - Dual Union Find
13. **Minimize Malware Spread** - Component analysis with optimization
14. **Rank Transform of Matrix** - Multi-dimensional Union Find

## Common Pitfalls and Solutions

### 1. **Forgetting Path Compression**
- **Problem**: Poor performance without optimization
- **Solution**: Always implement path compression in find operation

### 2. **Incorrect Union Logic**
- **Problem**: Not using union by rank/size leads to unbalanced trees
- **Solution**: Implement proper union optimization

### 3. **Index Mapping Errors**
- **Problem**: Incorrect mapping between problem elements and Union Find indices
- **Solution**: Careful design of mapping functions and validation

### 4. **Memory Management**
- **Problem**: Not handling dynamic element addition properly
- **Solution**: Use dictionary-based Union Find for dynamic elements

### 5. **Connectivity vs Path Queries**
- **Problem**: Using Union Find for shortest path problems
- **Solution**: Union Find only answers connectivity, use BFS/Dijkstra for paths

## When NOT to Use Union Find

1. **Need shortest paths**: Union Find only determines connectivity, not distances
2. **Frequent disconnections**: Union Find doesn't efficiently support edge removal
3. **Complex graph queries**: Advanced graph algorithms need different approaches
4. **Small static problems**: Simple array operations might be more straightforward
5. **Need intermediate nodes**: When path details matter, not just connectivity

## Tips for Success

1. **Choose right optimization**: Path compression and union by rank are essential
2. **Plan element mapping**: Design clear mapping between problem elements and indices
3. **Handle edge cases**: Empty sets, single elements, invalid operations
4. **Consider problem constraints**: Static vs dynamic element addition
5. **Test connectivity thoroughly**: Verify union and find operations work correctly
6. **Use appropriate data structures**: Arrays for fixed elements, dictionaries for dynamic
7. **Optimize for problem specifics**: Custom implementations for special requirements

## Conclusion

The Union Find pattern is essential for:
- Connectivity and reachability problems
- Dynamic set operations and grouping
- Graph component analysis
- Real-time clustering and classification
- Network and infrastructure analysis
- Game and simulation state management

Master this pattern by understanding when connectivity matters more than paths, implementing proper optimizations, and recognizing problems that involve dynamic grouping or set operations. The key insight is that many problems can be solved efficiently by tracking which elements belong to the same group, without needing the complexity of full graph algorithms.
