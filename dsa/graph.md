# Graph Algorithms Pattern

## Pattern Overview

**Graph Algorithms** encompass a collection of specialized techniques for solving optimization and analysis problems on graph data structures. While basic traversals (DFS/BFS) help find paths and connected components, graph algorithms address specific objectives like shortest paths, minimum spanning trees, topological ordering, and network flow optimization. These algorithms adapt traversal techniques to solve particular optimization problems associated with graphs.

## When to Use This Pattern

Your problem matches this pattern if the following conditions are fulfilled:

### ✅ Use Graph Algorithms When:

1. **Relationships between elements**: There is a network of interconnected objects with some relationship between them; that is, the data can be represented as a graph.

### Additional Indicators:
- Need shortest path between nodes (not just any path)
- Require minimum spanning tree (not just connectivity)
- Solving optimization problems on networks
- Analyzing flow, capacity, or cost in networks
- Finding optimal ordering with dependencies
- Detecting cycles or strongly connected components

### ❌ Don't Use Graph Algorithms When:

1. **Simple connectivity checks**: Basic traversal algorithms suffice
2. **Linear data structures**: Arrays, linked lists don't need graph algorithms
3. **Tree-specific problems**: Tree algorithms are more appropriate
4. **No optimization needed**: When any solution works, not necessarily optimal

## Core Graph Algorithm Categories

### Shortest Path Algorithms

**Dijkstra's Algorithm**: Finds shortest path in weighted graphs with non-negative edges
- Use case: GPS navigation, network routing
- Time Complexity: O((V + E) log V) with priority queue

**Bellman-Ford Algorithm**: Handles negative edge weights, detects negative cycles
- Use case: Currency arbitrage, network protocols
- Time Complexity: O(VE)

**Floyd-Warshall Algorithm**: All-pairs shortest paths
- Use case: Network analysis, transitive closure
- Time Complexity: O(V³)

### Minimum Spanning Tree

**Prim's Algorithm**: Grows MST from starting vertex
- Use case: Network design, clustering
- Time Complexity: O((V + E) log V)

**Kruskal's Algorithm**: Sorts edges and uses Union-Find
- Use case: Network optimization, clustering
- Time Complexity: O(E log E)

### Topological Ordering

**Topological Sort**: Orders vertices in DAG respecting dependencies
- Use case: Task scheduling, dependency resolution
- Time Complexity: O(V + E)

### Network Flow

**Max Flow Algorithms**: Find maximum flow through network
- Use case: Resource allocation, matching problems
- Various implementations with different complexities

## Essential Implementation Templates

### Dijkstra's Algorithm
```python
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    """Find shortest paths from start to all vertices"""
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    previous = {}
    
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        if current_distance > distances[current]:
            continue
        
        for neighbor, weight in graph[current]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    return distances, previous

def reconstruct_path(previous, start, end):
    """Reconstruct shortest path from start to end"""
    path = []
    current = end
    
    while current is not None:
        path.append(current)
        current = previous.get(current)
    
    path.reverse()
    return path if path[0] == start else []
```

### Bellman-Ford Algorithm
```python
def bellman_ford(graph, start):
    """Find shortest paths with negative edge detection"""
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    
    # Relax edges V-1 times
    for _ in range(len(graph) - 1):
        for vertex in graph:
            for neighbor, weight in graph[vertex]:
                if distances[vertex] + weight < distances[neighbor]:
                    distances[neighbor] = distances[vertex] + weight
    
    # Check for negative cycles
    for vertex in graph:
        for neighbor, weight in graph[vertex]:
            if distances[vertex] + weight < distances[neighbor]:
                return None, True  # Negative cycle detected
    
    return distances, False
```

### Prim's MST Algorithm
```python
def prim_mst(graph):
    """Find Minimum Spanning Tree using Prim's algorithm"""
    if not graph:
        return []
    
    start = next(iter(graph))
    visited = {start}
    mst_edges = []
    pq = []
    
    # Add all edges from start vertex
    for neighbor, weight in graph[start]:
        heapq.heappush(pq, (weight, start, neighbor))
    
    while pq and len(visited) < len(graph):
        weight, u, v = heapq.heappop(pq)
        
        if v in visited:
            continue
        
        # Add edge to MST
        mst_edges.append((u, v, weight))
        visited.add(v)
        
        # Add new edges from v
        for neighbor, edge_weight in graph[v]:
            if neighbor not in visited:
                heapq.heappush(pq, (edge_weight, v, neighbor))
    
    return mst_edges
```

### Topological Sort
```python
def topological_sort(graph):
    """Topological ordering using DFS"""
    visited = set()
    stack = []
    
    def dfs(vertex):
        visited.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(vertex)
    
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)
    
    return stack[::-1]  # Reverse to get topological order
```

## Problem Categories

### 1. **Shortest Path Problems**
- Single-source shortest path
- All-pairs shortest path
- Shortest path with constraints
- K-shortest paths

### 2. **Minimum Spanning Tree**
- Network design optimization
- Clustering and segmentation
- Infrastructure cost minimization
- Connection optimization

### 3. **Flow and Matching**
- Maximum flow problems
- Minimum cut problems
- Bipartite matching
- Network capacity planning

### 4. **Cycle Detection**
- Dependency cycle detection
- Deadlock detection
- Circuit analysis
- Scheduling conflict identification

### 5. **Connectivity Analysis**
- Strongly connected components
- Bridge and articulation point detection
- Network resilience analysis
- Component decomposition

## Real-World Applications

### 1. **Routing in Computer Networks**

**Business Problem**: Ensure data travels quickly from computers to servers across the internet without getting stuck or slowed down, while optimizing for cost, latency, and reliability.

**Technical Challenge**: Networks are dynamic with changing conditions, failures, and varying loads. Traditional static routing can't adapt to real-time conditions and may cause congestion or inefficient paths.

**Graph Algorithm Solution**:
```python
class NetworkRouter:
    def __init__(self):
        self.network_topology = defaultdict(list)  # (neighbor, latency, bandwidth, cost)
        self.routing_table = {}
        self.network_metrics = {}
    
    def add_network_link(self, node1, node2, latency, bandwidth, cost):
        """Add bidirectional network link with metrics"""
        self.network_topology[node1].append((node2, latency, bandwidth, cost))
        self.network_topology[node2].append((node1, latency, bandwidth, cost))
    
    def update_link_metrics(self, node1, node2, new_latency, new_bandwidth):
        """Update link metrics for dynamic conditions"""
        # Update in both directions
        for i, (neighbor, _, bandwidth, cost) in enumerate(self.network_topology[node1]):
            if neighbor == node2:
                self.network_topology[node1][i] = (neighbor, new_latency, new_bandwidth, cost)
                break
        
        for i, (neighbor, _, bandwidth, cost) in enumerate(self.network_topology[node2]):
            if neighbor == node1:
                self.network_topology[node2][i] = (neighbor, new_latency, new_bandwidth, cost)
                break
    
    def find_optimal_path(self, source, destination, optimization_criteria='latency'):
        """Find optimal path based on different criteria"""
        if optimization_criteria == 'latency':
            return self.dijkstra_latency_optimized(source, destination)
        elif optimization_criteria == 'cost':
            return self.dijkstra_cost_optimized(source, destination)
        elif optimization_criteria == 'bandwidth':
            return self.max_bandwidth_path(source, destination)
        else:
            return self.multi_criteria_optimization(source, destination)
    
    def dijkstra_latency_optimized(self, source, destination):
        """Optimize for minimum latency"""
        distances = {node: float('inf') for node in self.network_topology}
        distances[source] = 0
        previous = {}
        pq = [(0, source)]
        
        while pq:
            current_latency, current_node = heapq.heappop(pq)
            
            if current_node == destination:
                break
            
            if current_latency > distances[current_node]:
                continue
            
            for neighbor, latency, bandwidth, cost in self.network_topology[current_node]:
                # Consider bandwidth capacity constraint
                if bandwidth > 0:  # Link has available bandwidth
                    new_latency = current_latency + latency
                    
                    if new_latency < distances[neighbor]:
                        distances[neighbor] = new_latency
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (new_latency, neighbor))
        
        # Reconstruct path
        path = self.reconstruct_path(previous, source, destination)
        return {
            'path': path,
            'total_latency': distances[destination],
            'optimization': 'latency'
        }
    
    def max_bandwidth_path(self, source, destination):
        """Find path with maximum minimum bandwidth (bottleneck)"""
        # Use modified Dijkstra where we maximize minimum bandwidth
        bandwidths = {node: 0 for node in self.network_topology}
        bandwidths[source] = float('inf')
        previous = {}
        pq = [(-float('inf'), source)]  # Negative for max-heap behavior
        
        while pq:
            current_bandwidth, current_node = heapq.heappop(pq)
            current_bandwidth = -current_bandwidth
            
            if current_node == destination:
                break
            
            if current_bandwidth < bandwidths[current_node]:
                continue
            
            for neighbor, latency, bandwidth, cost in self.network_topology[current_node]:
                # Minimum bandwidth along path
                path_bandwidth = min(current_bandwidth, bandwidth)
                
                if path_bandwidth > bandwidths[neighbor]:
                    bandwidths[neighbor] = path_bandwidth
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (-path_bandwidth, neighbor))
        
        path = self.reconstruct_path(previous, source, destination)
        return {
            'path': path,
            'bottleneck_bandwidth': bandwidths[destination],
            'optimization': 'bandwidth'
        }
    
    def multi_criteria_optimization(self, source, destination, 
                                 latency_weight=0.4, cost_weight=0.3, bandwidth_weight=0.3):
        """Multi-criteria optimization combining latency, cost, and bandwidth"""
        scores = {node: float('inf') for node in self.network_topology}
        scores[source] = 0
        previous = {}
        pq = [(0, source)]
        
        while pq:
            current_score, current_node = heapq.heappop(pq)
            
            if current_node == destination:
                break
            
            if current_score > scores[current_node]:
                continue
            
            for neighbor, latency, bandwidth, cost in self.network_topology[current_node]:
                if bandwidth > 0:
                    # Normalize and combine metrics
                    normalized_latency = latency / 100.0  # Assume max latency 100ms
                    normalized_cost = cost / 10.0        # Assume max cost 10
                    normalized_bandwidth = 1.0 - (bandwidth / 1000.0)  # Higher bandwidth = lower score
                    
                    link_score = (latency_weight * normalized_latency + 
                                cost_weight * normalized_cost + 
                                bandwidth_weight * normalized_bandwidth)
                    
                    new_score = current_score + link_score
                    
                    if new_score < scores[neighbor]:
                        scores[neighbor] = new_score
                        previous[neighbor] = current_node
                        heapq.heappush(pq, (new_score, neighbor))
        
        path = self.reconstruct_path(previous, source, destination)
        return {
            'path': path,
            'composite_score': scores[destination],
            'optimization': 'multi_criteria'
        }
    
    def find_alternative_paths(self, source, destination, k=3):
        """Find k alternative paths for redundancy"""
        paths = []
        
        for attempt in range(k):
            # Find path with current network state
            result = self.find_optimal_path(source, destination)
            
            if result['path']:
                paths.append(result)
                
                # Temporarily remove edges in found path to find alternatives
                removed_edges = []
                path = result['path']
                
                for i in range(len(path) - 1):
                    current, next_node = path[i], path[i + 1]
                    
                    # Remove edge temporarily
                    for j, (neighbor, latency, bandwidth, cost) in enumerate(self.network_topology[current]):
                        if neighbor == next_node:
                            removed_edges.append((current, j, (neighbor, latency, bandwidth, cost)))
                            self.network_topology[current].pop(j)
                            break
                
                # Restore edges for next iteration
                for node, index, edge_data in removed_edges:
                    self.network_topology[node].insert(index, edge_data)
        
        return paths
    
    def reconstruct_path(self, previous, source, destination):
        """Reconstruct path from previous node mapping"""
        if destination not in previous and destination != source:
            return []
        
        path = []
        current = destination
        
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        return path if path[0] == source else []
```

**Business Impact**: Reduces network latency, improves data transmission reliability, optimizes infrastructure costs, and enables scalable network management for millions of users.

### 2. **Flight Route Optimization**

**Business Problem**: Airlines need to optimize flight routes to reduce fuel consumption, minimize flight time, maximize passenger capacity utilization, and ensure schedule reliability across global networks.

**Graph Algorithm Solution**:
```python
class FlightRouteOptimizer:
    def __init__(self):
        self.airport_network = defaultdict(list)  # (destination, distance, cost, duration, capacity)
        self.airport_info = {}  # Airport details and constraints
        self.weather_conditions = {}
        self.fuel_prices = {}
    
    def add_route(self, origin, destination, distance, base_cost, duration, aircraft_capacity):
        """Add flight route between airports"""
        self.airport_network[origin].append({
            'destination': destination,
            'distance': distance,
            'base_cost': base_cost,
            'duration': duration,
            'capacity': aircraft_capacity,
            'available_slots': aircraft_capacity
        })
    
    def update_dynamic_conditions(self, airport, weather_delay=0, fuel_price_multiplier=1.0):
        """Update real-time conditions affecting routes"""
        self.weather_conditions[airport] = weather_delay
        self.fuel_prices[airport] = fuel_price_multiplier
    
    def find_optimal_route(self, origin, destination, passengers, optimization='cost'):
        """Find optimal route based on different criteria"""
        if optimization == 'cost':
            return self.cost_optimized_route(origin, destination, passengers)
        elif optimization == 'time':
            return self.time_optimized_route(origin, destination, passengers)
        elif optimization == 'fuel':
            return self.fuel_optimized_route(origin, destination, passengers)
        else:
            return self.multi_objective_route(origin, destination, passengers)
    
    def cost_optimized_route(self, origin, destination, passengers):
        """Find minimum cost route considering capacity constraints"""
        costs = {airport: float('inf') for airport in self.airport_network}
        costs[origin] = 0
        previous = {}
        route_details = {}
        pq = [(0, origin, passengers)]
        
        while pq:
            current_cost, current_airport, remaining_passengers = heapq.heappop(pq)
            
            if current_airport == destination and remaining_passengers <= 0:
                break
            
            if current_cost > costs[current_airport]:
                continue
            
            for route in self.airport_network[current_airport]:
                dest_airport = route['destination']
                
                # Calculate dynamic cost including fuel prices and weather delays
                weather_delay = self.weather_conditions.get(current_airport, 0)
                fuel_multiplier = self.fuel_prices.get(current_airport, 1.0)
                
                # Base cost adjusted for conditions
                adjusted_cost = route['base_cost'] * fuel_multiplier
                delay_cost = weather_delay * 50  # Cost per minute delay
                total_cost = current_cost + adjusted_cost + delay_cost
                
                # Check capacity constraints
                passengers_this_flight = min(remaining_passengers, route['available_slots'])
                new_remaining = max(0, remaining_passengers - passengers_this_flight)
                
                if total_cost < costs[dest_airport]:
                    costs[dest_airport] = total_cost
                    previous[dest_airport] = current_airport
                    route_details[dest_airport] = {
                        'route_info': route,
                        'passengers_carried': passengers_this_flight,
                        'weather_delay': weather_delay,
                        'fuel_multiplier': fuel_multiplier
                    }
                    
                    heapq.heappush(pq, (total_cost, dest_airport, new_remaining))
        
        # Reconstruct route
        path = self.reconstruct_flight_path(previous, origin, destination)
        return {
            'path': path,
            'total_cost': costs[destination],
            'route_details': route_details,
            'optimization': 'cost'
        }
    
    def multi_stop_optimization(self, stops, start_airport):
        """Solve traveling salesman problem for multi-stop flights"""
        # Use approximation algorithm for practical solution
        unvisited = set(stops)
        current = start_airport
        total_cost = 0
        route = [current]
        
        while unvisited:
            # Find nearest unvisited airport
            min_cost = float('inf')
            next_airport = None
            
            for airport in unvisited:
                result = self.cost_optimized_route(current, airport, passengers=100)
                if result['total_cost'] < min_cost:
                    min_cost = result['total_cost']
                    next_airport = airport
            
            if next_airport:
                route.append(next_airport)
                total_cost += min_cost
                unvisited.remove(next_airport)
                current = next_airport
        
        return {
            'route': route,
            'total_cost': total_cost,
            'optimization': 'multi_stop'
        }
    
    def find_hub_optimization(self, airports, passenger_flows):
        """Find optimal hub locations for airline network"""
        # Calculate centrality scores for potential hubs
        hub_scores = {}
        
        for potential_hub in airports:
            total_distance = 0
            total_connections = 0
            
            for origin in airports:
                if origin != potential_hub:
                    result = self.cost_optimized_route(origin, potential_hub, passengers=50)
                    if result['path']:
                        total_distance += result['total_cost']
                        total_connections += 1
            
            # Score based on connectivity and cost efficiency
            if total_connections > 0:
                hub_scores[potential_hub] = {
                    'avg_cost': total_distance / total_connections,
                    'connectivity': total_connections,
                    'hub_score': total_connections / (total_distance / total_connections)
                }
        
        # Select top hubs
        sorted_hubs = sorted(hub_scores.items(), 
                           key=lambda x: x[1]['hub_score'], reverse=True)
        
        return sorted_hubs[:3]  # Top 3 hub candidates
    
    def reconstruct_flight_path(self, previous, origin, destination):
        """Reconstruct flight path with route details"""
        if destination not in previous and destination != origin:
            return []
        
        path = []
        current = destination
        
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        return path if path[0] == origin else []
```

**Business Impact**: Reduces operational costs by 15-25%, improves fuel efficiency, enhances passenger satisfaction through optimized schedules, and enables strategic network planning for airline profitability.

### 3. **Epidemic Spread Modeling**

**Business Problem**: Predict and model how infectious diseases spread through populations to inform public health decisions, resource allocation, and intervention strategies.

**Graph Algorithm Solution**:
```python
class EpidemicSpreadModel:
    def __init__(self):
        self.population_graph = defaultdict(list)  # (contact, interaction_strength, frequency)
        self.individual_status = {}  # 'susceptible', 'infected', 'recovered'
        self.transmission_rates = {}
        self.recovery_rates = {}
        self.simulation_history = []
    
    def add_individual(self, person_id, initial_status='susceptible', 
                      transmission_rate=0.1, recovery_rate=0.05):
        """Add individual to population model"""
        self.individual_status[person_id] = initial_status
        self.transmission_rates[person_id] = transmission_rate
        self.recovery_rates[person_id] = recovery_rate
        self.population_graph[person_id] = []
    
    def add_contact(self, person1, person2, interaction_strength=1.0, frequency=1.0):
        """Add contact relationship between individuals"""
        self.population_graph[person1].append({
            'contact': person2,
            'strength': interaction_strength,
            'frequency': frequency
        })
        self.population_graph[person2].append({
            'contact': person1,
            'strength': interaction_strength,
            'frequency': frequency
        })
    
    def simulate_spread(self, initial_infected, days=100, intervention_day=None):
        """Simulate epidemic spread over time"""
        # Initialize patient zero
        for person in initial_infected:
            self.individual_status[person] = 'infected'
        
        daily_stats = []
        
        for day in range(days):
            # Apply interventions
            if intervention_day and day >= intervention_day:
                self.apply_intervention('social_distancing', effectiveness=0.5)
            
            # Simulate one day of spread
            new_infections = self.simulate_daily_transmission()
            new_recoveries = self.simulate_daily_recovery()
            
            # Record statistics
            stats = self.calculate_daily_statistics(day, new_infections, new_recoveries)
            daily_stats.append(stats)
            
            # Check if epidemic is over
            if stats['active_infected'] == 0:
                break
        
        return daily_stats
    
    def simulate_daily_transmission(self):
        """Simulate disease transmission for one day"""
        new_infections = []
        
        for person in self.individual_status:
            if self.individual_status[person] == 'infected':
                # Check all contacts
                for contact_info in self.population_graph[person]:
                    contact = contact_info['contact']
                    
                    if self.individual_status[contact] == 'susceptible':
                        # Calculate transmission probability
                        base_prob = self.transmission_rates[person]
                        interaction_factor = contact_info['strength'] * contact_info['frequency']
                        transmission_prob = base_prob * interaction_factor
                        
                        # Random transmission event
                        if random.random() < transmission_prob:
                            new_infections.append(contact)
        
        # Apply new infections
        for person in new_infections:
            self.individual_status[person] = 'infected'
        
        return new_infections
    
    def simulate_daily_recovery(self):
        """Simulate recovery for one day"""
        new_recoveries = []
        
        for person in self.individual_status:
            if self.individual_status[person] == 'infected':
                recovery_prob = self.recovery_rates[person]
                
                if random.random() < recovery_prob:
                    new_recoveries.append(person)
        
        # Apply recoveries
        for person in new_recoveries:
            self.individual_status[person] = 'recovered'
        
        return new_recoveries
    
    def identify_super_spreaders(self, threshold=10):
        """Identify individuals with high transmission potential"""
        centrality_scores = {}
        
        for person in self.population_graph:
            # Calculate weighted degree centrality
            total_connection_strength = sum(
                contact['strength'] * contact['frequency'] 
                for contact in self.population_graph[person]
            )
            
            # Factor in transmission rate
            transmission_potential = (total_connection_strength * 
                                    self.transmission_rates[person])
            
            centrality_scores[person] = transmission_potential
        
        # Find super spreaders
        super_spreaders = [
            person for person, score in centrality_scores.items() 
            if score > threshold
        ]
        
        return super_spreaders, centrality_scores
    
    def find_critical_connections(self):
        """Find connections that are critical for disease spread"""
        # Use edge betweenness centrality concept
        critical_edges = []
        
        for person in self.population_graph:
            for contact_info in self.population_graph[person]:
                contact = contact_info['contact']
                edge_importance = (
                    contact_info['strength'] * 
                    contact_info['frequency'] *
                    self.transmission_rates[person] *
                    len(self.population_graph[contact])  # Contact's connectivity
                )
                
                critical_edges.append({
                    'edge': (person, contact),
                    'importance': edge_importance,
                    'contact_info': contact_info
                })
        
        # Sort by importance
        critical_edges.sort(key=lambda x: x['importance'], reverse=True)
        return critical_edges[:20]  # Top 20 critical connections
    
    def optimize_intervention_strategy(self, budget=100, strategies=None):
        """Find optimal intervention strategy given budget constraints"""
        if strategies is None:
            strategies = [
                {'name': 'isolate_individual', 'cost': 5, 'effectiveness': 0.9},
                {'name': 'reduce_contact_strength', 'cost': 2, 'effectiveness': 0.3},
                {'name': 'mass_testing', 'cost': 10, 'effectiveness': 0.6}
            ]
        
        # Identify high-priority targets
        super_spreaders, _ = self.identify_super_spreaders()
        critical_connections = self.find_critical_connections()
        
        intervention_plan = []
        remaining_budget = budget
        
        # Prioritize isolating super spreaders
        for person in super_spreaders:
            if remaining_budget >= strategies[0]['cost']:
                intervention_plan.append({
                    'action': 'isolate',
                    'target': person,
                    'cost': strategies[0]['cost'],
                    'expected_impact': strategies[0]['effectiveness']
                })
                remaining_budget -= strategies[0]['cost']
        
        # Reduce critical connections
        for edge_info in critical_connections:
            if remaining_budget >= strategies[1]['cost']:
                intervention_plan.append({
                    'action': 'reduce_contact',
                    'target': edge_info['edge'],
                    'cost': strategies[1]['cost'],
                    'expected_impact': strategies[1]['effectiveness']
                })
                remaining_budget -= strategies[1]['cost']
        
        return {
            'intervention_plan': intervention_plan,
            'total_cost': budget - remaining_budget,
            'remaining_budget': remaining_budget
        }
    
    def calculate_daily_statistics(self, day, new_infections, new_recoveries):
        """Calculate daily epidemic statistics"""
        susceptible = sum(1 for status in self.individual_status.values() if status == 'susceptible')
        infected = sum(1 for status in self.individual_status.values() if status == 'infected')
        recovered = sum(1 for status in self.individual_status.values() if status == 'recovered')
        
        return {
            'day': day,
            'susceptible': susceptible,
            'active_infected': infected,
            'recovered': recovered,
            'new_infections': len(new_infections),
            'new_recoveries': len(new_recoveries),
            'reproduction_rate': len(new_infections) / max(infected, 1)
        }
```

**Business Impact**: Enables evidence-based public health policy, optimizes resource allocation during outbreaks, reduces epidemic impact through targeted interventions, and saves lives through early prediction and prevention strategies.

### 4. **Recommendation Systems**

**Business Problem**: Provide personalized content recommendations that increase user engagement, retention, and revenue while discovering new content preferences and handling the cold start problem.

**Graph Algorithm Solution**:
```python
class GraphBasedRecommendationSystem:
    def __init__(self):
        self.user_item_graph = defaultdict(list)  # Users connected to items they've interacted with
        self.item_similarity_graph = defaultdict(list)  # Items connected to similar items
        self.user_similarity_graph = defaultdict(list)  # Users connected to similar users
        self.interaction_weights = {}  # (user, item) -> weight
        self.item_features = {}  # Item metadata
        self.user_profiles = {}  # User demographics and preferences
    
    def add_interaction(self, user_id, item_id, interaction_type='view', 
                       rating=None, timestamp=None):
        """Add user-item interaction"""
        # Weight interactions based on type
        weight_map = {'view': 1, 'like': 3, 'share': 5, 'purchase': 10}
        base_weight = weight_map.get(interaction_type, 1)
        
        if rating:
            weight = base_weight * (rating / 5.0)  # Normalize rating
        else:
            weight = base_weight
        
        # Add to user-item graph
        self.user_item_graph[user_id].append({
            'item': item_id,
            'weight': weight,
            'type': interaction_type,
            'timestamp': timestamp
        })
        
        self.interaction_weights[(user_id, item_id)] = weight
    
    def build_item_similarity_graph(self, similarity_threshold=0.3):
        """Build item-item similarity graph based on user interactions"""
        items = set()
        for user_interactions in self.user_item_graph.values():
            for interaction in user_interactions:
                items.add(interaction['item'])
        
        # Calculate item similarities
        for item1 in items:
            for item2 in items:
                if item1 != item2:
                    similarity = self.calculate_item_similarity(item1, item2)
                    
                    if similarity > similarity_threshold:
                        self.item_similarity_graph[item1].append({
                            'item': item2,
                            'similarity': similarity
                        })
    
    def calculate_item_similarity(self, item1, item2):
        """Calculate similarity between two items using cosine similarity"""
        users1 = set()
        users2 = set()
        
        # Find users who interacted with each item
        for user, interactions in self.user_item_graph.items():
            for interaction in interactions:
                if interaction['item'] == item1:
                    users1.add(user)
                elif interaction['item'] == item2:
                    users2.add(user)
        
        # Calculate Jaccard similarity
        intersection = len(users1.intersection(users2))
        union = len(users1.union(users2))
        
        return intersection / union if union > 0 else 0
    
    def recommend_items_collaborative(self, user_id, num_recommendations=10):
        """Collaborative filtering recommendations using graph traversal"""
        if user_id not in self.user_item_graph:
            return self.recommend_popular_items(num_recommendations)
        
        # Get user's interacted items
        user_items = set()
        for interaction in self.user_item_graph[user_id]:
            user_items.add(interaction['item'])
        
        # Find candidate items through graph traversal
        candidate_scores = defaultdict(float)
        
        # Method 1: Item-based collaborative filtering
        for interaction in self.user_item_graph[user_id]:
            item = interaction['item']
            user_rating = interaction['weight']
            
            # Find similar items
            for similar_item_info in self.item_similarity_graph[item]:
                similar_item = similar_item_info['item']
                similarity = similar_item_info['similarity']
                
                if similar_item not in user_items:
                    candidate_scores[similar_item] += user_rating * similarity
        
        # Method 2: User-based collaborative filtering
        similar_users = self.find_similar_users(user_id, top_k=50)
        
        for similar_user, user_similarity in similar_users:
            for interaction in self.user_item_graph[similar_user]:
                item = interaction['item']
                if item not in user_items:
                    candidate_scores[item] += interaction['weight'] * user_similarity
        
        # Sort and return top recommendations
        recommendations = sorted(candidate_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        
        return recommendations[:num_recommendations]
    
    def find_similar_users(self, target_user, top_k=20):
        """Find users similar to target user"""
        target_items = {}
        for interaction in self.user_item_graph[target_user]:
            target_items[interaction['item']] = interaction['weight']
        
        user_similarities = []
        
        for user in self.user_item_graph:
            if user != target_user:
                similarity = self.calculate_user_similarity(target_items, user)
                if similarity > 0:
                    user_similarities.append((user, similarity))
        
        # Sort by similarity and return top k
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        return user_similarities[:top_k]
    
    def calculate_user_similarity(self, target_items, other_user):
        """Calculate similarity between two users"""
        other_items = {}
        for interaction in self.user_item_graph[other_user]:
            other_items[interaction['item']] = interaction['weight']
        
        # Calculate cosine similarity
        common_items = set(target_items.keys()).intersection(set(other_items.keys()))
        
        if not common_items:
            return 0
        
        numerator = sum(target_items[item] * other_items[item] for item in common_items)
        
        target_norm = sum(weight ** 2 for weight in target_items.values()) ** 0.5
        other_norm = sum(weight ** 2 for weight in other_items.values()) ** 0.5
        
        if target_norm == 0 or other_norm == 0:
            return 0
        
        return numerator / (target_norm * other_norm)
    
    def recommend_with_diversity(self, user_id, num_recommendations=10, diversity_factor=0.3):
        """Recommendations with diversity to avoid filter bubbles"""
        base_recommendations = self.recommend_items_collaborative(user_id, num_recommendations * 2)
        
        if not base_recommendations:
            return []
        
        # Select diverse recommendations
        selected = []
        candidate_pool = base_recommendations
        
        while len(selected) < num_recommendations and candidate_pool:
            if not selected:
                # Select highest scoring item first
                selected.append(candidate_pool[0])
                candidate_pool = candidate_pool[1:]
            else:
                # Select item that balances score and diversity
                best_candidate = None
                best_score = -1
                
                for i, (item, score) in enumerate(candidate_pool):
                    # Calculate diversity from already selected items
                    diversity_score = self.calculate_diversity_score(item, 
                                                                   [s[0] for s in selected])
                    
                    # Combined score: relevance + diversity
                    combined_score = (1 - diversity_factor) * score + diversity_factor * diversity_score
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = (i, item, score)
                
                if best_candidate:
                    idx, item, score = best_candidate
                    selected.append((item, score))
                    candidate_pool.pop(idx)
                else:
                    break
        
        return selected
    
    def calculate_diversity_score(self, candidate_item, selected_items):
        """Calculate how diverse candidate item is from selected items"""
        if not selected_items:
            return 1.0
        
        # Calculate average dissimilarity
        total_dissimilarity = 0
        count = 0
        
        for selected_item in selected_items:
            # Find similarity between candidate and selected item
            similarity = 0
            for similar_item_info in self.item_similarity_graph[candidate_item]:
                if similar_item_info['item'] == selected_item:
                    similarity = similar_item_info['similarity']
                    break
            
            dissimilarity = 1 - similarity
            total_dissimilarity += dissimilarity
            count += 1
        
        return total_dissimilarity / count if count > 0 else 1.0
```

**Business Impact**: Increases user engagement by 20-40%, improves content discovery, drives revenue through better conversion rates, and enhances user satisfaction through personalized experiences.

## Advanced Techniques

### 1. **A* Search Algorithm**
Heuristic-guided shortest path finding for large graphs with goal-directed search.

### 2. **Johnson's Algorithm**
All-pairs shortest paths for sparse graphs, combining Dijkstra and Bellman-Ford.

### 3. **Network Flow Algorithms**
Maximum flow, minimum cut, and matching problems using Ford-Fulkerson or push-relabel methods.

### 4. **Strongly Connected Components**
Finding SCCs using Kosaraju's or Tarjan's algorithms for directed graph analysis.

### 5. **Graph Coloring and Partitioning**
Optimization problems for resource allocation and conflict resolution.

## Performance Characteristics

| Algorithm | Time Complexity | Space Complexity | Use Case |
|-----------|----------------|------------------|----------|
| Dijkstra | O((V + E) log V) | O(V) | Non-negative weights |
| Bellman-Ford | O(VE) | O(V) | Negative weights |
| Floyd-Warshall | O(V³) | O(V²) | All-pairs shortest path |
| Prim's MST | O((V + E) log V) | O(V) | Dense graphs |
| Kruskal's MST | O(E log E) | O(V) | Sparse graphs |

## When NOT to Use Graph Algorithms

1. **Simple data structures**: Arrays, lists don't require graph algorithms
2. **No optimization needed**: When any path/solution works
3. **Static analysis**: Pre-computed results are sufficient
4. **Linear relationships**: Sequential processing is more appropriate
5. **Memory constraints**: Graph algorithms can be memory-intensive

## Tips for Success

1. **Choose appropriate algorithm**: Match algorithm to problem constraints
2. **Consider graph representation**: Adjacency list vs matrix based on density
3. **Handle edge cases**: Disconnected graphs, self-loops, negative cycles
4. **Optimize data structures**: Use priority queues, Union-Find where appropriate
5. **Plan for scale**: Consider approximate algorithms for large graphs
6. **Validate results**: Test with known optimal solutions
7. **Monitor performance**: Profile algorithms with realistic data sizes

## Conclusion

Graph algorithms are essential for:
- Network optimization and routing
- Resource allocation and scheduling
- Social network analysis and recommendations
- Transportation and logistics optimization
- Infrastructure planning and design
- Modeling complex relationships and dependencies

Master these algorithms by understanding when optimization is needed beyond basic connectivity, choosing appropriate algorithms for specific constraints, and practicing with real-world network problems. The key insight is that many complex optimization problems can be modeled as graphs and solved efficiently using specialized algorithms designed for specific objective functions.
