# Heaps Pattern

## Pattern Overview

The **Heaps** pattern utilizes heap data structures to efficiently manage and retrieve elements based on priority. Heaps are specialized binary trees that maintain a specific ordering property, making them ideal for scenarios requiring quick access to minimum or maximum elements without the overhead of full sorting.

## When to Use This Pattern

Your problem matches this pattern if **any** of these conditions are fulfilled:

### ✅ Use Heaps When:

1. **Linear data**: If the input data is linear, it can be sorted or unsorted. A heap efficiently finds the maximum or minimum elements if the data is unsorted. Operations like insertion and deletion take O(log n) time, ensuring fast access to the top elements. If the data is sorted, a heap can still be useful when frequent insertions and deletions are required, as it allows for efficient updates and retrieval of the highest or lowest elements.

2. **Stream of data**: The input data continuously arrives in real time, often in an unpredictable order, requiring efficient handling and processing as it flows in. Heaps automatically enforce priority ordering (e.g., largest weight, smallest cost, highest frequency). This saves you from manually resorting or scanning each time your data changes.

3. **Calculation of maxima and minima**: The input data can be categorized into two parts, and we need to repeatedly calculate two maxima, two minima, or one maximum and one minimum from each set.

4. **Efficient retrieval of extreme values**: The solution needs to retrieve or update the min or max element repeatedly but cannot afford a full sort each time; a heap-based priority queue offers O(log n) insertion/removal and O(1) retrieval.

5. **Custom priority-based selection**: The problem involves selecting the next element based on specific priority at each step, such as processing the largest task or earliest event.

### ❌ Don't Use Heaps When:

1. **Random access needed**: When you need to access elements by index or position
2. **Searching for specific values**: Heaps don't support efficient searching for arbitrary elements
3. **Memory is extremely limited**: Heaps have overhead compared to simple arrays
4. **Static data with one-time access**: If you only need to access data once, sorting might be simpler

## Core Concepts

### Heap Properties

**Min Heap**: The value of each node is smaller than or equal to the values of its children. The root node holds the minimum value. A min heap always prioritizes the minimum value.

**Max Heap**: The value of each node is greater than or equal to the values of its children. The root node holds the maximum value. A max heap always prioritizes the maximum value.

**Priority Queue**: A priority queue is an abstract data type that retrieves elements based on their custom priority. It is often implemented using a heap for efficiency.

### Airport Flight Management Example

Imagine you're managing a busy airport. Flights are constantly landing and taking off, and you need to quickly find the next most important flight—an emergency landing or a VIP departure. At the same time, new flights must be integrated into the schedule. How do you track all this while finding the highest-priority flight quickly? 

Without an efficient data structure, you'd have to scan the entire schedule every time a decision is needed, which can be slow and error-prone as the number of flights grows. The time complexity of this inefficient system will be O(n) for each decision, where n is the number of flights because it requires scanning the entire schedule to find the highest-priority flight.

The solution is heaps. With a min heap, you can always find the flight with the earliest priority, and with a max heap, you can focus on flights that have been waiting for the longest—all while making updates quickly when new flights are added.

## Basic Implementation Templates

### Simple Min Heap Operations
```python
import heapq

# Create empty heap
min_heap = []

# Add elements
heapq.heappush(min_heap, 10)
heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 15)

# Get minimum (peek)
minimum = min_heap[0]  # O(1)

# Remove minimum
min_val = heapq.heappop(min_heap)  # O(log n)
```

### Simple Max Heap Operations
```python
# Python only has min heap, so negate values for max heap
max_heap = []

# Add elements (negate for max heap behavior)
heapq.heappush(max_heap, -10)
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -15)

# Get maximum (remember to negate back)
maximum = -max_heap[0]

# Remove maximum
max_val = -heapq.heappop(max_heap)
```

## Problem Categories

### 1. **Single Heap Problems**
- Finding kth largest/smallest element
- Priority queue implementations
- Task scheduling by priority
- Event processing by timestamp

### 2. **Two Heap Problems (Median)**
- Finding running median
- Balancing two datasets
- Streaming data analysis
- Range queries with updates

### 3. **K-Way Merge**
- Merging multiple sorted lists
- Combining sorted streams
- Distributed data aggregation
- Multi-source processing

### 4. **Custom Priority**
- CPU task scheduling
- Network packet prioritization
- Resource allocation
- Game AI decision making

### 5. **Heap with Hash Map**
- Top K frequent elements
- LFU cache implementation
- Frequency-based ranking
- Dynamic priority updates

## Real-World Applications

### 1. **Video Platforms - Demographic Analysis**
As part of a demographic study, we're interested in the median age of the viewers. We want to implement a functionality whereby the median age can be updated efficiently whenever a new user signs up for video streaming.

**Solution Approach**: Use two heaps - a max heap for the smaller half of ages and a min heap for the larger half. This allows O(1) median calculation and O(log n) insertion of new users.

**Business Impact**: Real-time demographic insights help with content recommendation, ad targeting, and user experience optimization without expensive batch processing.

### 2. **Gaming Matchmaking System**
Matching players of similar skill levels is crucial for a balanced and enjoyable gaming experience. By maintaining two heaps (one for minimum skill level and one for maximum skill level), matchmaking algorithms can efficiently pair players based on their skill levels.

**Solution Approach**: 
- Use min heap to track lowest-skilled players waiting for matches
- Use max heap to track highest-skilled players
- Efficiently find players within skill range for balanced matches
- Handle dynamic player pool with real-time updates

**Business Impact**: Better player retention through fair matches, reduced wait times, and improved gaming experience leading to higher engagement and revenue.

### 3. **Hospital Emergency Room Prioritization**
Emergency rooms need to process patients based on medical urgency, not arrival time. Critical patients must be seen immediately while others wait in order of severity.

**Solution Approach**:
- Max heap based on urgency scores (1-10 scale)
- Dynamic priority updates as patient conditions change
- Efficient insertion of new patients
- Quick retrieval of next most urgent case

**Business Impact**: Life-saving decision making, optimal resource utilization, regulatory compliance, and improved patient outcomes.

### 4. **Stock Trading System**
High-frequency trading systems need to process buy/sell orders efficiently while maintaining price-time priority.

**Solution Approach**:
- Separate heaps for buy orders (max heap by price) and sell orders (min heap by price)
- Within same price, maintain time priority
- Efficient order matching and execution
- Real-time market data processing

**Business Impact**: Reduced latency in order execution, fair market operations, improved liquidity, and competitive advantage in trading.

### 5. **Cloud Resource Scheduling**
Cloud platforms need to allocate computing resources efficiently based on job priority, resource requirements, and availability.

**Solution Approach**:
- Priority heaps for different job types (CPU-intensive, memory-intensive, etc.)
- Dynamic resource allocation based on current system load
- Preemption handling for higher-priority jobs
- Load balancing across multiple servers

**Business Impact**: Optimal resource utilization, improved service quality, cost reduction, and better customer satisfaction through faster job completion.

## Advanced Techniques

### 1. **Lazy Deletion**
Instead of immediately removing elements, mark them as deleted and clean up later. Useful when deletions are frequent and heap structure changes are expensive.

### 2. **Heap Merging**
Efficiently combine multiple heaps into one. Important for distributed systems and parallel processing scenarios.

### 3. **Custom Comparators**
Define complex priority rules using custom comparison functions for objects with multiple attributes.

### 4. **Persistent Heaps**
Maintain historical versions of heap state for undo operations or temporal queries.

### 5. **Approximate Heaps**
Trade exact ordering for improved performance in scenarios where approximate priority is sufficient.

## Performance Characteristics

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Insert | O(log n) | O(1) | Maintains heap property |
| Extract Min/Max | O(log n) | O(1) | Root removal and reheapify |
| Peek Min/Max | O(1) | O(1) | Access root element |
| Build Heap | O(n) | O(1) additional | From existing array |
| Merge Heaps | O(log n) | O(1) additional | For specific heap types |

## Common Patterns and Variations

### 1. **Two Heaps Pattern**
- **Use Case**: Finding median, balancing datasets
- **Structure**: Max heap (smaller half) + Min heap (larger half)
- **Maintenance**: Keep size difference ≤ 1

### 2. **K-Way Merge Pattern**
- **Use Case**: Merging k sorted arrays/streams
- **Structure**: Min heap with (value, array_index, element_index)
- **Process**: Always extract minimum and add next from same array

### 3. **Top K Pattern**
- **Use Case**: Finding k largest/smallest elements
- **Structure**: Fixed-size heap of size k
- **Optimization**: Only maintain k elements, not entire dataset

### 4. **Sliding Window with Heap**
- **Use Case**: Maximum/minimum in sliding windows
- **Challenge**: Heap doesn't support arbitrary deletion
- **Solution**: Lazy deletion or balanced BST

### 5. **Frequency-Based Heaps**
- **Use Case**: Top k frequent elements
- **Structure**: Heap + Hash map for frequency tracking
- **Pattern**: Count frequencies, then use heap for selection

## Practical Problem Examples

### Beginner Level
1. **Kth Largest Element** - Basic heap usage
2. **Last Stone Weight** - Simulation with max heap
3. **Meeting Rooms** - Interval scheduling with min heap
4. **Merge Two Sorted Lists** - Simple merging concept

### Intermediate Level
5. **Find Median from Data Stream** - Two heaps pattern
6. **Top K Frequent Elements** - Frequency counting + heap
7. **Merge k Sorted Lists** - K-way merge pattern
8. **Task Scheduler** - Priority-based scheduling
9. **Ugly Number II** - Multiple heaps coordination

### Advanced Level
10. **Sliding Window Maximum** - Complex heap management
11. **Design Twitter** - Multiple heaps with time priority
12. **IPO** - Greedy with dual heap optimization
13. **Median of Two Sorted Arrays** - Advanced median concepts
14. **Shortest Path in Graph** - Dijkstra's algorithm with heap

## Common Pitfalls and Solutions

### 1. **Forgetting Heap Property**
- **Problem**: Modifying elements without maintaining heap property
- **Solution**: Always use proper heap operations (push/pop)

### 2. **Max Heap in Python**
- **Problem**: Python only provides min heap
- **Solution**: Negate values or use custom comparison

### 3. **Memory Management**
- **Problem**: Heap growing too large with deleted elements
- **Solution**: Implement lazy deletion or periodic cleanup

### 4. **Tie Breaking**
- **Problem**: Elements with same priority need consistent ordering
- **Solution**: Use tuples with secondary sort criteria

### 5. **Dynamic Priority Updates**
- **Problem**: Changing priority of existing elements
- **Solution**: Use indexed heap or remove/re-insert pattern

## When NOT to Use Heaps

1. **Need for random access**: Arrays or lists are better
2. **Simple min/max of static data**: Linear scan is simpler
3. **Frequent arbitrary deletions**: Balanced BST might be better
4. **Memory-constrained environments**: Simple arrays use less memory
5. **Stable sorting requirements**: Heaps don't preserve insertion order

## Tips for Success

1. **Choose the right heap type**: Min heap for smallest, max heap for largest
2. **Consider two-heap patterns**: Great for median and balancing problems
3. **Handle edge cases**: Empty heaps, single elements, equal priorities
4. **Plan for scalability**: Consider memory usage and performance at scale
5. **Test with edge cases**: Empty input, single element, all equal elements
6. **Use tuples for complex priorities**: Multiple criteria sorting
7. **Consider alternatives**: Sometimes simple sorting is better for small datasets

## Conclusion

The Heaps pattern is fundamental for:
- Priority-based processing and scheduling
- Efficient extreme value retrieval
- Real-time data stream processing
- Median and quantile calculations
- Resource allocation and optimization
- Event-driven system design

Master this pattern by understanding heap properties, recognizing when priority-based access is needed, and practicing with various heap configurations. The key insight is that heaps excel when you need efficient access to extreme values while maintaining the ability to dynamically insert and remove elements.
