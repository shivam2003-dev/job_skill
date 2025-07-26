# Top K Elements Pattern

## Pattern Overview

The **Top K Elements** pattern is used to efficiently find a specific subset of elements from an unsorted collection based on criteria like size, frequency, or other comparative metrics. It leverages heap data structures and selection algorithms to avoid full sorting when only the top K elements are needed.

## When to Use This Pattern

Your problem matches this pattern if **both** conditions are fulfilled:

### ✅ Use Top K Elements When:

1. **Unsorted list analysis**: We need to extract a specific subset of elements based on their size (largest or smallest), frequency (most or least frequent), or other similar criteria from an unsorted list. This may be the requirement of the final solution, or it may be necessary as an intermediate step toward the final solution.

2. **Identifying a specific subset**: The goal is to identify a subset rather than just a single extreme value. When phrases like "top k", "kth largest/smallest", "k most frequent", "k closest", or "k highest/lowest" describe our task, it suggests the top k elements pattern is ideal for efficiently identifying a specific subset.

### ❌ Don't Use Top K Elements When:

1. **Presorted input**: The input data is already sorted according to the criteria relevant to solving the problem.

2. **Single extreme value**: If only 1 extreme value (either the maximum or minimum) is required, that is, k=1, as that problem can be solved in O(n) with a simple linear scan through the input data.

### Additional Considerations:
- When you need all elements sorted (use full sort instead)
- When k is very close to n (total elements)
- When memory is extremely constrained

## Core Concepts

### Heap-Based Approach
- **Min Heap for Top K Largest**: Keep k largest elements, root is smallest of the k largest
- **Max Heap for Top K Smallest**: Keep k smallest elements, root is largest of the k smallest
- **Time Complexity**: O(n log k) where n is total elements
- **Space Complexity**: O(k)

### Quick Select Approach
- **Partition-based**: Similar to quicksort partitioning
- **Time Complexity**: O(n) average, O(n²) worst case
- **Space Complexity**: O(1) if in-place
- **Best for**: Finding kth element, not necessarily maintaining order

## Implementation Templates

### Basic Top K Largest Template
```python
import heapq

def top_k_largest(nums, k):
    """Find k largest elements using min heap"""
    if k >= len(nums):
        return nums
    
    # Use min heap to keep track of k largest elements
    min_heap = []
    
    for num in nums:
        if len(min_heap) < k:
            heapq.heappush(min_heap, num)
        elif num > min_heap[0]:
            heapq.heapreplace(min_heap, num)
    
    # Return in descending order
    result = []
    while min_heap:
        result.append(heapq.heappop(min_heap))
    
    return result[::-1]  # Reverse to get descending order
```

### Basic Top K Smallest Template
```python
def top_k_smallest(nums, k):
    """Find k smallest elements using max heap"""
    if k >= len(nums):
        return sorted(nums)
    
    # Use max heap (negative values) to keep track of k smallest elements
    max_heap = []
    
    for num in nums:
        if len(max_heap) < k:
            heapq.heappush(max_heap, -num)
        elif num < -max_heap[0]:
            heapq.heapreplace(max_heap, -num)
    
    # Return in ascending order
    result = []
    while max_heap:
        result.append(-heapq.heappop(max_heap))
    
    return result[::-1]  # Reverse to get ascending order
```

### Top K Frequent Elements Template
```python
def top_k_frequent(nums, k):
    """Find k most frequent elements"""
    from collections import Counter
    
    # Count frequencies
    count = Counter(nums)
    
    # Use min heap to keep k most frequent elements
    heap = []
    
    for num, freq in count.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        elif freq > heap[0][0]:
            heapq.heapreplace(heap, (freq, num))
    
    # Extract elements (not frequencies)
    return [num for freq, num in heap]
```

### Top K Closest to Target Template
```python
def top_k_closest(nums, target, k):
    """Find k numbers closest to target"""
    if k >= len(nums):
        return nums
    
    # Use max heap to keep k closest elements
    # Store (distance, original_index, value) to handle ties
    max_heap = []
    
    for i, num in enumerate(nums):
        distance = abs(num - target)
        
        if len(max_heap) < k:
            heapq.heappush(max_heap, (-distance, -i, num))
        elif distance < -max_heap[0][0]:
            heapq.heapreplace(max_heap, (-distance, -i, num))
    
    return [num for _, _, num in max_heap]
```

### Quick Select Template (Kth Largest)
```python
import random

def quick_select_kth_largest(nums, k):
    """Find kth largest element using quickselect"""
    def partition(left, right, pivot_index):
        pivot_value = nums[pivot_index]
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        store_index = left
        for i in range(left, right):
            if nums[i] > pivot_value:  # For kth largest
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        
        # Move pivot to its final place
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index
    
    def select(left, right, k_smallest):
        if left == right:
            return nums[left]
        
        pivot_index = random.randint(left, right)
        pivot_index = partition(left, right, pivot_index)
        
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return select(left, pivot_index - 1, k_smallest)
        else:
            return select(pivot_index + 1, right, k_smallest)
    
    return select(0, len(nums) - 1, k - 1)
```

### Bucket Sort Approach Template
```python
def top_k_frequent_bucket_sort(nums, k):
    """Top K frequent using bucket sort approach"""
    from collections import defaultdict, Counter
    
    count = Counter(nums)
    
    # Create buckets for each frequency
    buckets = defaultdict(list)
    for num, freq in count.items():
        buckets[freq].append(num)
    
    result = []
    # Start from highest frequency
    for freq in range(len(nums), 0, -1):
        if freq in buckets:
            result.extend(buckets[freq])
            if len(result) >= k:
                break
    
    return result[:k]
```

## Advanced Techniques

### 1. **Streaming Top K**
```python
class StreamingTopK:
    def __init__(self, k):
        self.k = k
        self.min_heap = []  # For top k largest
    
    def add(self, val):
        """Add value to stream and maintain top k"""
        if len(self.min_heap) < self.k:
            heapq.heappush(self.min_heap, val)
        elif val > self.min_heap[0]:
            heapq.heapreplace(self.min_heap, val)
    
    def get_top_k(self):
        """Get current top k elements"""
        return sorted(self.min_heap, reverse=True)
    
    def get_kth_largest(self):
        """Get kth largest element"""
        return self.min_heap[0] if self.min_heap else None
```

### 2. **Top K with Custom Comparator**
```python
def top_k_custom(items, k, key_func, reverse=False):
    """Top K with custom comparison function"""
    import heapq
    
    if reverse:  # For largest values
        heap = []
        for item in items:
            key_val = key_func(item)
            if len(heap) < k:
                heapq.heappush(heap, (key_val, item))
            elif key_val > heap[0][0]:
                heapq.heapreplace(heap, (key_val, item))
    else:  # For smallest values
        heap = []
        for item in items:
            key_val = key_func(item)
            if len(heap) < k:
                heapq.heappush(heap, (-key_val, item))
            elif key_val < -heap[0][0]:
                heapq.heapreplace(heap, (-key_val, item))
    
    return [item for _, item in heap]

# Example usage
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __repr__(self):
        return f"Person({self.name}, {self.age})"

people = [Person("Alice", 25), Person("Bob", 30), Person("Charlie", 20)]
oldest_2 = top_k_custom(people, 2, lambda p: p.age, reverse=True)
```

### 3. **Distributed Top K**
```python
def merge_top_k_lists(lists, k):
    """Merge multiple top-k lists into single top-k"""
    import heapq
    
    # Use heap to merge multiple sorted lists
    heap = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (-lst[0], 0, i))  # (negative value, index, list_id)
    
    result = []
    
    while heap and len(result) < k:
        neg_val, idx, list_id = heapq.heappop(heap)
        value = -neg_val
        result.append(value)
        
        # Add next element from the same list
        if idx + 1 < len(lists[list_id]):
            next_val = lists[list_id][idx + 1]
            heapq.heappush(heap, (-next_val, idx + 1, list_id))
    
    return result
```

### 4. **Top K with Sliding Window**
```python
def top_k_sliding_window(nums, k, window_size):
    """Find top k elements in each sliding window"""
    from collections import deque, Counter
    
    result = []
    window_count = Counter()
    
    # Initialize first window
    for i in range(window_size):
        window_count[nums[i]] += 1
    
    # Get top k for first window
    result.append(get_top_k_from_counter(window_count, k))
    
    # Slide the window
    for i in range(window_size, len(nums)):
        # Remove leftmost element
        left_elem = nums[i - window_size]
        window_count[left_elem] -= 1
        if window_count[left_elem] == 0:
            del window_count[left_elem]
        
        # Add rightmost element
        window_count[nums[i]] += 1
        
        # Get top k for current window
        result.append(get_top_k_from_counter(window_count, k))
    
    return result

def get_top_k_from_counter(counter, k):
    """Extract top k elements from Counter"""
    return [item for item, count in counter.most_common(k)]
```

## Problem Categories

### 1. **Kth Element Problems**
- Kth largest/smallest element
- Kth most frequent element
- Kth closest to target
- Kth unique element

### 2. **Top K Selection**
- K largest/smallest elements
- K most/least frequent elements
- K closest points to origin
- K strongest elements

### 3. **Frequency-Based Top K**
- Most frequent words
- Top trending hashtags
- Most visited pages
- Popular products

### 4. **Distance-Based Top K**
- K nearest neighbors
- K closest points
- K similar items
- K best matches

### 5. **Performance-Based Top K**
- Top performers
- Best scores
- Highest ratings
- Most efficient algorithms

### 6. **Streaming Top K**
- Real-time top trends
- Live leaderboards
- Dynamic rankings
- Continuous monitoring

## Step-by-Step Problem-Solving Approach

### 1. **Identify the Selection Criteria**
- What metric determines "top"? (size, frequency, distance, score)
- Do you need largest or smallest values?
- Are there any tie-breaking rules?

### 2. **Choose the Right Algorithm**
- **Heap**: Most common, O(n log k), good for streaming
- **Quick Select**: O(n) average, good for one-time kth element
- **Bucket Sort**: O(n), good when range is limited
- **Full Sort**: O(n log n), when k is close to n

### 3. **Determine Data Structures**
- **Min Heap**: For k largest elements
- **Max Heap**: For k smallest elements
- **Custom Heap**: For complex comparison criteria
- **Counter + Heap**: For frequency-based problems

### 4. **Handle Edge Cases**
- k >= n (return all elements)
- k <= 0 (return empty or handle error)
- Empty input
- Duplicate values and tie-breaking

### 5. **Optimize Based on Constraints**
- Memory limitations (prefer streaming approaches)
- Real-time requirements (maintain running top-k)
- Distribution needs (parallel processing)

## Real-World Applications

### 1. **Location-Based Services in Ride-Sharing Apps (Uber)**
Find the n closest drivers to ensure quick pickup:

```python
class RideShareMatcher:
    def __init__(self):
        self.drivers = {}  # driver_id -> (lat, lon, status)
    
    def add_driver(self, driver_id, lat, lon, status="available"):
        """Add or update driver location"""
        self.drivers[driver_id] = (lat, lon, status)
    
    def find_closest_drivers(self, user_lat, user_lon, k=5):
        """Find k closest available drivers to user"""
        import heapq
        import math
        
        # Max heap to keep k closest drivers
        max_heap = []
        
        for driver_id, (lat, lon, status) in self.drivers.items():
            if status != "available":
                continue
            
            # Calculate distance (simplified Euclidean distance)
            distance = math.sqrt((lat - user_lat)**2 + (lon - user_lon)**2)
            
            if len(max_heap) < k:
                heapq.heappush(max_heap, (-distance, driver_id, lat, lon))
            elif distance < -max_heap[0][0]:
                heapq.heapreplace(max_heap, (-distance, driver_id, lat, lon))
        
        # Return drivers sorted by distance (closest first)
        result = []
        while max_heap:
            neg_distance, driver_id, lat, lon = heapq.heappop(max_heap)
            result.append({
                'driver_id': driver_id,
                'distance': -neg_distance,
                'lat': lat,
                'lon': lon
            })
        
        return result[::-1]  # Reverse to get closest first
    
    def get_driver_demand_insights(self, k=10):
        """Get top k areas with highest driver concentration"""
        from collections import defaultdict
        
        # Group drivers by grid cells (simplified)
        grid_counts = defaultdict(int)
        
        for driver_id, (lat, lon, status) in self.drivers.items():
            if status == "available":
                # Simplified grid: round to nearest 0.01 degree
                grid_key = (round(lat, 2), round(lon, 2))
                grid_counts[grid_key] += 1
        
        # Find top k grid cells with most drivers
        top_areas = []
        heap = []
        
        for grid_key, count in grid_counts.items():
            if len(heap) < k:
                heapq.heappush(heap, (count, grid_key))
            elif count > heap[0][0]:
                heapq.heapreplace(heap, (count, grid_key))
        
        result = []
        while heap:
            count, (lat, lon) = heapq.heappop(heap)
            result.append({
                'area': f"({lat}, {lon})",
                'driver_count': count
            })
        
        return result[::-1]  # Highest count first

# Usage example
matcher = RideShareMatcher()

# Add drivers
matcher.add_driver("driver1", 40.7128, -74.0060, "available")  # NYC
matcher.add_driver("driver2", 40.7614, -73.9776, "available")  # Central Park
matcher.add_driver("driver3", 40.6892, -74.0445, "busy")       # Brooklyn
matcher.add_driver("driver4", 40.7505, -73.9934, "available")  # Times Square

# Find closest drivers for a user
user_location = (40.7580, -73.9855)  # Near Times Square
closest_drivers = matcher.find_closest_drivers(user_location[0], user_location[1], k=3)
print("Closest drivers:", closest_drivers)

# Get area insights
demand_insights = matcher.get_driver_demand_insights(k=5)
print("High-demand areas:", demand_insights)
```

### 2. **Performance Analysis in Financial Markets**
Identify top-performing brokers by transaction volume and success metrics:

```python
class FinancialPerformanceAnalyzer:
    def __init__(self):
        self.broker_data = {}  # broker_id -> metrics
        self.transaction_history = []  # List of transactions
    
    def add_transaction(self, broker_id, volume, profit, timestamp, transaction_type):
        """Record a broker transaction"""
        transaction = {
            'broker_id': broker_id,
            'volume': volume,
            'profit': profit,
            'timestamp': timestamp,
            'type': transaction_type
        }
        self.transaction_history.append(transaction)
        
        # Update broker metrics
        if broker_id not in self.broker_data:
            self.broker_data[broker_id] = {
                'total_volume': 0,
                'total_profit': 0,
                'transaction_count': 0,
                'profitable_trades': 0,
                'win_rate': 0.0
            }
        
        broker = self.broker_data[broker_id]
        broker['total_volume'] += volume
        broker['total_profit'] += profit
        broker['transaction_count'] += 1
        
        if profit > 0:
            broker['profitable_trades'] += 1
        
        broker['win_rate'] = broker['profitable_trades'] / broker['transaction_count']
    
    def get_top_brokers_by_volume(self, k=10):
        """Find top k brokers by transaction volume"""
        import heapq
        
        min_heap = []
        
        for broker_id, metrics in self.broker_data.items():
            volume = metrics['total_volume']
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (volume, broker_id))
            elif volume > min_heap[0][0]:
                heapq.heapreplace(min_heap, (volume, broker_id))
        
        # Return sorted by volume (highest first)
        result = []
        while min_heap:
            volume, broker_id = heapq.heappop(min_heap)
            result.append({
                'broker_id': broker_id,
                'total_volume': volume,
                'metrics': self.broker_data[broker_id]
            })
        
        return result[::-1]
    
    def get_top_brokers_by_profit(self, k=10):
        """Find top k brokers by total profit"""
        import heapq
        
        min_heap = []
        
        for broker_id, metrics in self.broker_data.items():
            profit = metrics['total_profit']
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (profit, broker_id))
            elif profit > min_heap[0][0]:
                heapq.heapreplace(min_heap, (profit, broker_id))
        
        result = []
        while min_heap:
            profit, broker_id = heapq.heappop(min_heap)
            result.append({
                'broker_id': broker_id,
                'total_profit': profit,
                'metrics': self.broker_data[broker_id]
            })
        
        return result[::-1]
    
    def get_top_brokers_by_win_rate(self, k=10, min_transactions=10):
        """Find top k brokers by win rate (with minimum transaction requirement)"""
        import heapq
        
        min_heap = []
        
        for broker_id, metrics in self.broker_data.items():
            if metrics['transaction_count'] < min_transactions:
                continue
            
            win_rate = metrics['win_rate']
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (win_rate, broker_id))
            elif win_rate > min_heap[0][0]:
                heapq.heapreplace(min_heap, (win_rate, broker_id))
        
        result = []
        while min_heap:
            win_rate, broker_id = heapq.heappop(min_heap)
            result.append({
                'broker_id': broker_id,
                'win_rate': win_rate,
                'metrics': self.broker_data[broker_id]
            })
        
        return result[::-1]
    
    def get_composite_performance_ranking(self, k=10, 
                                        volume_weight=0.3, 
                                        profit_weight=0.4, 
                                        win_rate_weight=0.3):
        """Rank brokers using composite score"""
        import heapq
        
        # Normalize metrics first
        volumes = [m['total_volume'] for m in self.broker_data.values()]
        profits = [m['total_profit'] for m in self.broker_data.values()]
        win_rates = [m['win_rate'] for m in self.broker_data.values()]
        
        max_volume = max(volumes) if volumes else 1
        max_profit = max(profits) if profits else 1
        max_win_rate = max(win_rates) if win_rates else 1
        
        min_heap = []
        
        for broker_id, metrics in self.broker_data.items():
            # Calculate composite score
            normalized_volume = metrics['total_volume'] / max_volume
            normalized_profit = metrics['total_profit'] / max_profit
            normalized_win_rate = metrics['win_rate'] / max_win_rate
            
            composite_score = (
                volume_weight * normalized_volume +
                profit_weight * normalized_profit +
                win_rate_weight * normalized_win_rate
            )
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (composite_score, broker_id))
            elif composite_score > min_heap[0][0]:
                heapq.heapreplace(min_heap, (composite_score, broker_id))
        
        result = []
        while min_heap:
            score, broker_id = heapq.heappop(min_heap)
            result.append({
                'broker_id': broker_id,
                'composite_score': score,
                'metrics': self.broker_data[broker_id]
            })
        
        return result[::-1]
    
    def analyze_time_period_performance(self, start_time, end_time, k=5):
        """Analyze top performers in specific time period"""
        from collections import defaultdict
        
        period_data = defaultdict(lambda: {
            'volume': 0, 'profit': 0, 'transactions': 0
        })
        
        for transaction in self.transaction_history:
            if start_time <= transaction['timestamp'] <= end_time:
                broker_id = transaction['broker_id']
                period_data[broker_id]['volume'] += transaction['volume']
                period_data[broker_id]['profit'] += transaction['profit']
                period_data[broker_id]['transactions'] += 1
        
        # Find top k by volume in this period
        import heapq
        min_heap = []
        
        for broker_id, data in period_data.items():
            volume = data['volume']
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (volume, broker_id))
            elif volume > min_heap[0][0]:
                heapq.heapreplace(min_heap, (volume, broker_id))
        
        result = []
        while min_heap:
            volume, broker_id = heapq.heappop(min_heap)
            result.append({
                'broker_id': broker_id,
                'period_volume': volume,
                'period_data': period_data[broker_id]
            })
        
        return result[::-1]

# Usage example
analyzer = FinancialPerformanceAnalyzer()

# Add sample transactions
import time
current_time = time.time()

brokers = ['BROKER_A', 'BROKER_B', 'BROKER_C', 'BROKER_D', 'BROKER_E']
for i in range(100):
    broker = brokers[i % len(brokers)]
    volume = 1000 + (i * 100)
    profit = (-50 + (i % 3) * 100)  # Some losses, some gains
    analyzer.add_transaction(broker, volume, profit, current_time + i, 'STOCK')

# Analyze performance
top_by_volume = analyzer.get_top_brokers_by_volume(k=3)
top_by_profit = analyzer.get_top_brokers_by_profit(k=3)
top_by_win_rate = analyzer.get_top_brokers_by_win_rate(k=3)
composite_ranking = analyzer.get_composite_performance_ranking(k=3)

print("Top brokers by volume:", top_by_volume)
print("Top brokers by profit:", top_by_profit)
print("Top brokers by win rate:", top_by_win_rate)
print("Composite ranking:", composite_ranking)
```

### 3. **Social Media Trend Analysis**
Identify trending topics by analyzing hashtags and keywords:

```python
class SocialMediaTrendAnalyzer:
    def __init__(self, trend_window_hours=24):
        self.hashtag_counts = {}    # hashtag -> count
        self.keyword_counts = {}    # keyword -> count
        self.posts_timeline = []    # (timestamp, hashtags, keywords, engagement)
        self.trend_window = trend_window_hours * 3600  # Convert to seconds
        self.engagement_weights = {'like': 1, 'share': 3, 'comment': 2}
    
    def add_post(self, hashtags, keywords, engagement_data, timestamp):
        """Add a social media post for analysis"""
        # Calculate engagement score
        engagement_score = 0
        for action, count in engagement_data.items():
            weight = self.engagement_weights.get(action, 1)
            engagement_score += count * weight
        
        # Store post data
        self.posts_timeline.append({
            'timestamp': timestamp,
            'hashtags': hashtags,
            'keywords': keywords,
            'engagement': engagement_score
        })
        
        # Update hashtag counts
        for hashtag in hashtags:
            self.hashtag_counts[hashtag] = self.hashtag_counts.get(hashtag, 0) + 1
        
        # Update keyword counts
        for keyword in keywords:
            self.keyword_counts[keyword] = self.keyword_counts.get(keyword, 0) + 1
    
    def get_trending_hashtags(self, k=10, current_time=None):
        """Get top k trending hashtags in recent time window"""
        import heapq
        import time
        
        if current_time is None:
            current_time = time.time()
        
        cutoff_time = current_time - self.trend_window
        
        # Count hashtags in recent time window
        recent_hashtag_counts = {}
        recent_engagement_weights = {}
        
        for post in self.posts_timeline:
            if post['timestamp'] >= cutoff_time:
                for hashtag in post['hashtags']:
                    recent_hashtag_counts[hashtag] = recent_hashtag_counts.get(hashtag, 0) + 1
                    recent_engagement_weights[hashtag] = recent_engagement_weights.get(hashtag, 0) + post['engagement']
        
        # Calculate trending score (frequency + engagement)
        min_heap = []
        
        for hashtag, count in recent_hashtag_counts.items():
            engagement = recent_engagement_weights.get(hashtag, 0)
            trending_score = count + (engagement / 100)  # Weighted combination
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (trending_score, hashtag))
            elif trending_score > min_heap[0][0]:
                heapq.heapreplace(min_heap, (trending_score, hashtag))
        
        # Return results
        result = []
        while min_heap:
            score, hashtag = heapq.heappop(min_heap)
            result.append({
                'hashtag': hashtag,
                'trending_score': score,
                'recent_count': recent_hashtag_counts[hashtag],
                'engagement': recent_engagement_weights.get(hashtag, 0)
            })
        
        return result[::-1]  # Highest score first
    
    def get_emerging_trends(self, k=5, growth_threshold=2.0):
        """Find hashtags with rapid growth (emerging trends)"""
        import heapq
        import time
        
        current_time = time.time()
        
        # Compare recent period vs previous period
        recent_start = current_time - self.trend_window
        previous_start = current_time - (2 * self.trend_window)
        previous_end = recent_start
        
        recent_counts = {}
        previous_counts = {}
        
        for post in self.posts_timeline:
            timestamp = post['timestamp']
            
            if timestamp >= recent_start:
                # Recent period
                for hashtag in post['hashtags']:
                    recent_counts[hashtag] = recent_counts.get(hashtag, 0) + 1
            elif previous_start <= timestamp < previous_end:
                # Previous period
                for hashtag in post['hashtags']:
                    previous_counts[hashtag] = previous_counts.get(hashtag, 0) + 1
        
        # Calculate growth rates
        min_heap = []
        
        for hashtag, recent_count in recent_counts.items():
            previous_count = previous_counts.get(hashtag, 1)  # Avoid division by zero
            growth_rate = recent_count / previous_count
            
            if growth_rate >= growth_threshold:
                if len(min_heap) < k:
                    heapq.heappush(min_heap, (growth_rate, hashtag))
                elif growth_rate > min_heap[0][0]:
                    heapq.heapreplace(min_heap, (growth_rate, hashtag))
        
        result = []
        while min_heap:
            growth_rate, hashtag = heapq.heappop(min_heap)
            result.append({
                'hashtag': hashtag,
                'growth_rate': growth_rate,
                'recent_count': recent_counts[hashtag],
                'previous_count': previous_counts.get(hashtag, 0)
            })
        
        return result[::-1]
    
    def get_top_influencers_by_hashtag(self, hashtag, k=10):
        """Find top users posting about specific hashtag"""
        from collections import defaultdict
        import heapq
        
        user_engagement = defaultdict(int)
        user_post_count = defaultdict(int)
        
        # This would typically require user information in posts
        # For simulation, we'll use post IDs as user proxies
        for i, post in enumerate(self.posts_timeline):
            if hashtag in post['hashtags']:
                user_id = f"user_{i % 20}"  # Simulate users
                user_engagement[user_id] += post['engagement']
                user_post_count[user_id] += 1
        
        # Calculate influence score
        min_heap = []
        
        for user_id, total_engagement in user_engagement.items():
            post_count = user_post_count[user_id]
            avg_engagement = total_engagement / post_count
            influence_score = total_engagement + (avg_engagement * 0.1)
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (influence_score, user_id))
            elif influence_score > min_heap[0][0]:
                heapq.heapreplace(min_heap, (influence_score, user_id))
        
        result = []
        while min_heap:
            score, user_id = heapq.heappop(min_heap)
            result.append({
                'user_id': user_id,
                'influence_score': score,
                'total_engagement': user_engagement[user_id],
                'post_count': user_post_count[user_id]
            })
        
        return result[::-1]
    
    def analyze_hashtag_correlations(self, target_hashtag, k=5):
        """Find hashtags frequently used together with target hashtag"""
        from collections import defaultdict
        import heapq
        
        co_occurrence = defaultdict(int)
        target_appearances = 0
        
        for post in self.posts_timeline:
            hashtags = set(post['hashtags'])
            
            if target_hashtag in hashtags:
                target_appearances += 1
                for hashtag in hashtags:
                    if hashtag != target_hashtag:
                        co_occurrence[hashtag] += 1
        
        # Calculate correlation strength
        min_heap = []
        
        for hashtag, count in co_occurrence.items():
            correlation_strength = count / target_appearances if target_appearances > 0 else 0
            
            if len(min_heap) < k:
                heapq.heappush(min_heap, (correlation_strength, hashtag))
            elif correlation_strength > min_heap[0][0]:
                heapq.heapreplace(min_heap, (correlation_strength, hashtag))
        
        result = []
        while min_heap:
            strength, hashtag = heapq.heappop(min_heap)
            result.append({
                'hashtag': hashtag,
                'correlation_strength': strength,
                'co_occurrences': co_occurrence[hashtag],
                'target_appearances': target_appearances
            })
        
        return result[::-1]
    
    def generate_trend_report(self, k=10):
        """Generate comprehensive trend analysis report"""
        import time
        
        current_time = time.time()
        
        report = {
            'timestamp': current_time,
            'top_trending': self.get_trending_hashtags(k),
            'emerging_trends': self.get_emerging_trends(k//2),
            'total_posts_analyzed': len(self.posts_timeline),
            'unique_hashtags': len(self.hashtag_counts),
            'time_window_hours': self.trend_window / 3600
        }
        
        return report

# Usage example
analyzer = SocialMediaTrendAnalyzer(trend_window_hours=24)

# Simulate social media posts
import time
import random

current_time = time.time()
hashtag_pool = ['#AI', '#MachineLearning', '#Python', '#DataScience', '#TechNews', 
               '#Innovation', '#Startup', '#Cloud', '#Programming', '#BigData']
keyword_pool = ['artificial intelligence', 'machine learning', 'data analysis', 
               'technology', 'innovation', 'programming', 'cloud computing']

# Add sample posts
for i in range(200):
    timestamp = current_time - random.randint(0, 48*3600)  # Random time in last 48 hours
    
    # Random hashtags and keywords
    post_hashtags = random.sample(hashtag_pool, random.randint(1, 4))
    post_keywords = random.sample(keyword_pool, random.randint(1, 3))
    
    # Random engagement
    engagement = {
        'like': random.randint(10, 1000),
        'share': random.randint(1, 100),
        'comment': random.randint(1, 50)
    }
    
    analyzer.add_post(post_hashtags, post_keywords, engagement, timestamp)

# Generate analysis
trending = analyzer.get_trending_hashtags(k=5)
emerging = analyzer.get_emerging_trends(k=3)
correlations = analyzer.analyze_hashtag_correlations('#AI', k=3)
report = analyzer.generate_trend_report(k=5)

print("Trending hashtags:", trending)
print("Emerging trends:", emerging)
print("Hashtags correlated with #AI:", correlations)
print("Full trend report:", report)
```

## Practice Problems

### Beginner Level
1. **Kth Largest Element in Array** - Basic heap usage
2. **Top K Frequent Elements** - Frequency counting with heap
3. **K Closest Points to Origin** - Distance-based selection
4. **Kth Smallest Element in Sorted Matrix** - 2D array selection
5. **Find K Pairs with Smallest Sums** - Pair generation and selection

### Intermediate Level
6. **Top K Frequent Words** - String frequency with custom sorting
7. **K Closest Points to Origin** - Advanced distance calculations
8. **Kth Largest Element in Stream** - Streaming top K
9. **Reorganize String** - Frequency-based string construction
10. **Task Scheduler** - Scheduling with frequency constraints
11. **Top K Buzzing Words** - Real-time trending analysis
12. **K Highest Ranked Items** - Multi-criteria ranking

### Advanced Level
13. **Merge K Sorted Lists** - Distributed top K merging
14. **Smallest Range Covering K Lists** - Complex optimization
15. **Maximum Performance of Team** - Multi-objective optimization
16. **Campus Bikes II** - Assignment optimization with distance
17. **Split Array into Consecutive Subsequences** - Frequency-based partitioning
18. **Design A Leaderboard** - Dynamic ranking system

## Common Patterns and Optimizations

### 1. **Heap Size Optimization**
```python
def top_k_optimized(nums, k):
    """Optimize heap operations"""
    import heapq
    
    if k >= len(nums):
        return sorted(nums, reverse=True)
    
    # Use smaller heap size
    if k <= len(nums) - k:
        # Use min heap for top k largest
        min_heap = nums[:k]
        heapq.heapify(min_heap)
        
        for num in nums[k:]:
            if num > min_heap[0]:
                heapq.heapreplace(min_heap, num)
        
        return sorted(min_heap, reverse=True)
    else:
        # Use max heap for bottom (n-k) smallest
        max_heap = [-x for x in nums[:len(nums)-k]]
        heapq.heapify(max_heap)
        
        for num in nums[len(nums)-k:]:
            if num > -max_heap[0]:
                heapq.heapreplace(max_heap, -num)
        
        # Return complement
        bottom_elements = set(-x for x in max_heap)
        return [x for x in nums if x not in bottom_elements]
```

### 2. **Early Termination**
```python
def kth_largest_with_early_termination(nums, k):
    """Stop early when kth element is found"""
    import heapq
    
    # Use max heap and stop when we have k elements
    max_heap = []
    
    for num in nums:
        heapq.heappush(max_heap, -num)
        
        # Early termination for streaming scenario
        if len(max_heap) == k:
            # We have k largest elements so far
            pass
    
    # Extract kth largest
    for _ in range(k-1):
        heapq.heappop(max_heap)
    
    return -max_heap[0]
```

### 3. **Memory-Efficient Streaming**
```python
def streaming_top_k_memory_efficient(stream, k):
    """Memory-efficient streaming top K"""
    import heapq
    
    min_heap = []
    
    for value in stream:
        if len(min_heap) < k:
            heapq.heappush(min_heap, value)
        elif value > min_heap[0]:
            heapq.heapreplace(min_heap, value)
        
        # Yield current kth largest
        if len(min_heap) == k:
            yield min_heap[0]
```

### 4. **Parallel Top K**
```python
def parallel_top_k(data_chunks, k):
    """Find top K across multiple data chunks in parallel"""
    import heapq
    from concurrent.futures import ThreadPoolExecutor
    
    def find_local_top_k(chunk):
        """Find top k in single chunk"""
        return top_k_largest(chunk, k)
    
    # Process chunks in parallel
    with ThreadPoolExecutor() as executor:
        local_results = list(executor.map(find_local_top_k, data_chunks))
    
    # Merge results
    all_candidates = []
    for result in local_results:
        all_candidates.extend(result)
    
    # Find global top k
    return top_k_largest(all_candidates, k)
```

## Time and Space Complexity Analysis

| Algorithm | Time Complexity | Space Complexity | Use Case |
|-----------|----------------|------------------|----------|
| Heap-based | O(n log k) | O(k) | General purpose, streaming |
| Quick Select | O(n) average, O(n²) worst | O(1) | One-time kth element |
| Full Sort | O(n log n) | O(1) additional | k close to n |
| Bucket Sort | O(n) | O(n) | Limited range values |
| Counting Sort | O(n + r) | O(r) | Small integer range |

## Tips for Success

1. **Choose Right Heap Type**: Min heap for k largest, max heap for k smallest
2. **Consider k vs n**: When k is large relative to n, consider alternatives
3. **Handle Duplicates**: Decide on tie-breaking rules early
4. **Memory Constraints**: Use streaming approaches for large datasets
5. **Custom Comparisons**: Use key functions or custom objects for complex criteria
6. **Early Termination**: Stop processing when result is determined
7. **Batch vs Streaming**: Choose based on data arrival pattern
8. **Validation**: Always validate k is within reasonable bounds

## When NOT to Use Top K Pattern

- **Pre-sorted Data**: Just take first/last k elements
- **k = 1**: Simple linear scan is more efficient
- **k = n**: Full sort might be better
- **Memory Critical**: When heap overhead is too expensive
- **Real-time Critical**: When heap operations are too slow

## Conclusion

The Top K Elements pattern is essential for:
- Efficient selection without full sorting
- Real-time analytics and monitoring
- Resource optimization and prioritization
- Performance analysis and ranking
- Recommendation systems
- Data stream processing

Master this pattern by understanding when to use heaps vs alternatives, choosing the right heap type, and optimizing for your specific constraints. The key is recognizing when you need a subset of elements based on some ranking criteria, rather than processing all elements or finding just one extreme value.
