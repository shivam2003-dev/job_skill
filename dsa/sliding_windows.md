# Sliding Windows Pattern

## Pattern Overview

The **Sliding Windows** pattern is a technique for efficiently processing contiguous subarrays or substrings by maintaining a window that slides across the input data. Instead of recalculating results from scratch for each position, this pattern leverages overlapping computations to achieve optimal time complexity. The window can have fixed or variable size depending on the problem requirements.

## When to Use This Pattern

Your problem matches this pattern if **all** of these conditions are fulfilled:

### ✅ Use Sliding Windows When:

1. **Contiguous data**: The input data is stored in a contiguous manner, such as an array or string.

2. **Processing subsets of elements**: The problem requires repeated computations on a contiguous subset of data elements (a subarray or a substring), such that the window moves across the input array from one end to the other. The size of the window may be fixed or variable, depending on the requirements of the problem.

3. **Efficient computation time complexity**: The computations performed every time the window moves take constant or very small time.

### ❌ Don't Use Sliding Windows When:

1. **Non-contiguous data**: Elements to be processed are scattered or require random access
2. **Single pass processing**: Problem only requires one iteration without window-based computation
3. **Complex nested loops needed**: Window pattern doesn't simplify the core algorithm
4. **Tree or graph structures**: Data isn't linear or array-like

## Core Concepts

### Window Types

**Fixed-Size Window**: Window maintains constant size as it slides
- Use case: Maximum sum of k consecutive elements
- Technique: Add new element, remove old element

**Variable-Size Window**: Window size changes based on conditions
- Use case: Longest substring with unique characters
- Technique: Expand window when condition met, shrink when violated

**Multiple Windows**: Track multiple windows simultaneously
- Use case: Comparing patterns in different parts of data
- Technique: Maintain separate pointers for each window

### Core Mechanics

**Expansion**: Increase window size by moving right pointer 
**Contraction**: Decrease window size by moving left pointer
**Sliding**: Move entire window maintaining size
**State Tracking**: Maintain aggregated information about current window

## Essential Implementation Templates

### Fixed-Size Sliding Window
```python
def fixed_window_template(arr, k):
    """Template for fixed-size sliding window problems"""
    if len(arr) < k:
        return []
    
    # Initialize window state
    window_sum = sum(arr[:k])
    result = [window_sum]
    
    # Slide the window
    for i in range(k, len(arr)):
        # Remove leftmost element, add rightmost element
        window_sum = window_sum - arr[i - k] + arr[i]
        result.append(window_sum)
    
    return result
```

### Variable-Size Sliding Window
```python
def variable_window_template(arr, condition_func):
    """Template for variable-size sliding window problems"""
    left = 0
    result = []
    window_state = {}  # Track window properties
    
    for right in range(len(arr)):
        # Expand window by including arr[right]
        # Update window_state
        
        # Shrink window while condition is violated
        while condition_violated(window_state):
            # Remove arr[left] from window_state
            left += 1
        
        # Process current valid window
        if condition_met(window_state):
            result.append(calculate_result(left, right, window_state))
    
    return result
```

### Two-Pointer Sliding Window
```python
def two_pointer_window(arr, target):
    """Template for two-pointer sliding window approach"""
    left = right = 0
    current_sum = 0
    
    while right < len(arr):
        # Expand window
        current_sum += arr[right]
        
        # Shrink window if needed
        while current_sum > target and left <= right:
            current_sum -= arr[left]
            left += 1
        
        # Check if current window meets criteria
        if current_sum == target:
            return [left, right]
        
        right += 1
    
    return [-1, -1]  # Not found
```

## Problem Categories

### 1. **Fixed-Size Window Problems**
- Maximum/minimum sum of k consecutive elements
- Average of k consecutive elements
- Product of k consecutive elements
- Count of specific patterns in windows

### 2. **Variable-Size Window Problems**
- Longest substring with unique characters
- Minimum window substring containing pattern
- Longest subarray with sum ≤ target
- Maximum length subarray with condition

### 3. **Multiple Window Problems**
- Comparing patterns across different positions
- Finding anagrams in string
- Multiple sliding maximums
- Parallel window processing

### 4. **String Processing Windows**
- Longest palindromic substring
- Character frequency windows
- Pattern matching with sliding
- DNA sequence analysis

### 5. **Optimization Windows**
- Maximum profit in stock trading
- Resource allocation optimization
- Time-based analysis windows
- Performance monitoring windows

## Real-World Applications

### 1. **Telecommunications Network Analysis**

**Business Problem**: Find the maximum number of users connected to a cellular network's base station in every k-millisecond sliding window.

**Technical Challenge**: Network administrators need to monitor peak usage patterns to prevent congestion, plan capacity, and ensure quality of service. Processing connection data in real-time requires efficient algorithms that can handle high-frequency updates.

**Sliding Window Solution**:
```python
class NetworkMonitor:
    def __init__(self, window_size_ms):
        self.window_size = window_size_ms
        self.connection_timestamps = []
        self.max_connections_per_window = []
    
    def add_connection_event(self, timestamp, event_type):
        """Add connection/disconnection event"""
        self.connection_timestamps.append((timestamp, event_type))
    
    def analyze_peak_usage(self, start_time, end_time):
        """Find maximum concurrent connections in each sliding window"""
        events = sorted(self.connection_timestamps)
        results = []
        
        # Fixed-size time window sliding approach
        for window_start in range(start_time, end_time - self.window_size + 1, self.window_size):
            window_end = window_start + self.window_size
            concurrent_connections = self.count_concurrent_in_window(events, window_start, window_end)
            results.append({
                'window_start': window_start,
                'window_end': window_end,
                'max_concurrent': concurrent_connections
            })
        
        return results
    
    def count_concurrent_in_window(self, events, start_time, end_time):
        """Count maximum concurrent connections in time window"""
        active_connections = 0
        max_concurrent = 0
        
        for timestamp, event_type in events:
            if start_time <= timestamp < end_time:
                if event_type == 'connect':
                    active_connections += 1
                    max_concurrent = max(max_concurrent, active_connections)
                elif event_type == 'disconnect':
                    active_connections -= 1
        
        return max_concurrent
```

**Business Impact**: Enables proactive network management, prevents service degradation, optimizes resource allocation, and improves customer satisfaction through better service quality.

### 2. **Video Streaming Analytics**

**Business Problem**: Given a stream of numbers representing the number of buffering events in a given user session, calculate the median number of buffering events in each one-minute interval.

**Technical Challenge**: Streaming platforms need to monitor video quality metrics in real-time to detect issues, optimize CDN performance, and improve user experience. Processing continuous streams of quality metrics requires efficient windowing techniques.

**Sliding Window Solution**:
```python
import heapq
from collections import deque

class StreamingQualityMonitor:
    def __init__(self, window_duration_seconds=60):
        self.window_duration = window_duration_seconds
        self.events = deque()  # (timestamp, buffering_count)
        self.quality_metrics = []
    
    def add_buffering_event(self, timestamp, buffering_count):
        """Add buffering event data point"""
        self.events.append((timestamp, buffering_count))
        
        # Remove events outside current window
        while self.events and self.events[0][0] < timestamp - self.window_duration:
            self.events.popleft()
    
    def calculate_sliding_median(self, events_stream):
        """Calculate median buffering events in sliding windows"""
        results = []
        
        for timestamp, buffering_count in events_stream:
            self.add_buffering_event(timestamp, buffering_count)
            
            # Calculate median for current window
            if len(self.events) >= 10:  # Minimum data points for reliable median
                window_values = [count for _, count in self.events]
                median = self.find_median(window_values)
                results.append({
                    'timestamp': timestamp,
                    'window_median': median,
                    'sample_size': len(window_values)
                })
        
        return results
    
    def find_median(self, values):
        """Find median using two heaps approach"""
        values_sorted = sorted(values)
        n = len(values_sorted)
        if n % 2 == 0:
            return (values_sorted[n//2 - 1] + values_sorted[n//2]) / 2
        else:
            return values_sorted[n//2]
    
    def detect_quality_issues(self, median_threshold=5):
        """Detect periods with excessive buffering"""
        issues = []
        for metric in self.quality_metrics:
            if metric['window_median'] > median_threshold:
                issues.append({
                    'timestamp': metric['timestamp'],
                    'severity': 'high' if metric['window_median'] > median_threshold * 2 else 'medium',
                    'median_buffering': metric['window_median']
                })
        return issues
```

**Business Impact**: Improves video streaming quality, reduces user churn, enables real-time issue detection, and helps optimize content delivery networks for better performance.

### 3. **Social Media Content Mining**

**Business Problem**: Given the lists of topics that two users have posted about, find the shortest sequence of posts by one user that includes all the topics that the other user has posted about.

**Technical Challenge**: Social media platforms need to analyze user interests, recommend connections, and identify content similarities. Finding minimum covering sequences helps in content recommendation, influence analysis, and community detection.

**Sliding Window Solution**:
```python
class SocialMediaAnalyzer:
    def __init__(self):
        self.user_posts = {}  # user_id -> [(timestamp, topics), ...]
        self.topic_relationships = {}
    
    def add_user_post(self, user_id, timestamp, topics):
        """Add a post with associated topics"""
        if user_id not in self.user_posts:
            self.user_posts[user_id] = []
        self.user_posts[user_id].append((timestamp, set(topics)))
    
    def find_shortest_covering_sequence(self, user1_id, user2_id):
        """Find shortest sequence of user1 posts covering all user2 topics"""
        if user1_id not in self.user_posts or user2_id not in self.user_posts:
            return None
        
        # Get all topics posted by user2
        user2_topics = set()
        for _, topics in self.user_posts[user2_id]:
            user2_topics.update(topics)
        
        if not user2_topics:
            return []
        
        # Find minimum window covering all user2 topics
        user1_posts = self.user_posts[user1_id]
        min_window = self.minimum_window_covering_topics(user1_posts, user2_topics)
        
        return min_window
    
    def minimum_window_covering_topics(self, posts, target_topics):
        """Find minimum window of posts covering all target topics"""
        if not target_topics:
            return []
        
        left = 0
        min_length = float('inf')
        min_window = []
        covered_topics = set()
        topic_count = {}
        
        for right in range(len(posts)):
            # Expand window - add topics from current post
            _, current_topics = posts[right]
            for topic in current_topics:
                if topic in target_topics:
                    topic_count[topic] = topic_count.get(topic, 0) + 1
                    if topic_count[topic] == 1:
                        covered_topics.add(topic)
            
            # Shrink window while all topics are covered
            while len(covered_topics) == len(target_topics):
                # Update minimum window if current is smaller
                if right - left + 1 < min_length:
                    min_length = right - left + 1
                    min_window = posts[left:right + 1]
                
                # Remove leftmost post topics
                _, left_topics = posts[left]
                for topic in left_topics:
                    if topic in target_topics:
                        topic_count[topic] -= 1
                        if topic_count[topic] == 0:
                            covered_topics.remove(topic)
                
                left += 1
        
        return min_window
    
    def analyze_user_similarity(self, user1_id, user2_id):
        """Analyze content similarity between users"""
        covering_sequence = self.find_shortest_covering_sequence(user1_id, user2_id)
        reverse_sequence = self.find_shortest_covering_sequence(user2_id, user1_id)
        
        return {
            'user1_covers_user2': len(covering_sequence) if covering_sequence else None,
            'user2_covers_user1': len(reverse_sequence) if reverse_sequence else None,
            'similarity_score': self.calculate_similarity_score(covering_sequence, reverse_sequence)
        }
    
    def calculate_similarity_score(self, seq1, seq2):
        """Calculate similarity score based on covering sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        # Lower covering sequence length indicates higher similarity
        max_possible = max(len(self.user_posts.get(uid, [])) for uid in self.user_posts.keys())
        normalized_score = 1.0 - (min(len(seq1), len(seq2)) / max_possible)
        return max(0.0, min(1.0, normalized_score))
```

**Business Impact**: Enables intelligent content recommendation, improves user engagement, facilitates community building, and helps identify influential users and trending topics.

### 4. **Financial Market Analysis**

**Business Problem**: Analyze stock price movements and trading volumes using sliding windows to detect patterns, calculate moving averages, and identify trading opportunities.

**Sliding Window Solution**:
```python
class MarketAnalyzer:
    def __init__(self):
        self.price_data = []  # (timestamp, price, volume)
        self.indicators = {}
    
    def add_price_data(self, timestamp, price, volume):
        """Add new price/volume data point"""
        self.price_data.append((timestamp, price, volume))
    
    def calculate_moving_averages(self, window_sizes=[5, 10, 20, 50]):
        """Calculate moving averages for different window sizes"""
        results = {}
        
        for window_size in window_sizes:
            ma_values = []
            for i in range(len(self.price_data)):
                if i + 1 >= window_size:
                    window_prices = [self.price_data[j][1] for j in range(i - window_size + 1, i + 1)]
                    ma = sum(window_prices) / window_size
                    ma_values.append((self.price_data[i][0], ma))
            
            results[f'MA_{window_size}'] = ma_values
        
        return results
    
    def detect_trend_changes(self, short_window=5, long_window=20):
        """Detect trend changes using dual moving averages"""
        signals = []
        short_ma = self.calculate_moving_averages([short_window])[f'MA_{short_window}']
        long_ma = self.calculate_moving_averages([long_window])[f'MA_{long_window}']
        
        # Find crossover points
        for i in range(1, min(len(short_ma), len(long_ma))):
            prev_short, prev_long = short_ma[i-1][1], long_ma[i-1][1]
            curr_short, curr_long = short_ma[i][1], long_ma[i][1]
            
            # Golden cross (bullish signal)
            if prev_short <= prev_long and curr_short > curr_long:
                signals.append({
                    'timestamp': short_ma[i][0],
                    'signal': 'BUY',
                    'type': 'golden_cross'
                })
            
            # Death cross (bearish signal)
            elif prev_short >= prev_long and curr_short < curr_long:
                signals.append({
                    'timestamp': short_ma[i][0],
                    'signal': 'SELL',
                    'type': 'death_cross'
                })
        
        return signals
```

**Business Impact**: Enables algorithmic trading strategies, improves investment decision-making, reduces emotional trading bias, and helps identify optimal entry/exit points.

## Advanced Techniques

### 1. **Deque-Based Sliding Window Maximum**
Efficiently maintain maximum/minimum in sliding windows using double-ended queues.

### 2. **Hash Map Window State Tracking**
Use hash maps to track character frequencies or element counts in variable-size windows.

### 3. **Multi-Level Sliding Windows**
Implement nested windows for complex pattern detection and multi-scale analysis.

### 4. **Sliding Window with Preprocessing**
Combine sliding windows with prefix sums or other preprocessing techniques for optimization.

### 5. **Parallel Sliding Windows**
Process multiple windows simultaneously for improved performance on large datasets.

## Performance Characteristics

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Fixed window slide | O(1) per move | O(k) | k is window size |
| Variable window | O(n) total | O(k) | Amortized analysis |
| Window maximum | O(1) per move | O(k) | Using deque |
| String window | O(n) total | O(alphabet) | Character frequency tracking |
| Multiple windows | O(n*w) | O(w*k) | w windows, k window size |

## Common Patterns and Variations

### 1. **Two-Pointer Technique**
- **Use Case**: Variable-size windows with specific sum/condition
- **Technique**: Expand right pointer, contract left pointer
- **Applications**: Subarray sum problems, string patterns

### 2. **Sliding Window Maximum/Minimum**
- **Use Case**: Track extremes in fixed-size windows
- **Technique**: Use deque to maintain potential candidates
- **Applications**: Stock analysis, signal processing

### 3. **Character Frequency Windows**
- **Use Case**: String problems with character constraints
- **Technique**: Hash map to track character counts
- **Applications**: Anagram detection, substring problems

### 4. **Prefix Sum Windows**
- **Use Case**: Range sum queries in sliding windows
- **Technique**: Combine prefix sums with window sliding
- **Applications**: Subarray sum optimization

### 5. **Multiple Criteria Windows**
- **Use Case**: Windows with multiple constraints
- **Technique**: Track multiple conditions simultaneously
- **Applications**: Complex pattern matching

## Practical Problem Examples

### Beginner Level
1. **Maximum Sum Subarray of Size K** - Basic fixed window
2. **Average of Subarrays of Size K** - Fixed window with division
3. **Longest Substring with K Distinct Characters** - Variable window
4. **Fruits into Baskets** - Variable window with constraints

### Intermediate Level
5. **Minimum Window Substring** - Complex variable window
6. **Sliding Window Maximum** - Deque-based optimization
7. **Find All Anagrams in String** - Character frequency windows
8. **Longest Repeating Character Replacement** - Window with character limit
9. **Permutation in String** - Pattern matching with sliding

### Advanced Level
10. **Sliding Window Median** - Two heaps with sliding window
11. **Minimum Number of K Consecutive Bit Flips** - Complex state tracking
12. **Subarrays with K Different Integers** - Multiple window coordination
13. **Count Number of Nice Subarrays** - Transform and slide
14. **Get Equal Substrings Within Budget** - Cost-based sliding window

## Common Pitfalls and Solutions

### 1. **Off-by-One Errors**
- **Problem**: Incorrect window boundary calculations
- **Solution**: Carefully track left and right pointers, test edge cases

### 2. **Window State Management**
- **Problem**: Forgetting to update window state when sliding
- **Solution**: Consistent add/remove operations for window elements

### 3. **Empty Windows**
- **Problem**: Not handling cases where window becomes empty
- **Solution**: Add boundary checks and handle empty window scenarios

### 4. **Infinite Loops**
- **Problem**: Incorrect termination conditions in variable windows
- **Solution**: Ensure progress is made in each iteration

### 5. **Memory Leaks**
- **Problem**: Accumulating data without cleanup in long-running windows
- **Solution**: Proper cleanup of expired window data

## When NOT to Use Sliding Windows

1. **Non-contiguous processing**: When elements to process aren't adjacent
2. **Single-pass algorithms**: When no overlapping computation benefits exist
3. **Random access patterns**: When processing order isn't sequential
4. **Tree/graph traversal**: When data structure isn't linear
5. **Simple iteration suffices**: When sliding window doesn't reduce complexity

## Tips for Success

1. **Identify window type early**: Fixed vs variable size determines approach
2. **Plan window state carefully**: What information needs tracking during slide
3. **Handle edge cases**: Empty arrays, single elements, window larger than array
4. **Optimize data structures**: Choose appropriate structures for window state
5. **Test boundary conditions**: First/last windows, minimum/maximum sizes
6. **Consider amortized analysis**: Variable windows may have O(n) total time
7. **Use helper functions**: Separate window logic from business logic

## Conclusion

The Sliding Windows pattern is essential for:
- Efficient subarray and substring processing
- Real-time data stream analysis
- Time series and signal processing
- Performance monitoring and analytics
- Pattern recognition and matching
- Optimization problems with locality

Master this pattern by understanding when overlapping computations can be leveraged, choosing appropriate window types, and practicing with problems that involve contiguous data processing. The key insight is that many problems involving sequences can be optimized by avoiding redundant calculations through clever window management.
