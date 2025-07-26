# Knowing What to Track Pattern

## Pattern Overview

The **Knowing What to Track** pattern involves counting the occurrences of elements in a given data structure and using this frequency information to solve problems efficiently. This pattern is fundamental for frequency analysis, pattern recognition, and decision-making based on element occurrences.

## When to Use This Pattern

Your problem matches this pattern if **any** of these conditions is fulfilled:

### ✅ Use Knowing What to Track When:

1. **Frequency tracking**: If the problem involves counting the frequencies of elements in a dataset, either individually or in combinations.

2. **Pattern recognition**: Look for patterns in the data where certain elements or combinations of elements repeat frequently, indicating a potential use for frequency counting.

3. **Fixed set of possibilities**: The problem requires choosing the output from a fixed set of possibilities: Yes/No, True/False, Valid/Invalid, Player 1/Player 2.

### ❌ When This Pattern Might Not Be Optimal:

- Problems requiring sorted order without frequency considerations
- Simple sequential processing without counting needs
- Memory-constrained environments with large datasets
- Real-time streaming where maintaining counts is impractical

## Core Phases of the Pattern

### 1. **Counting Phase**
Iterate through the elements of the data structure and count the frequency of each element using:
- **Hash maps/dictionaries**: Most common, flexible for any data type
- **Arrays**: When elements are integers within a known range
- **Simple variables**: For binary or limited element types

### 2. **Utilization Phase**
Use the frequency information to solve specific problems:
- Find the most frequent element
- Identify elements that occur only once
- Check if two arrays are permutations of each other
- Determine game winners
- Make data-driven decisions

## Implementation Templates

### Basic Frequency Counting Template
```python
def count_frequencies(arr):
    """Basic frequency counting using hash map"""
    freq_map = {}
    
    # Counting phase
    for element in arr:
        freq_map[element] = freq_map.get(element, 0) + 1
    
    # Utilization phase examples
    most_frequent = max(freq_map, key=freq_map.get)
    unique_elements = [k for k, v in freq_map.items() if v == 1]
    
    return freq_map, most_frequent, unique_elements
```

### Array-Based Frequency Counting Template
```python
def count_frequencies_array(arr, max_value):
    """Frequency counting using array (for integers 0 to max_value)"""
    freq_array = [0] * (max_value + 1)
    
    # Counting phase
    for element in arr:
        if 0 <= element <= max_value:
            freq_array[element] += 1
    
    # Utilization phase
    most_frequent_value = freq_array.index(max(freq_array))
    unique_count = sum(1 for count in freq_array if count == 1)
    
    return freq_array, most_frequent_value, unique_count
```

### Character Frequency Template
```python
def analyze_string_frequency(s):
    """Character frequency analysis for strings"""
    char_count = {}
    
    # Counting phase
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    # Common utilization patterns
    results = {
        'most_frequent_char': max(char_count, key=char_count.get),
        'unique_chars': [c for c, count in char_count.items() if count == 1],
        'has_duplicates': any(count > 1 for count in char_count.values()),
        'is_palindrome_possible': sum(count % 2 for count in char_count.values()) <= 1
    }
    
    return char_count, results
```

### Sliding Window Frequency Template
```python
def sliding_window_frequency(arr, window_size):
    """Track frequencies in sliding window"""
    if len(arr) < window_size:
        return []
    
    window_freq = {}
    results = []
    
    # Initialize first window
    for i in range(window_size):
        element = arr[i]
        window_freq[element] = window_freq.get(element, 0) + 1
    
    # Process first window
    results.append(analyze_window(window_freq))
    
    # Slide the window
    for i in range(window_size, len(arr)):
        # Remove leftmost element
        left_element = arr[i - window_size]
        window_freq[left_element] -= 1
        if window_freq[left_element] == 0:
            del window_freq[left_element]
        
        # Add rightmost element
        right_element = arr[i]
        window_freq[right_element] = window_freq.get(right_element, 0) + 1
        
        # Analyze current window
        results.append(analyze_window(window_freq))
    
    return results

def analyze_window(freq_map):
    """Analyze frequency map for current window"""
    if not freq_map:
        return None
    
    return {
        'most_frequent': max(freq_map, key=freq_map.get),
        'unique_count': len([k for k, v in freq_map.items() if v == 1]),
        'total_unique': len(freq_map)
    }
```

### Majority Element Template
```python
def find_majority_element(arr):
    """Find element that appears more than n/2 times"""
    # Method 1: Using frequency counting
    freq_map = {}
    n = len(arr)
    
    for element in arr:
        freq_map[element] = freq_map.get(element, 0) + 1
        if freq_map[element] > n // 2:
            return element
    
    return None

def boyer_moore_majority(arr):
    """Boyer-Moore Majority Vote Algorithm - O(1) space"""
    candidate = None
    count = 0
    
    # Find candidate
    for element in arr:
        if count == 0:
            candidate = element
            count = 1
        elif element == candidate:
            count += 1
        else:
            count -= 1
    
    # Verify candidate (if majority is guaranteed to exist, skip this)
    if arr.count(candidate) > len(arr) // 2:
        return candidate
    return None
```

## Common Problem Categories

### 1. **Most/Least Frequent Elements**
- Find most frequent element
- Top K frequent elements
- Least frequent element
- Elements with specific frequency

### 2. **Unique Element Problems**
- Find elements appearing once
- Single number problems
- Remove duplicates
- Count unique elements

### 3. **Anagram and Permutation**
- Valid anagram checking
- Group anagrams
- Permutation validation
- String rearrangement

### 4. **Majority/Minority Elements**
- Majority element detection
- Minority element finding
- Voting system simulation
- Consensus problems

### 5. **Pattern Matching**
- Substring pattern matching
- Frequency-based matching
- Template matching
- Data validation

### 6. **Game Theory**
- Winner determination
- Move counting
- Score tracking
- Turn-based analysis

## Advanced Tracking Techniques

### 1. **Multi-Dimensional Frequency Tracking**
```python
class MultiDimensionalTracker:
    def __init__(self):
        self.single_freq = {}      # element -> count
        self.pair_freq = {}        # (elem1, elem2) -> count
        self.sequence_freq = {}    # tuple(sequence) -> count
    
    def track_element(self, element):
        """Track single element frequency"""
        self.single_freq[element] = self.single_freq.get(element, 0) + 1
    
    def track_pair(self, elem1, elem2):
        """Track pair frequency"""
        pair = (elem1, elem2)
        self.pair_freq[pair] = self.pair_freq.get(pair, 0) + 1
    
    def track_sequence(self, sequence):
        """Track sequence frequency"""
        seq_tuple = tuple(sequence)
        self.sequence_freq[seq_tuple] = self.sequence_freq.get(seq_tuple, 0) + 1
    
    def process_data(self, data):
        """Process data with multi-dimensional tracking"""
        for i, element in enumerate(data):
            # Track individual elements
            self.track_element(element)
            
            # Track pairs
            if i > 0:
                self.track_pair(data[i-1], element)
            
            # Track sequences of length 3
            if i >= 2:
                self.track_sequence(data[i-2:i+1])
    
    def get_most_frequent_pair(self):
        """Get most frequently occurring pair"""
        if not self.pair_freq:
            return None
        return max(self.pair_freq, key=self.pair_freq.get)
    
    def get_pattern_insights(self):
        """Get insights from tracked patterns"""
        return {
            'total_elements': len(self.single_freq),
            'total_pairs': len(self.pair_freq),
            'total_sequences': len(self.sequence_freq),
            'most_frequent_element': max(self.single_freq, key=self.single_freq.get) if self.single_freq else None,
            'most_frequent_pair': self.get_most_frequent_pair()
        }
```

### 2. **Real-Time Frequency Tracker**
```python
import time
from collections import deque

class RealTimeTracker:
    def __init__(self, window_size_seconds=60):
        self.window_size = window_size_seconds
        self.events = deque()  # (timestamp, element)
        self.current_freq = {}
    
    def _cleanup_old_events(self):
        """Remove events outside the time window"""
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        while self.events and self.events[0][0] < cutoff_time:
            _, old_element = self.events.popleft()
            self.current_freq[old_element] -= 1
            if self.current_freq[old_element] == 0:
                del self.current_freq[old_element]
    
    def track_event(self, element):
        """Track a new event"""
        current_time = time.time()
        
        # Clean up old events
        self._cleanup_old_events()
        
        # Add new event
        self.events.append((current_time, element))
        self.current_freq[element] = self.current_freq.get(element, 0) + 1
    
    def get_current_frequencies(self):
        """Get current frequency snapshot"""
        self._cleanup_old_events()
        return dict(self.current_freq)
    
    def get_most_frequent_now(self):
        """Get currently most frequent element"""
        self._cleanup_old_events()
        if not self.current_freq:
            return None
        return max(self.current_freq, key=self.current_freq.get)
    
    def get_activity_level(self):
        """Get current activity level"""
        self._cleanup_old_events()
        return len(self.events)
```

### 3. **Approximate Frequency Tracking (Count-Min Sketch)**
```python
import hashlib

class CountMinSketch:
    def __init__(self, width=1000, depth=5):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]
        self.hash_functions = [
            lambda x, i=i: int(hashlib.md5(f"{x}_{i}".encode()).hexdigest(), 16) % width
            for i in range(depth)
        ]
    
    def update(self, item, count=1):
        """Update count for an item"""
        for i in range(self.depth):
            col = self.hash_functions[i](item)
            self.table[i][col] += count
    
    def query(self, item):
        """Estimate count for an item"""
        min_count = float('inf')
        for i in range(self.depth):
            col = self.hash_functions[i](item)
            min_count = min(min_count, self.table[i][col])
        return min_count
    
    def heavy_hitters(self, threshold):
        """Find items that likely exceed threshold"""
        # This is a simplified version - real implementation would be more complex
        candidates = set()
        
        # Collect potential heavy hitters from high-count cells
        for i in range(self.depth):
            for j in range(self.width):
                if self.table[i][j] >= threshold:
                    # In practice, you'd need to track actual items
                    # This is simplified for demonstration
                    pass
        
        return candidates
```

## Step-by-Step Problem-Solving Approach

### 1. **Identify What to Track**
- What elements need counting?
- Individual elements or combinations?
- Fixed time window or entire dataset?
- Real-time or batch processing?

### 2. **Choose Tracking Method**
- **Hash Map**: Most flexible, any data type
- **Array**: Integers in known range, faster access
- **Bit Vector**: Boolean tracking, memory efficient
- **Advanced**: Count-Min Sketch for approximate counting

### 3. **Design the Counting Phase**
- Single pass vs multiple passes
- Memory vs time trade-offs
- Handle edge cases (empty data, null values)
- Consider data size and memory constraints

### 4. **Plan the Utilization Phase**
- What decisions based on frequencies?
- Threshold-based logic
- Comparative analysis
- Output format requirements

### 5. **Optimize if Needed**
- Early termination conditions
- Space optimization techniques
- Parallel processing for large datasets
- Approximate algorithms for massive data

## Real-World Applications

### 1. **DNA Sequence Analysis**
Analyze frequency of nucleotides and amino acids:
```python
class DNAAnalyzer:
    def __init__(self):
        self.nucleotide_freq = {}
        self.codon_freq = {}
        self.pattern_freq = {}
    
    def analyze_sequence(self, dna_sequence):
        """Analyze DNA sequence for various patterns"""
        sequence = dna_sequence.upper()
        
        # Track individual nucleotides
        for nucleotide in sequence:
            if nucleotide in 'ATCG':
                self.nucleotide_freq[nucleotide] = self.nucleotide_freq.get(nucleotide, 0) + 1
        
        # Track codons (3-nucleotide sequences)
        for i in range(0, len(sequence) - 2, 3):
            codon = sequence[i:i+3]
            if len(codon) == 3 and all(n in 'ATCG' for n in codon):
                self.codon_freq[codon] = self.codon_freq.get(codon, 0) + 1
        
        # Track specific patterns
        patterns = ['ATG', 'TAA', 'TAG', 'TGA']  # Start and stop codons
        for pattern in patterns:
            count = 0
            for i in range(len(sequence) - len(pattern) + 1):
                if sequence[i:i+len(pattern)] == pattern:
                    count += 1
            self.pattern_freq[pattern] = count
    
    def get_gc_content(self):
        """Calculate GC content percentage"""
        total = sum(self.nucleotide_freq.values())
        gc_count = self.nucleotide_freq.get('G', 0) + self.nucleotide_freq.get('C', 0)
        return (gc_count / total * 100) if total > 0 else 0
    
    def find_most_common_codon(self):
        """Find most frequently occurring codon"""
        if not self.codon_freq:
            return None
        return max(self.codon_freq, key=self.codon_freq.get)
    
    def identify_genetic_variations(self, reference_freq):
        """Compare with reference frequency to identify variations"""
        variations = {}
        for nucleotide in 'ATCG':
            current_freq = self.nucleotide_freq.get(nucleotide, 0)
            reference = reference_freq.get(nucleotide, 0)
            if abs(current_freq - reference) > 0.1 * reference:  # 10% threshold
                variations[nucleotide] = {
                    'current': current_freq,
                    'reference': reference,
                    'variation': current_freq - reference
                }
        return variations

# Usage example
analyzer = DNAAnalyzer()
dna_sample = "ATGCGATCGATCGATCGATAA"
analyzer.analyze_sequence(dna_sample)
print(f"GC Content: {analyzer.get_gc_content():.2f}%")
print(f"Most common codon: {analyzer.find_most_common_codon()}")
```

### 2. **Video Streaming - Continue Watching**
Track viewing patterns to enhance user experience:
```python
class StreamingTracker:
    def __init__(self):
        self.user_watch_history = {}    # user_id -> {show_id -> watch_count}
        self.show_popularity = {}       # show_id -> total_views
        self.recent_activity = {}       # user_id -> [(timestamp, show_id, duration)]
        self.continue_watching = {}     # user_id -> [(show_id, last_position, timestamp)]
    
    def track_viewing_session(self, user_id, show_id, watch_duration, total_duration, timestamp):
        """Track a viewing session"""
        # Update user's watch history
        if user_id not in self.user_watch_history:
            self.user_watch_history[user_id] = {}
        
        self.user_watch_history[user_id][show_id] = self.user_watch_history[user_id].get(show_id, 0) + 1
        
        # Update show popularity
        self.show_popularity[show_id] = self.show_popularity.get(show_id, 0) + 1
        
        # Update recent activity
        if user_id not in self.recent_activity:
            self.recent_activity[user_id] = []
        self.recent_activity[user_id].append((timestamp, show_id, watch_duration))
        
        # Update continue watching if partially watched
        watch_percentage = watch_duration / total_duration
        if 0.1 <= watch_percentage < 0.9:  # 10% to 90% watched
            if user_id not in self.continue_watching:
                self.continue_watching[user_id] = []
            
            # Update or add to continue watching
            updated = False
            for i, (existing_show, _, _) in enumerate(self.continue_watching[user_id]):
                if existing_show == show_id:
                    self.continue_watching[user_id][i] = (show_id, watch_duration, timestamp)
                    updated = True
                    break
            
            if not updated:
                self.continue_watching[user_id].append((show_id, watch_duration, timestamp))
    
    def get_most_watched_show(self, user_id):
        """Get user's most frequently watched show"""
        if user_id not in self.user_watch_history:
            return None
        
        user_history = self.user_watch_history[user_id]
        if not user_history:
            return None
        
        return max(user_history, key=user_history.get)
    
    def get_continue_watching_list(self, user_id, limit=5):
        """Get continue watching recommendations"""
        if user_id not in self.continue_watching:
            return []
        
        # Sort by most recent first
        continue_list = sorted(
            self.continue_watching[user_id],
            key=lambda x: x[2],  # Sort by timestamp
            reverse=True
        )
        
        return continue_list[:limit]
    
    def get_trending_shows(self, limit=10):
        """Get most popular shows overall"""
        if not self.show_popularity:
            return []
        
        sorted_shows = sorted(
            self.show_popularity.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_shows[:limit]
    
    def get_personalized_recommendations(self, user_id, limit=5):
        """Get recommendations based on viewing patterns"""
        if user_id not in self.user_watch_history:
            return []
        
        user_shows = set(self.user_watch_history[user_id].keys())
        
        # Find shows watched by users with similar preferences
        similar_shows = {}
        for other_user, other_history in self.user_watch_history.items():
            if other_user == user_id:
                continue
            
            # Check for overlap in viewing history
            other_shows = set(other_history.keys())
            overlap = len(user_shows & other_shows)
            
            if overlap >= 2:  # Similar taste threshold
                for show in other_shows - user_shows:
                    similar_shows[show] = similar_shows.get(show, 0) + other_history[show]
        
        # Sort by frequency and return top recommendations
        recommendations = sorted(
            similar_shows.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return recommendations[:limit]

# Usage example
tracker = StreamingTracker()
import time

# Simulate viewing sessions
current_time = time.time()
tracker.track_viewing_session("user1", "show_friends", 1200, 1500, current_time)
tracker.track_viewing_session("user1", "show_office", 2000, 2100, current_time + 3600)
tracker.track_viewing_session("user1", "show_friends", 300, 1500, current_time + 7200)

# Get recommendations
most_watched = tracker.get_most_watched_show("user1")
continue_watching = tracker.get_continue_watching_list("user1")
print(f"Most watched: {most_watched}")
print(f"Continue watching: {continue_watching}")
```

### 3. **E-commerce - Product Recommendations**
Track items frequently viewed or purchased together:
```python
class EcommerceTracker:
    def __init__(self):
        self.view_frequency = {}           # product_id -> view_count
        self.purchase_frequency = {}       # product_id -> purchase_count
        self.co_viewed = {}               # (product1, product2) -> co_view_count
        self.co_purchased = {}            # (product1, product2) -> co_purchase_count
        self.user_sessions = {}           # user_id -> [product_ids in session]
        self.category_preferences = {}    # user_id -> {category -> preference_score}
    
    def track_product_view(self, user_id, product_id, category, session_id):
        """Track when a user views a product"""
        # Update view frequency
        self.view_frequency[product_id] = self.view_frequency.get(product_id, 0) + 1
        
        # Track user session
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {}
        if session_id not in self.user_sessions[user_id]:
            self.user_sessions[user_id][session_id] = []
        
        session_products = self.user_sessions[user_id][session_id]
        session_products.append(product_id)
        
        # Update co-viewed products
        for other_product in session_products[:-1]:  # Exclude current product
            pair = tuple(sorted([product_id, other_product]))
            self.co_viewed[pair] = self.co_viewed.get(pair, 0) + 1
        
        # Update category preferences
        if user_id not in self.category_preferences:
            self.category_preferences[user_id] = {}
        self.category_preferences[user_id][category] = self.category_preferences[user_id].get(category, 0) + 1
    
    def track_product_purchase(self, user_id, product_ids, transaction_id):
        """Track when a user purchases products"""
        for product_id in product_ids:
            self.purchase_frequency[product_id] = self.purchase_frequency.get(product_id, 0) + 1
        
        # Track co-purchased products
        for i, product1 in enumerate(product_ids):
            for product2 in product_ids[i+1:]:
                pair = tuple(sorted([product1, product2]))
                self.co_purchased[pair] = self.co_purchased.get(pair, 0) + 1
    
    def get_frequently_viewed_together(self, product_id, limit=5):
        """Get products frequently viewed with the given product"""
        related_products = {}
        
        for (prod1, prod2), count in self.co_viewed.items():
            if prod1 == product_id:
                related_products[prod2] = count
            elif prod2 == product_id:
                related_products[prod1] = count
        
        # Sort by frequency
        sorted_products = sorted(
            related_products.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_products[:limit]
    
    def get_frequently_bought_together(self, product_id, limit=5):
        """Get products frequently purchased with the given product"""
        related_products = {}
        
        for (prod1, prod2), count in self.co_purchased.items():
            if prod1 == product_id:
                related_products[prod2] = count
            elif prod2 == product_id:
                related_products[prod1] = count
        
        # Sort by frequency
        sorted_products = sorted(
            related_products.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_products[:limit]
    
    def get_trending_products(self, metric='views', limit=10):
        """Get trending products based on views or purchases"""
        if metric == 'views':
            frequency_data = self.view_frequency
        elif metric == 'purchases':
            frequency_data = self.purchase_frequency
        else:
            return []
        
        sorted_products = sorted(
            frequency_data.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_products[:limit]
    
    def get_user_recommendations(self, user_id, limit=5):
        """Get personalized recommendations for user"""
        if user_id not in self.category_preferences:
            return self.get_trending_products(limit=limit)
        
        # Get user's preferred categories
        user_prefs = self.category_preferences[user_id]
        top_categories = sorted(user_prefs.items(), key=lambda x: x[1], reverse=True)
        
        # This would be enhanced with actual product-category mapping
        # For now, return trending products
        return self.get_trending_products(limit=limit)
    
    def analyze_purchase_patterns(self):
        """Analyze overall purchase patterns"""
        total_views = sum(self.view_frequency.values())
        total_purchases = sum(self.purchase_frequency.values())
        
        # Calculate conversion rate per product
        conversion_rates = {}
        for product_id in self.view_frequency:
            views = self.view_frequency[product_id]
            purchases = self.purchase_frequency.get(product_id, 0)
            conversion_rates[product_id] = purchases / views if views > 0 else 0
        
        return {
            'total_views': total_views,
            'total_purchases': total_purchases,
            'overall_conversion_rate': total_purchases / total_views if total_views > 0 else 0,
            'top_converting_products': sorted(
                conversion_rates.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

# Usage example
tracker = EcommerceTracker()

# Simulate user activity
tracker.track_product_view("user1", "laptop_001", "Electronics", "session_1")
tracker.track_product_view("user1", "mouse_001", "Electronics", "session_1")
tracker.track_product_view("user1", "keyboard_001", "Electronics", "session_1")

tracker.track_product_purchase("user1", ["laptop_001", "mouse_001"], "transaction_1")

# Get recommendations
viewed_together = tracker.get_frequently_viewed_together("laptop_001")
bought_together = tracker.get_frequently_bought_together("laptop_001")
trending = tracker.get_trending_products()

print(f"Frequently viewed with laptop: {viewed_together}")
print(f"Frequently bought with laptop: {bought_together}")
print(f"Trending products: {trending}")
```

### 4. **Clickstream Analysis**
Analyze web user behavior through frequency tracking:
```python
class ClickstreamAnalyzer:
    def __init__(self):
        self.page_views = {}              # page_url -> view_count
        self.user_journeys = {}           # user_id -> [page_sequence]
        self.page_transitions = {}        # (from_page, to_page) -> transition_count
        self.session_data = {}            # session_id -> {pages: [], duration: int, user_id: str}
        self.conversion_funnels = {}      # funnel_name -> {step: page, conversions: count}
        self.bounce_rate_data = {}        # page_url -> {sessions: int, bounces: int}
    
    def track_page_visit(self, user_id, session_id, page_url, timestamp, duration=None):
        """Track a page visit event"""
        # Update page view count
        self.page_views[page_url] = self.page_views.get(page_url, 0) + 1
        
        # Update user journey
        if user_id not in self.user_journeys:
            self.user_journeys[user_id] = []
        self.user_journeys[user_id].append((timestamp, page_url))
        
        # Update session data
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                'pages': [],
                'duration': 0,
                'user_id': user_id,
                'start_time': timestamp
            }
        
        session = self.session_data[session_id]
        session['pages'].append(page_url)
        if duration:
            session['duration'] += duration
        
        # Track page transitions
        if len(session['pages']) > 1:
            from_page = session['pages'][-2]
            to_page = page_url
            transition = (from_page, to_page)
            self.page_transitions[transition] = self.page_transitions.get(transition, 0) + 1
        
        # Update bounce rate data
        if page_url not in self.bounce_rate_data:
            self.bounce_rate_data[page_url] = {'sessions': 0, 'bounces': 0}
        
        if len(session['pages']) == 1:  # First page in session
            self.bounce_rate_data[page_url]['sessions'] += 1
    
    def finalize_session(self, session_id):
        """Mark session as complete and calculate bounce rate"""
        if session_id in self.session_data:
            session = self.session_data[session_id]
            if len(session['pages']) == 1:  # Single page session = bounce
                first_page = session['pages'][0]
                self.bounce_rate_data[first_page]['bounces'] += 1
    
    def get_most_popular_pages(self, limit=10):
        """Get most visited pages"""
        sorted_pages = sorted(
            self.page_views.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_pages[:limit]
    
    def get_common_user_paths(self, min_frequency=5, path_length=3):
        """Find common user journey patterns"""
        path_frequency = {}
        
        for user_id, journey in self.user_journeys.items():
            # Sort by timestamp
            sorted_journey = sorted(journey, key=lambda x: x[0])
            pages = [page for _, page in sorted_journey]
            
            # Extract paths of specified length
            for i in range(len(pages) - path_length + 1):
                path = tuple(pages[i:i + path_length])
                path_frequency[path] = path_frequency.get(path, 0) + 1
        
        # Filter by minimum frequency
        common_paths = {path: freq for path, freq in path_frequency.items() 
                       if freq >= min_frequency}
        
        return sorted(common_paths.items(), key=lambda x: x[1], reverse=True)
    
    def analyze_page_flow(self, start_page):
        """Analyze flow from a specific page"""
        outgoing_flows = {}
        total_exits = 0
        
        for (from_page, to_page), count in self.page_transitions.items():
            if from_page == start_page:
                outgoing_flows[to_page] = count
                total_exits += count
        
        # Calculate flow percentages
        flow_percentages = {}
        for to_page, count in outgoing_flows.items():
            flow_percentages[to_page] = (count / total_exits * 100) if total_exits > 0 else 0
        
        return {
            'total_exits': total_exits,
            'destinations': sorted(flow_percentages.items(), key=lambda x: x[1], reverse=True)
        }
    
    def calculate_bounce_rates(self):
        """Calculate bounce rate for each page"""
        bounce_rates = {}
        
        for page, data in self.bounce_rate_data.items():
            if data['sessions'] > 0:
                bounce_rates[page] = data['bounces'] / data['sessions'] * 100
            else:
                bounce_rates[page] = 0
        
        return sorted(bounce_rates.items(), key=lambda x: x[1], reverse=True)
    
    def setup_conversion_funnel(self, funnel_name, pages):
        """Setup a conversion funnel to track"""
        self.conversion_funnels[funnel_name] = {
            'steps': pages,
            'step_counts': {page: 0 for page in pages},
            'conversions': []
        }
    
    def analyze_conversion_funnel(self, funnel_name):
        """Analyze conversion rates through a funnel"""
        if funnel_name not in self.conversion_funnels:
            return None
        
        funnel = self.conversion_funnels[funnel_name]
        steps = funnel['steps']
        step_counts = {page: 0 for page in steps}
        
        # Count users who visited each step
        for user_id, journey in self.user_journeys.items():
            pages_visited = {page for _, page in journey}
            for page in steps:
                if page in pages_visited:
                    step_counts[page] += 1
        
        # Calculate conversion rates
        conversion_rates = []
        total_entered = step_counts.get(steps[0], 0)
        
        for i, page in enumerate(steps):
            count = step_counts[page]
            if i == 0:
                rate = 100.0  # First step is 100%
            else:
                rate = (count / total_entered * 100) if total_entered > 0 else 0
            
            conversion_rates.append({
                'step': i + 1,
                'page': page,
                'users': count,
                'conversion_rate': rate
            })
        
        return conversion_rates
    
    def get_user_behavior_insights(self):
        """Get overall user behavior insights"""
        total_sessions = len(self.session_data)
        total_page_views = sum(self.page_views.values())
        
        # Calculate average pages per session
        pages_per_session = []
        for session in self.session_data.values():
            pages_per_session.append(len(session['pages']))
        
        avg_pages_per_session = sum(pages_per_session) / len(pages_per_session) if pages_per_session else 0
        
        # Calculate average session duration
        session_durations = [s['duration'] for s in self.session_data.values() if s['duration'] > 0]
        avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        
        return {
            'total_sessions': total_sessions,
            'total_page_views': total_page_views,
            'avg_pages_per_session': avg_pages_per_session,
            'avg_session_duration_seconds': avg_session_duration,
            'most_popular_pages': self.get_most_popular_pages(5),
            'highest_bounce_rates': self.calculate_bounce_rates()[:5]
        }

# Usage example
analyzer = ClickstreamAnalyzer()

# Simulate user activity
import time
current_time = time.time()

analyzer.track_page_visit("user1", "session1", "/home", current_time, 30)
analyzer.track_page_visit("user1", "session1", "/products", current_time + 30, 45)
analyzer.track_page_visit("user1", "session1", "/product/123", current_time + 75, 60)
analyzer.track_page_visit("user1", "session1", "/cart", current_time + 135, 20)

analyzer.finalize_session("session1")

# Setup and analyze conversion funnel
analyzer.setup_conversion_funnel("purchase_funnel", ["/home", "/products", "/cart", "/checkout", "/confirmation"])

# Get insights
popular_pages = analyzer.get_most_popular_pages()
user_paths = analyzer.get_common_user_paths(min_frequency=1)
page_flow = analyzer.analyze_page_flow("/products")
insights = analyzer.get_user_behavior_insights()

print(f"Popular pages: {popular_pages}")
print(f"Common paths: {user_paths}")
print(f"Flow from products: {page_flow}")
print(f"Overall insights: {insights}")
```

## Practice Problems

### Beginner Level
1. **Majority Element** - Element appearing more than n/2 times
2. **Valid Anagram** - Check if two strings are anagrams
3. **Single Number** - Find element appearing once while others appear twice
4. **Most Common Word** - Find most frequent word in text
5. **Contains Duplicate** - Check if array has duplicates

### Intermediate Level
6. **Top K Frequent Elements** - Find K most frequent elements
7. **Group Anagrams** - Group strings that are anagrams
8. **Frequency Sort** - Sort characters by frequency
9. **Find All Anagrams** - Find all anagram substrings
10. **Subarray Sum Equals K** - Count subarrays with specific sum
11. **Longest Substring Without Repeating** - Track character frequencies in window
12. **Minimum Window Substring** - Find minimum window containing all characters

### Advanced Level
13. **Sliding Window Maximum** - Track frequencies in sliding window
14. **Data Stream as Disjoint Intervals** - Track number ranges
15. **Design Twitter** - Track user interactions and frequencies
16. **LFU Cache** - Implement Least Frequently Used cache
17. **Random Pick with Weight** - Weighted random selection based on frequencies
18. **Design Search Autocomplete** - Track search query frequencies

## Common Patterns and Optimizations

### 1. **Early Termination Patterns**
```python
def has_majority_element(arr):
    """Check if majority element exists without finding it"""
    freq_map = {}
    threshold = len(arr) // 2
    
    for element in arr:
        freq_map[element] = freq_map.get(element, 0) + 1
        if freq_map[element] > threshold:
            return True  # Early termination
    
    return False
```

### 2. **Space-Optimized Counting**
```python
def count_with_limited_space(arr, max_unique=100):
    """Use array instead of hash map when range is known"""
    if not arr:
        return {}
    
    min_val, max_val = min(arr), max(arr)
    
    if max_val - min_val + 1 <= max_unique:
        # Use array-based counting
        counts = [0] * (max_val - min_val + 1)
        for num in arr:
            counts[num - min_val] += 1
        
        # Convert back to dictionary
        result = {}
        for i, count in enumerate(counts):
            if count > 0:
                result[i + min_val] = count
        return result
    else:
        # Fall back to hash map
        freq_map = {}
        for num in arr:
            freq_map[num] = freq_map.get(num, 0) + 1
        return freq_map
```

### 3. **Streaming Frequency Tracking**
```python
def track_top_k_streaming(stream, k):
    """Track top K elements in a data stream"""
    import heapq
    from collections import defaultdict
    
    freq_map = defaultdict(int)
    min_heap = []  # (frequency, element)
    
    for element in stream:
        # Update frequency
        old_freq = freq_map[element]
        freq_map[element] += 1
        new_freq = freq_map[element]
        
        # Update heap
        if len(min_heap) < k:
            heapq.heappush(min_heap, (new_freq, element))
        elif new_freq > min_heap[0][0]:
            heapq.heapreplace(min_heap, (new_freq, element))
        
        # Yield current top K
        yield sorted(min_heap, key=lambda x: x[0], reverse=True)
```

## Time and Space Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Basic Frequency Count | O(n) | O(k) | k = unique elements |
| Array-based Count | O(n) | O(r) | r = range of values |
| Streaming Top-K | O(n log k) | O(k) | Using heap for top K |
| Sliding Window | O(n) | O(w) | w = window size |
| Most Frequent | O(n) | O(k) | Single pass counting |
| Majority Element (Boyer-Moore) | O(n) | O(1) | Constant space algorithm |

## Tips for Success

1. **Choose Right Data Structure**: Hash map vs array vs specialized structures
2. **Consider Space Constraints**: Array for limited range, hash map for flexibility
3. **Think About Updates**: Static vs dynamic frequency tracking
4. **Handle Edge Cases**: Empty data, single elements, all same elements
5. **Optimize for Common Cases**: Early termination, threshold checking
6. **Consider Approximation**: For massive datasets, use probabilistic structures
7. **Profile Memory Usage**: Monitor frequency map growth
8. **Use Built-in Tools**: Collections.Counter in Python, similar in other languages

## When NOT to Use This Pattern

- **Sequential Processing Only**: When you don't need frequency information
- **Real-time Constraints**: When counting overhead is too expensive
- **Ordered Requirements**: When you need sorted data without frequency considerations
- **Memory Critical**: When frequency maps would exceed memory limits

## Conclusion

The "Knowing What to Track" pattern is essential for:
- Data analysis and insights
- Pattern recognition and detection
- Optimization based on frequency
- Decision making with statistical evidence
- User behavior analysis
- Performance monitoring and tuning

Master this pattern by understanding what elements to count, choosing appropriate data structures, and efficiently utilizing frequency information to solve problems. The key is recognizing when frequency-based analysis provides valuable insights for your specific problem domain.
