# Hash Map Pattern

## Pattern Overview

The **Hash Map** pattern is a fundamental technique that uses key-value pairs to solve problems requiring fast data access and storage of relationships between different sets of data. It leverages hashing for O(1) average-case lookups, insertions, and deletions.

## When to Use This Pattern

Your problem matches this pattern if **both** conditions are fulfilled:

### ✅ Use Hash Map When:

1. **Data access**: When we require repeated fast access to data during the execution of the algorithm.

2. **Pair-wise relation**: We need to store the relationship between two sets of data in order to compute the required result. This is achieved through the mechanism of a key-value pair, where we store the hashed version of the key to enable fast look-ups.

### ❌ When Hash Map Might Not Be Optimal:

- Ordered data requirements (use TreeMap/sorted structures)
- Memory constraints with large datasets
- When keys don't hash well (poor distribution)
- Simple sequential access patterns

## Hash Map Fundamentals

### Core Concepts
- **Hashing**: Convert keys to array indices using hash function
- **Key-Value Pairs**: Associate data with unique identifiers
- **Collision Handling**: Manage multiple keys hashing to same index
- **Load Factor**: Ratio of stored elements to array size

### Basic Hash Map Operations
```python
# Built-in dictionary in Python
hash_map = {}

# Basic operations
hash_map[key] = value      # Insert/Update - O(1) average
value = hash_map[key]      # Access - O(1) average
del hash_map[key]          # Delete - O(1) average
exists = key in hash_map   # Check existence - O(1) average

# Safe operations
value = hash_map.get(key, default_value)  # Safe access
hash_map.setdefault(key, default_value)   # Insert if not exists
```

## Implementation Templates

### Basic Hash Map Usage Template
```python
def hash_map_solution(data):
    # Initialize hash map
    hash_map = {}
    result = []
    
    for item in data:
        # Check if key exists
        if item.key in hash_map:
            # Process existing key
            hash_map[item.key] += item.value
        else:
            # Add new key
            hash_map[item.key] = item.value
    
    # Process results
    for key, value in hash_map.items():
        if meets_condition(value):
            result.append(key)
    
    return result
```

### Frequency Counter Template
```python
def frequency_counter(arr):
    freq_map = {}
    
    # Count frequencies
    for element in arr:
        freq_map[element] = freq_map.get(element, 0) + 1
    
    # Alternative using defaultdict
    from collections import defaultdict
    freq_map_alt = defaultdict(int)
    for element in arr:
        freq_map_alt[element] += 1
    
    return freq_map
```

### Two Sum Pattern Template
```python
def two_sum(nums, target):
    num_map = {}  # value -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in num_map:
            return [num_map[complement], i]
        
        num_map[num] = i
    
    return []
```

### Group Anagrams Template
```python
def group_anagrams(strs):
    from collections import defaultdict
    
    anagram_map = defaultdict(list)
    
    for s in strs:
        # Create key (sorted characters)
        key = ''.join(sorted(s))
        anagram_map[key].append(s)
    
    return list(anagram_map.values())
```

### Sliding Window with Hash Map Template
```python
def sliding_window_hash(s, pattern):
    if len(pattern) > len(s):
        return []
    
    pattern_map = {}
    window_map = {}
    
    # Build pattern frequency map
    for char in pattern:
        pattern_map[char] = pattern_map.get(char, 0) + 1
    
    result = []
    left = 0
    
    for right in range(len(s)):
        # Expand window
        char = s[right]
        window_map[char] = window_map.get(char, 0) + 1
        
        # Shrink window if needed
        if right - left + 1 > len(pattern):
            left_char = s[left]
            window_map[left_char] -= 1
            if window_map[left_char] == 0:
                del window_map[left_char]
            left += 1
        
        # Check if window matches pattern
        if window_map == pattern_map:
            result.append(left)
    
    return result
```

## Common Problem Categories

### 1. **Frequency and Counting**
- Character/element frequency counting
- Most frequent elements
- Frequency-based sorting
- Duplicate detection

### 2. **Two-Pointer with Hash Map**
- Two Sum, Three Sum variants
- Pair sum problems
- Complement finding
- Target sum combinations

### 3. **Anagram and String Problems**
- Group anagrams
- Valid anagram checking
- String pattern matching
- Character rearrangement

### 4. **Subarray/Substring Problems**
- Subarray sum equals K
- Longest substring problems
- Sliding window optimizations
- Prefix sum applications

### 5. **Graph and Tree Problems**
- Node mapping and relationships
- Parent-child mappings
- Component grouping
- Path and cycle detection

### 6. **Design Problems**
- LRU Cache implementation
- Design data structures
- System design with caching
- Custom hash table design

## Advanced Hash Map Techniques

### 1. **Custom Hash Map Implementation**
```python
class HashMapCustom:
    def __init__(self, initial_capacity=16):
        self.capacity = initial_capacity
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        self.load_factor_threshold = 0.75
    
    def _hash(self, key):
        return hash(key) % self.capacity
    
    def _resize(self):
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]
        
        # Rehash all elements
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    def put(self, key, value):
        if self.size >= self.capacity * self.load_factor_threshold:
            self._resize()
        
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self.size += 1
    
    def get(self, key):
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for k, v in bucket:
            if k == key:
                return v
        
        raise KeyError(key)
    
    def remove(self, key):
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]
        
        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return v
        
        raise KeyError(key)
```

### 2. **LRU Cache with Hash Map**
```python
class LRUCache:
    class Node:
        def __init__(self, key=0, value=0):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> node
        
        # Create dummy head and tail
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_to_head(self, node):
        """Add node right after head"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _remove_node(self, node):
        """Remove an existing node"""
        node.prev.next = node.next
        node.next.prev = node.prev
    
    def _move_to_head(self, node):
        """Move node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_to_head(node)
    
    def _pop_tail(self):
        """Remove last node"""
        last_node = self.tail.prev
        self._remove_node(last_node)
        return last_node
    
    def get(self, key):
        node = self.cache.get(key)
        if node:
            # Move to head (recently used)
            self._move_to_head(node)
            return node.value
        return -1
    
    def put(self, key, value):
        node = self.cache.get(key)
        
        if node:
            # Update existing node
            node.value = value
            self._move_to_head(node)
        else:
            # Add new node
            new_node = self.Node(key, value)
            
            if len(self.cache) >= self.capacity:
                # Remove least recently used
                tail = self._pop_tail()
                del self.cache[tail.key]
            
            self.cache[key] = new_node
            self._add_to_head(new_node)
```

### 3. **Multi-Value Hash Map**
```python
from collections import defaultdict

class MultiValueHashMap:
    def __init__(self):
        self.map = defaultdict(list)
    
    def put(self, key, value):
        self.map[key].append(value)
    
    def get(self, key):
        return self.map[key]
    
    def remove(self, key, value):
        if key in self.map:
            self.map[key].remove(value)
            if not self.map[key]:
                del self.map[key]
    
    def get_all_values(self):
        all_values = []
        for values in self.map.values():
            all_values.extend(values)
        return all_values
```

### 4. **Trie with Hash Map**
```python
class TrieNode:
    def __init__(self):
        self.children = {}  # char -> TrieNode
        self.is_end = False
        self.word = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word
    
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end
    
    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def find_words_with_prefix(self, prefix):
        words = []
        node = self.root
        
        # Navigate to prefix end
        for char in prefix:
            if char not in node.children:
                return words
            node = node.children[char]
        
        # DFS to find all words
        def dfs(node):
            if node.is_end:
                words.append(node.word)
            for child in node.children.values():
                dfs(child)
        
        dfs(node)
        return words
```

## Step-by-Step Problem-Solving Approach

### 1. **Identify Key-Value Relationship**
- What serves as the key?
- What information needs to be stored as value?
- Is there a natural mapping between data elements?

### 2. **Choose the Right Hash Map Type**
- **Basic dict**: Simple key-value pairs
- **defaultdict**: Automatic default values
- **Counter**: Frequency counting
- **Custom**: Specific requirements

### 3. **Design the Algorithm**
- When to insert/update values?
- When to lookup values?
- How to handle missing keys?
- What post-processing is needed?

### 4. **Handle Edge Cases**
- Empty input
- Duplicate keys
- Large datasets
- Hash collisions

### 5. **Optimize if Needed**
- Space complexity considerations
- Time complexity optimizations
- Memory usage patterns
- Custom hashing strategies

## Real-World Applications

### 1. **Telecommunications - Phone Book**
Store person names as keys and phone numbers as values:
```python
class PhoneBook:
    def __init__(self):
        self.contacts = {}  # name -> phone_number
        self.reverse_lookup = {}  # phone_number -> name
    
    def add_contact(self, name, phone_number):
        # Handle updates
        if name in self.contacts:
            old_number = self.contacts[name]
            del self.reverse_lookup[old_number]
        
        self.contacts[name] = phone_number
        self.reverse_lookup[phone_number] = name
    
    def get_number(self, name):
        return self.contacts.get(name, "Contact not found")
    
    def get_name(self, phone_number):
        return self.reverse_lookup.get(phone_number, "Number not found")
    
    def search_by_prefix(self, prefix):
        matches = []
        for name in self.contacts:
            if name.lower().startswith(prefix.lower()):
                matches.append((name, self.contacts[name]))
        return matches
    
    def remove_contact(self, name):
        if name in self.contacts:
            phone_number = self.contacts[name]
            del self.contacts[name]
            del self.reverse_lookup[phone_number]
            return True
        return False
    
    def list_all_contacts(self):
        return [(name, number) for name, number in sorted(self.contacts.items())]

# Usage example
phone_book = PhoneBook()
phone_book.add_contact("Alice Johnson", "+1-555-0123")
phone_book.add_contact("Bob Smith", "+1-555-0456")
phone_book.add_contact("Charlie Brown", "+1-555-0789")

# Fast lookups
print(phone_book.get_number("Alice Johnson"))  # +1-555-0123
print(phone_book.get_name("+1-555-0456"))      # Bob Smith
print(phone_book.search_by_prefix("Al"))        # [('Alice Johnson', '+1-555-0123')]
```

### 2. **E-commerce - Product Catalog**
Search for product details using product ID as key:
```python
class ProductCatalog:
    def __init__(self):
        self.products = {}  # product_id -> product_info
        self.category_index = {}  # category -> [product_ids]
        self.price_index = {}  # price_range -> [product_ids]
    
    def add_product(self, product_id, name, category, price, description):
        product_info = {
            'name': name,
            'category': category,
            'price': price,
            'description': description,
            'in_stock': True
        }
        
        self.products[product_id] = product_info
        
        # Update category index
        if category not in self.category_index:
            self.category_index[category] = []
        self.category_index[category].append(product_id)
        
        # Update price index
        price_range = self._get_price_range(price)
        if price_range not in self.price_index:
            self.price_index[price_range] = []
        self.price_index[price_range].append(product_id)
    
    def get_product(self, product_id):
        return self.products.get(product_id, None)
    
    def search_by_category(self, category):
        product_ids = self.category_index.get(category, [])
        return [self.products[pid] for pid in product_ids if pid in self.products]
    
    def search_by_price_range(self, min_price, max_price):
        matching_products = []
        for product_id, product in self.products.items():
            if min_price <= product['price'] <= max_price:
                matching_products.append(product)
        return matching_products
    
    def update_stock(self, product_id, in_stock):
        if product_id in self.products:
            self.products[product_id]['in_stock'] = in_stock
            return True
        return False
    
    def _get_price_range(self, price):
        if price < 50:
            return "budget"
        elif price < 200:
            return "mid-range"
        else:
            return "premium"
    
    def get_recommendations(self, product_id, limit=5):
        """Get similar products based on category"""
        if product_id not in self.products:
            return []
        
        category = self.products[product_id]['category']
        similar_products = self.search_by_category(category)
        
        # Filter out the current product and limit results
        recommendations = [p for p in similar_products 
                         if p != self.products[product_id]][:limit]
        return recommendations

# Usage example
catalog = ProductCatalog()
catalog.add_product("LAPTOP001", "Gaming Laptop", "Electronics", 1299.99, "High-performance gaming laptop")
catalog.add_product("BOOK001", "Python Programming", "Books", 39.99, "Learn Python programming")
catalog.add_product("PHONE001", "Smartphone", "Electronics", 699.99, "Latest smartphone model")

# Fast product lookup
product = catalog.get_product("LAPTOP001")
print(f"Product: {product['name']}, Price: ${product['price']}")

# Category-based search
electronics = catalog.search_by_category("Electronics")
print(f"Electronics products: {len(electronics)}")
```

### 3. **File System - File Path Mapping**
Store correspondence between file names and their paths:
```python
class FileSystem:
    def __init__(self):
        self.file_paths = {}  # filename -> full_path
        self.directory_contents = {}  # directory_path -> [filenames]
        self.file_metadata = {}  # full_path -> metadata
    
    def add_file(self, filename, directory_path, file_size, file_type):
        full_path = f"{directory_path}/{filename}"
        
        # Store file path mapping
        self.file_paths[filename] = full_path
        
        # Update directory contents
        if directory_path not in self.directory_contents:
            self.directory_contents[directory_path] = []
        self.directory_contents[directory_path].append(filename)
        
        # Store metadata
        self.file_metadata[full_path] = {
            'size': file_size,
            'type': file_type,
            'created': __import__('datetime').datetime.now(),
            'accessed': __import__('datetime').datetime.now()
        }
    
    def find_file(self, filename):
        """Quick file lookup by name"""
        return self.file_paths.get(filename, None)
    
    def list_directory(self, directory_path):
        """List all files in a directory"""
        return self.directory_contents.get(directory_path, [])
    
    def get_file_info(self, filename):
        """Get detailed file information"""
        full_path = self.find_file(filename)
        if full_path:
            return self.file_metadata.get(full_path, None)
        return None
    
    def search_by_extension(self, extension):
        """Find all files with specific extension"""
        matching_files = []
        for filename, full_path in self.file_paths.items():
            if filename.endswith(extension):
                matching_files.append({
                    'filename': filename,
                    'path': full_path,
                    'metadata': self.file_metadata.get(full_path, {})
                })
        return matching_files
    
    def calculate_directory_size(self, directory_path):
        """Calculate total size of all files in directory"""
        total_size = 0
        filenames = self.directory_contents.get(directory_path, [])
        
        for filename in filenames:
            full_path = self.file_paths[filename]
            metadata = self.file_metadata.get(full_path, {})
            total_size += metadata.get('size', 0)
        
        return total_size
    
    def move_file(self, filename, new_directory):
        """Move file to new directory"""
        old_path = self.find_file(filename)
        if not old_path:
            return False
        
        # Extract old directory
        old_directory = '/'.join(old_path.split('/')[:-1])
        
        # Update file path
        new_path = f"{new_directory}/{filename}"
        self.file_paths[filename] = new_path
        
        # Update directory contents
        if old_directory in self.directory_contents:
            self.directory_contents[old_directory].remove(filename)
        
        if new_directory not in self.directory_contents:
            self.directory_contents[new_directory] = []
        self.directory_contents[new_directory].append(filename)
        
        # Update metadata
        metadata = self.file_metadata.pop(old_path, {})
        self.file_metadata[new_path] = metadata
        
        return True

# Usage example
fs = FileSystem()
fs.add_file("document.pdf", "/home/user/documents", 2048, "PDF")
fs.add_file("photo.jpg", "/home/user/pictures", 4096, "JPEG")
fs.add_file("script.py", "/home/user/projects", 1024, "Python")

# Fast file lookup
path = fs.find_file("document.pdf")
print(f"Document location: {path}")

# Directory listing
documents = fs.list_directory("/home/user/documents")
print(f"Documents folder: {documents}")

# Search by file type
python_files = fs.search_by_extension(".py")
print(f"Python files: {python_files}")
```

### 4. **Database Indexing**
Hash maps for database query optimization:
```python
class DatabaseIndex:
    def __init__(self):
        self.primary_index = {}  # primary_key -> record
        self.secondary_indexes = {}  # field_name -> {value -> [primary_keys]}
    
    def create_index(self, field_name):
        """Create secondary index on a field"""
        if field_name not in self.secondary_indexes:
            self.secondary_indexes[field_name] = {}
    
    def insert_record(self, primary_key, record):
        """Insert a record and update all indexes"""
        self.primary_index[primary_key] = record
        
        # Update secondary indexes
        for field_name, index in self.secondary_indexes.items():
            if field_name in record:
                field_value = record[field_name]
                if field_value not in index:
                    index[field_value] = []
                index[field_value].append(primary_key)
    
    def get_record(self, primary_key):
        """Fast primary key lookup"""
        return self.primary_index.get(primary_key, None)
    
    def query_by_field(self, field_name, field_value):
        """Fast secondary index lookup"""
        if field_name in self.secondary_indexes:
            primary_keys = self.secondary_indexes[field_name].get(field_value, [])
            return [self.primary_index[pk] for pk in primary_keys]
        return []
    
    def update_record(self, primary_key, updated_record):
        """Update record and maintain index consistency"""
        old_record = self.primary_index.get(primary_key)
        if not old_record:
            return False
        
        # Remove from secondary indexes
        for field_name, index in self.secondary_indexes.items():
            if field_name in old_record:
                old_value = old_record[field_name]
                if old_value in index:
                    index[old_value].remove(primary_key)
                    if not index[old_value]:
                        del index[old_value]
        
        # Update primary index
        self.primary_index[primary_key] = updated_record
        
        # Add to secondary indexes
        for field_name, index in self.secondary_indexes.items():
            if field_name in updated_record:
                new_value = updated_record[field_name]
                if new_value not in index:
                    index[new_value] = []
                index[new_value].append(primary_key)
        
        return True

# Usage example
db_index = DatabaseIndex()
db_index.create_index("department")
db_index.create_index("salary")

# Insert employee records
db_index.insert_record("EMP001", {"name": "Alice", "department": "Engineering", "salary": 75000})
db_index.insert_record("EMP002", {"name": "Bob", "department": "Marketing", "salary": 65000})
db_index.insert_record("EMP003", {"name": "Charlie", "department": "Engineering", "salary": 80000})

# Fast queries
employee = db_index.get_record("EMP001")
print(f"Employee: {employee}")

engineering_employees = db_index.query_by_field("department", "Engineering")
print(f"Engineering team: {len(engineering_employees)} employees")
```

### 5. **Caching System**
Hash map-based caching for web applications:
```python
import time
from collections import OrderedDict

class WebCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()  # Maintains insertion order for LRU
        self.timestamps = {}  # url -> timestamp
    
    def _is_expired(self, url):
        """Check if cached item has expired"""
        if url not in self.timestamps:
            return True
        return time.time() - self.timestamps[url] > self.ttl_seconds
    
    def _evict_expired(self):
        """Remove expired items"""
        current_time = time.time()
        expired_urls = []
        
        for url, timestamp in self.timestamps.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_urls.append(url)
        
        for url in expired_urls:
            if url in self.cache:
                del self.cache[url]
            del self.timestamps[url]
    
    def get(self, url):
        """Get cached content for URL"""
        self._evict_expired()
        
        if url in self.cache and not self._is_expired(url):
            # Move to end (most recently used)
            self.cache.move_to_end(url)
            return self.cache[url]
        
        return None
    
    def put(self, url, content):
        """Cache content for URL"""
        self._evict_expired()
        
        # If already exists, update and move to end
        if url in self.cache:
            self.cache[url] = content
            self.cache.move_to_end(url)
            self.timestamps[url] = time.time()
            return
        
        # If at capacity, remove least recently used
        if len(self.cache) >= self.max_size:
            oldest_url = next(iter(self.cache))
            del self.cache[oldest_url]
            del self.timestamps[oldest_url]
        
        # Add new item
        self.cache[url] = content
        self.timestamps[url] = time.time()
    
    def invalidate(self, url):
        """Remove specific URL from cache"""
        if url in self.cache:
            del self.cache[url]
        if url in self.timestamps:
            del self.timestamps[url]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def stats(self):
        """Get cache statistics"""
        self._evict_expired()
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'urls': list(self.cache.keys())
        }

# Usage example
cache = WebCache(max_size=100, ttl_seconds=1800)  # 30 minutes TTL

# Cache web page content
cache.put("/home", "<html>Home page content</html>")
cache.put("/products", "<html>Products page content</html>")
cache.put("/about", "<html>About page content</html>")

# Fast retrieval
home_content = cache.get("/home")
if home_content:
    print("Cache hit: Serving cached content")
else:
    print("Cache miss: Need to generate content")
```

## Practice Problems

### Beginner Level
1. **Two Sum** - Basic hash map usage for complement finding
2. **Valid Anagram** - Character frequency comparison
3. **Contains Duplicate** - Duplicate detection using hash set
4. **Single Number** - XOR with hash map alternative
5. **Intersection of Two Arrays** - Set operations with hash maps

### Intermediate Level
6. **Group Anagrams** - Complex key generation and grouping
7. **Top K Frequent Elements** - Frequency counting with heap
8. **Subarray Sum Equals K** - Prefix sum with hash map
9. **Longest Substring Without Repeating Characters** - Sliding window with hash map
10. **4Sum II** - Multi-hash map approach
11. **Find All Anagrams in a String** - Pattern matching with hash map
12. **Minimum Window Substring** - Advanced sliding window

### Advanced Level
13. **LRU Cache** - Design problem with hash map and doubly linked list
14. **Design Twitter** - Complex system design with multiple hash maps
15. **Alien Dictionary** - Topological sort with hash map
16. **Word Pattern II** - Backtracking with hash map
17. **Design Search Autocomplete System** - Trie with hash map optimization
18. **Random Pick with Blacklist** - Advanced hash map remapping

## Common Patterns and Optimizations

### 1. **Frequency Pattern**
```python
def frequency_analysis(arr):
    from collections import Counter
    
    # Method 1: Manual counting
    freq_map = {}
    for item in arr:
        freq_map[item] = freq_map.get(item, 0) + 1
    
    # Method 2: Using Counter
    freq_counter = Counter(arr)
    
    # Method 3: Using defaultdict
    from collections import defaultdict
    freq_default = defaultdict(int)
    for item in arr:
        freq_default[item] += 1
    
    return freq_map, freq_counter, dict(freq_default)
```

### 2. **Prefix Sum Pattern**
```python
def subarray_sum_k(nums, k):
    """Count subarrays with sum equal to k"""
    prefix_sum = 0
    sum_count = {0: 1}  # prefix_sum -> count
    result = 0
    
    for num in nums:
        prefix_sum += num
        
        # Check if (prefix_sum - k) exists
        if (prefix_sum - k) in sum_count:
            result += sum_count[prefix_sum - k]
        
        # Update count of current prefix sum
        sum_count[prefix_sum] = sum_count.get(prefix_sum, 0) + 1
    
    return result
```

### 3. **Complement Pattern**
```python
def find_pairs_with_sum(arr, target):
    """Find all pairs that sum to target"""
    seen = set()
    pairs = []
    
    for num in arr:
        complement = target - num
        
        if complement in seen:
            pairs.append((min(num, complement), max(num, complement)))
        
        seen.add(num)
    
    return list(set(pairs))  # Remove duplicates
```

### 4. **Grouping Pattern**
```python
def group_by_property(items, key_func):
    """Group items by a property"""
    from collections import defaultdict
    
    groups = defaultdict(list)
    
    for item in items:
        key = key_func(item)
        groups[key].append(item)
    
    return dict(groups)

# Example usage
words = ["eat", "tea", "tan", "ate", "nat", "bat"]
anagram_groups = group_by_property(words, lambda x: ''.join(sorted(x)))
```

## Time and Space Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Insert | O(1) average, O(n) worst | O(1) per element | Worst case with many collisions |
| Lookup | O(1) average, O(n) worst | O(1) | Hash table lookup |
| Delete | O(1) average, O(n) worst | O(1) | Similar to lookup |
| Iteration | O(n) | O(1) additional | Iterate through all elements |
| Frequency Count | O(n) | O(k) | k = number of unique elements |
| Two Sum | O(n) | O(n) | Single pass with hash map |

## Common Pitfalls and Solutions

### 1. **Unhashable Key Types**
```python
# Problem: Using lists as keys
# wrong_map = {[1, 2, 3]: "value"}  # TypeError

# Solution: Use tuples or convert to string
correct_map = {(1, 2, 3): "value"}
string_key_map = {str([1, 2, 3]): "value"}
```

### 2. **Mutable Default Values**
```python
# Problem: Mutable default in function parameter
def wrong_function(key, value, cache={}):  # Dangerous!
    cache[key] = value
    return cache

# Solution: Use None and create new dict
def correct_function(key, value, cache=None):
    if cache is None:
        cache = {}
    cache[key] = value
    return cache
```

### 3. **Key Existence Checking**
```python
# Method 1: Using 'in' operator (Recommended)
if key in hash_map:
    value = hash_map[key]

# Method 2: Using get() with default
value = hash_map.get(key, default_value)

# Method 3: Using setdefault()
value = hash_map.setdefault(key, default_value)

# Avoid: Using try/except for normal flow
try:
    value = hash_map[key]
except KeyError:
    value = default_value
```

### 4. **Memory Leaks with Large Hash Maps**
```python
# Problem: Hash map grows indefinitely
cache = {}
def cache_result(key, result):
    cache[key] = result  # Never cleaned up

# Solution: Use LRU cache or periodic cleanup
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_function(key):
    return expensive_computation(key)

# Or implement cleanup logic
MAX_CACHE_SIZE = 1000
def cache_with_limit(key, result):
    if len(cache) >= MAX_CACHE_SIZE:
        # Remove oldest entries (simplified)
        oldest_key = next(iter(cache))
        del cache[oldest_key]
    cache[key] = result
```

## Tips for Success

1. **Choose Right Data Structure**: dict, defaultdict, Counter, or custom
2. **Handle Missing Keys**: Use get(), setdefault(), or defaultdict
3. **Consider Memory Usage**: Be aware of hash map growth
4. **Use Appropriate Keys**: Ensure keys are hashable and meaningful
5. **Optimize for Common Cases**: Pre-size hash maps when possible
6. **Profile Performance**: Monitor hash function quality and collision rates
7. **Consider Alternatives**: Trees for ordered data, sets for membership only
8. **Handle Edge Cases**: Empty inputs, duplicate keys, large datasets

## When NOT to Use Hash Maps

- **Ordered Data**: When you need sorted order (use TreeMap/sorted structures)
- **Range Queries**: When you need range-based lookups
- **Memory Constraints**: When memory is very limited
- **Poor Hash Distribution**: When keys don't hash well
- **Simple Sequential Access**: When you only need sequential processing

## Conclusion

Hash Maps are essential for:
- Fast data access and retrieval
- Frequency counting and analysis
- Implementing caches and lookup tables
- Solving complement and pairing problems
- Building indexes and mapping systems
- System design and optimization

Master this pattern by understanding the key-value relationship in your problems and choosing the appropriate hash map variant. The key is recognizing when fast lookup and storage of relationships between data sets will provide the most efficient solution.
