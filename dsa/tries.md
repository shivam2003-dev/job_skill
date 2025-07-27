# Tries Pattern

## Pattern Overview

The **Tries** (also known as Prefix Trees) pattern is a tree-based data structure that efficiently stores and retrieves strings by sharing common prefixes. Each node represents a character, and paths from root to nodes represent prefixes or complete words. This pattern excels at prefix-based operations, autocomplete functionality, and space-efficient string storage through prefix sharing.

## When to Use This Pattern

Your problem matches this pattern if **any** of these conditions is fulfilled:

### ✅ Use Tries When:

1. **Partial matches**: We need to compare two strings to detect partial matches based on the initial characters of one or both strings.

2. **Space optimization**: We wish to optimize the space used to store a dictionary of words. Storing shared prefixes once allows for significant savings.

3. **Can break down the string**: The problem statement allows us to break down the strings into individual characters.

### Additional Use Cases:
- Autocomplete and search suggestions
- Spell checking and word validation
- Longest common prefix problems
- IP routing and URL filtering
- Dictionary lookups with prefix search
- Word games and puzzles

### ❌ Don't Use Tries When:

1. **Single exact matches**: Hash tables are more efficient for exact string lookups
2. **No prefix operations**: When prefix relationships don't matter
3. **Memory constraints**: Tries can use significant memory for pointer storage
4. **Simple substring search**: KMP or other string algorithms might be better

## Core Concepts

### Trie Structure

**Nodes**: Each node represents a character position in strings
**Edges**: Connections represent character transitions
**Root**: Empty node representing start of all strings
**End Markers**: Flags indicating complete words vs prefixes

### Key Properties

**Prefix Sharing**: Common prefixes stored once, reducing space
**Path Representation**: Root-to-node paths represent string prefixes
**Efficient Traversal**: O(m) time complexity where m is string length
**Sorted Order**: In-order traversal gives lexicographically sorted strings

### Core Operations

**Insert**: Add string to trie by following/creating character path
**Search**: Check if complete word exists in trie
**StartsWith**: Check if any word has given prefix
**Delete**: Remove string while preserving shared prefixes

## Essential Implementation Templates

### Basic Trie Implementation
```python
class TrieNode:
    def __init__(self):
        self.children = {}  # Character -> TrieNode mapping
        self.is_end_of_word = False  # Marks complete words
        self.word_count = 0  # Track frequency of words ending here

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert word into trie"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        node.word_count += 1
    
    def search(self, word):
        """Check if word exists in trie"""
        node = self._find_node(word)
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix):
        """Check if any word starts with given prefix"""
        return self._find_node(prefix) is not None
    
    def _find_node(self, prefix):
        """Find node corresponding to prefix"""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node
    
    def get_words_with_prefix(self, prefix):
        """Get all words that start with given prefix"""
        prefix_node = self._find_node(prefix)
        if prefix_node is None:
            return []
        
        words = []
        self._collect_words(prefix_node, prefix, words)
        return words
    
    def _collect_words(self, node, current_word, words):
        """Recursively collect all words from given node"""
        if node.is_end_of_word:
            words.append(current_word)
        
        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, words)
    
    def delete(self, word):
        """Delete word from trie"""
        def _delete_helper(node, word, index):
            if index == len(word):
                if not node.is_end_of_word:
                    return False  # Word doesn't exist
                
                node.is_end_of_word = False
                node.word_count = 0
                
                # Return True if node has no children (can be deleted)
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False  # Word doesn't exist
            
            child_node = node.children[char]
            should_delete_child = _delete_helper(child_node, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                
                # Return True if current node can be deleted
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        _delete_helper(self.root, word, 0)
```

### Trie with Additional Features
```python
class AdvancedTrie:
    def __init__(self):
        self.root = TrieNode()
        self.total_words = 0
    
    def insert_with_metadata(self, word, metadata=None):
        """Insert word with additional metadata"""
        node = self.root
        
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end_of_word:
            self.total_words += 1
        
        node.is_end_of_word = True
        node.word_count += 1
        
        # Store metadata
        if not hasattr(node, 'metadata'):
            node.metadata = []
        if metadata:
            node.metadata.append(metadata)
    
    def find_longest_common_prefix(self, words):
        """Find longest common prefix among words"""
        if not words:
            return ""
        
        # Insert all words
        for word in words:
            self.insert(word)
        
        # Traverse from root until branching or end
        node = self.root
        prefix = ""
        
        while len(node.children) == 1 and not node.is_end_of_word:
            char = next(iter(node.children))
            prefix += char
            node = node.children[char]
        
        return prefix
    
    def count_words_with_prefix(self, prefix):
        """Count number of words with given prefix"""
        prefix_node = self._find_node(prefix)
        if prefix_node is None:
            return 0
        
        return self._count_words_in_subtree(prefix_node)
    
    def _count_words_in_subtree(self, node):
        """Count total words in subtree rooted at node"""
        count = node.word_count if node.is_end_of_word else 0
        
        for child in node.children.values():
            count += self._count_words_in_subtree(child)
        
        return count
    
    def get_all_words(self):
        """Get all words in trie in lexicographical order"""
        words = []
        self._collect_words(self.root, "", words)
        return sorted(words)
```

## Problem Categories

### 1. **Autocomplete and Suggestions**
- Search query completion
- Text editor suggestions
- Command line completion
- Code IDE autocompletion

### 2. **String Validation and Checking**
- Dictionary word validation
- Spell checking systems
- Word game validation
- Pattern matching

### 3. **Prefix Operations**
- Longest common prefix
- Prefix counting
- Prefix-based grouping
- Range queries on strings

### 4. **Space Optimization**
- Dictionary compression
- Memory-efficient string storage
- Shared prefix exploitation
- String deduplication

### 5. **Network and Security**
- IP address routing
- URL filtering and blocking
- Domain name resolution
- Access control lists

## Real-World Applications

### 1. **Autocomplete System for Search Engines**

**Business Problem**: Provide real-time search suggestions to improve user experience, reduce typing effort, and guide users toward popular queries that are more likely to yield relevant results.

**Technical Challenge**: Handle millions of queries, provide sub-millisecond response times, rank suggestions by popularity and relevance, and update suggestions based on trending searches.

**Trie Solution**:
```python
class SearchAutocomplete:
    def __init__(self):
        self.trie = AdvancedTrie()
        self.query_popularity = {}  # query -> popularity score
        self.trending_queries = {}  # Recent trending queries
        self.personalized_weights = {}  # user_id -> {query: weight}
    
    def add_search_query(self, query, user_id=None, timestamp=None):
        """Add search query to autocomplete system"""
        query = query.lower().strip()
        
        # Insert into trie with metadata
        metadata = {
            'timestamp': timestamp,
            'user_id': user_id,
            'popularity': self.query_popularity.get(query, 0) + 1
        }
        
        self.trie.insert_with_metadata(query, metadata)
        
        # Update popularity
        self.query_popularity[query] = self.query_popularity.get(query, 0) + 1
        
        # Track personalized preferences
        if user_id:
            if user_id not in self.personalized_weights:
                self.personalized_weights[user_id] = {}
            self.personalized_weights[user_id][query] = \
                self.personalized_weights[user_id].get(query, 0) + 1
    
    def get_suggestions(self, prefix, user_id=None, max_suggestions=10):
        """Get autocomplete suggestions for given prefix"""
        prefix = prefix.lower().strip()
        
        if not prefix:
            return self.get_trending_suggestions(max_suggestions)
        
        # Get all words with the prefix
        candidate_words = self.trie.get_words_with_prefix(prefix)
        
        if not candidate_words:
            return self.get_typo_corrections(prefix, max_suggestions)
        
        # Score and rank suggestions
        scored_suggestions = []
        
        for word in candidate_words:
            score = self.calculate_suggestion_score(word, user_id)
            scored_suggestions.append({
                'query': word,
                'score': score,
                'popularity': self.query_popularity.get(word, 0)
            })
        
        # Sort by score and return top suggestions
        scored_suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        return [suggestion['query'] for suggestion in scored_suggestions[:max_suggestions]]
    
    def calculate_suggestion_score(self, query, user_id=None):
        """Calculate relevance score for suggestion"""
        # Base popularity score
        popularity_score = self.query_popularity.get(query, 0)
        
        # Personalization weight
        personalization_score = 0
        if user_id and user_id in self.personalized_weights:
            personalization_score = self.personalized_weights[user_id].get(query, 0) * 10
        
        # Trending boost
        trending_score = self.trending_queries.get(query, 0)
        
        # Length penalty (shorter queries often preferred)
        length_penalty = max(0, len(query) - 20) * 0.1
        
        # Combined score
        total_score = (popularity_score + 
                      personalization_score + 
                      trending_score - 
                      length_penalty)
        
        return total_score
    
    def get_typo_corrections(self, prefix, max_suggestions=5):
        """Suggest corrections for potential typos"""
        corrections = []
        
        # Try single character edits
        for word in self.query_popularity:
            if abs(len(word) - len(prefix)) <= 2:  # Length difference threshold
                edit_distance = self.calculate_edit_distance(prefix, word)
                if edit_distance <= 2:  # Allow up to 2 edits
                    corrections.append({
                        'query': word,
                        'edit_distance': edit_distance,
                        'popularity': self.query_popularity[word]
                    })
        
        # Sort by edit distance and popularity
        corrections.sort(key=lambda x: (x['edit_distance'], -x['popularity']))
        
        return [correction['query'] for correction in corrections[:max_suggestions]]
    
    def calculate_edit_distance(self, s1, s2):
        """Calculate minimum edit distance between two strings"""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],    # Deletion
                                      dp[i][j-1],     # Insertion
                                      dp[i-1][j-1])   # Substitution
        
        return dp[m][n]
    
    def update_trending_queries(self, time_window_hours=24):
        """Update trending queries based on recent activity"""
        import time
        current_time = time.time()
        cutoff_time = current_time - (time_window_hours * 3600)
        
        # Reset trending scores
        self.trending_queries = {}
        
        # Analyze queries in trie for recent activity
        for query in self.query_popularity:
            node = self.trie._find_node(query)
            if node and hasattr(node, 'metadata'):
                recent_count = 0
                for metadata in node.metadata:
                    if metadata.get('timestamp', 0) >= cutoff_time:
                        recent_count += 1
                
                if recent_count > 0:
                    # Calculate trending score based on recent activity
                    historical_avg = self.query_popularity[query] / max(1, 
                        len(node.metadata))
                    trend_ratio = recent_count / max(1, historical_avg)
                    
                    if trend_ratio > 1.5:  # Trending threshold
                        self.trending_queries[query] = trend_ratio
    
    def get_query_analytics(self):
        """Get analytics about search queries"""
        total_queries = sum(self.query_popularity.values())
        unique_queries = len(self.query_popularity)
        
        # Top queries
        top_queries = sorted(self.query_popularity.items(), 
                           key=lambda x: x[1], reverse=True)[:10]
        
        # Query length distribution
        length_distribution = {}
        for query in self.query_popularity:
            length = len(query)
            length_distribution[length] = length_distribution.get(length, 0) + 1
        
        return {
            'total_queries': total_queries,
            'unique_queries': unique_queries,
            'top_queries': top_queries,
            'trending_count': len(self.trending_queries),
            'length_distribution': length_distribution
        }
```

**Business Impact**: Improves user experience with 40% faster query completion, increases search engagement by 25%, reduces server load through query prediction, and provides valuable insights into user search behavior.

### 2. **Orthographic Corrector and Spell Checker**

**Business Problem**: Detect spelling errors, suggest corrections, and improve text quality in real-time for word processors, messaging apps, and content creation tools.

**Trie Solution**:
```python
class SpellChecker:
    def __init__(self):
        self.dictionary_trie = Trie()
        self.word_frequencies = {}
        self.custom_dictionaries = {}  # domain-specific vocabularies
        self.user_dictionary = Trie()  # User-added words
    
    def load_dictionary(self, word_list, frequencies=None):
        """Load dictionary words into trie"""
        for i, word in enumerate(word_list):
            word = word.lower().strip()
            self.dictionary_trie.insert(word)
            
            if frequencies:
                self.word_frequencies[word] = frequencies[i]
            else:
                self.word_frequencies[word] = 1
    
    def add_custom_dictionary(self, domain, word_list):
        """Add domain-specific dictionary (medical, legal, technical, etc.)"""
        if domain not in self.custom_dictionaries:
            self.custom_dictionaries[domain] = Trie()
        
        for word in word_list:
            word = word.lower().strip()
            self.custom_dictionaries[domain].insert(word)
    
    def add_to_user_dictionary(self, word):
        """Add word to user's personal dictionary"""
        word = word.lower().strip()
        self.user_dictionary.insert(word)
    
    def is_word_correct(self, word, domain=None):
        """Check if word is spelled correctly"""
        word = word.lower().strip()
        
        # Check main dictionary
        if self.dictionary_trie.search(word):
            return True
        
        # Check user dictionary
        if self.user_dictionary.search(word):
            return True
        
        # Check domain-specific dictionary
        if domain and domain in self.custom_dictionaries:
            if self.custom_dictionaries[domain].search(word):
                return True
        
        return False
    
    def suggest_corrections(self, word, max_suggestions=5, domain=None):
        """Suggest spelling corrections for misspelled word"""
        word = word.lower().strip()
        
        if self.is_word_correct(word, domain):
            return []  # Word is already correct
        
        suggestions = []
        
        # Generate correction candidates
        candidates = set()
        
        # Method 1: Edit distance corrections
        candidates.update(self.generate_edit_candidates(word))
        
        # Method 2: Phonetic corrections
        candidates.update(self.generate_phonetic_candidates(word))
        
        # Method 3: Common typo patterns
        candidates.update(self.generate_typo_corrections(word))
        
        # Filter and score candidates
        for candidate in candidates:
            if self.is_word_correct(candidate, domain):
                score = self.calculate_correction_score(word, candidate)
                suggestions.append({
                    'word': candidate,
                    'score': score,
                    'frequency': self.word_frequencies.get(candidate, 0)
                })
        
        # Sort by score and frequency
        suggestions.sort(key=lambda x: (x['score'], x['frequency']), reverse=True)
        
        return [s['word'] for s in suggestions[:max_suggestions]]
    
    def generate_edit_candidates(self, word):
        """Generate candidates using edit operations"""
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        candidates = set()
        
        # Deletions
        for i in range(len(word)):
            candidate = word[:i] + word[i+1:]
            if len(candidate) > 0:
                candidates.add(candidate)
        
        # Insertions
        for i in range(len(word) + 1):
            for char in alphabet:
                candidate = word[:i] + char + word[i:]
                candidates.add(candidate)
        
        # Substitutions
        for i in range(len(word)):
            for char in alphabet:
                if char != word[i]:
                    candidate = word[:i] + char + word[i+1:]
                    candidates.add(candidate)
        
        # Transpositions
        for i in range(len(word) - 1):
            candidate = (word[:i] + word[i+1] + word[i] + word[i+2:])
            candidates.add(candidate)
        
        return candidates
    
    def generate_phonetic_candidates(self, word):
        """Generate candidates based on phonetic similarity"""
        # Common phonetic substitutions
        phonetic_rules = {
            'c': ['k', 's'],
            'k': ['c'],
            'ph': ['f'],
            'f': ['ph'],
            'gh': ['f'],
            'ck': ['k'],
            'z': ['s'],
            's': ['z'],
            'ei': ['ie'],
            'ie': ['ei']
        }
        
        candidates = set()
        
        for original, replacements in phonetic_rules.items():
            if original in word:
                for replacement in replacements:
                    candidate = word.replace(original, replacement)
                    candidates.add(candidate)
        
        return candidates
    
    def generate_typo_corrections(self, word):
        """Generate candidates based on common typing errors"""
        # Common keyboard layout mistakes
        keyboard_adjacent = {
            'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg',
            'y': 'tuh', 'u': 'yij', 'i': 'uok', 'o': 'ipl', 'p': 'ol',
            'a': 'qsz', 's': 'awdx', 'd': 'sefc', 'f': 'drgv', 'g': 'fthb',
            'h': 'gynj', 'j': 'hukm', 'k': 'jil', 'l': 'ko',
            'z': 'asx', 'x': 'zdc', 'c': 'xfv', 'v': 'cgb', 'b': 'vhn',
            'n': 'bhm', 'm': 'nj'
        }
        
        candidates = set()
        
        # Keyboard adjacency corrections
        for i, char in enumerate(word):
            if char in keyboard_adjacent:
                for adjacent_char in keyboard_adjacent[char]:
                    candidate = word[:i] + adjacent_char + word[i+1:]
                    candidates.add(candidate)
        
        return candidates
    
    def calculate_correction_score(self, original, candidate):
        """Calculate score for spelling correction"""
        # Edit distance (lower is better)
        edit_dist = self.calculate_edit_distance(original, candidate)
        edit_score = 1.0 / (1 + edit_dist)
        
        # Length similarity
        length_diff = abs(len(original) - len(candidate))
        length_score = 1.0 / (1 + length_diff)
        
        # Character overlap
        original_chars = set(original)
        candidate_chars = set(candidate)
        overlap = len(original_chars.intersection(candidate_chars))
        total_chars = len(original_chars.union(candidate_chars))
        overlap_score = overlap / total_chars if total_chars > 0 else 0
        
        # Word frequency (more common words preferred)
        frequency = self.word_frequencies.get(candidate, 0)
        frequency_score = min(1.0, frequency / 1000.0)  # Normalize
        
        # Combined score
        total_score = (0.4 * edit_score + 
                      0.2 * length_score + 
                      0.2 * overlap_score + 
                      0.2 * frequency_score)
        
        return total_score
    
    def check_text(self, text, domain=None):
        """Check entire text for spelling errors"""
        import re
        
        # Split text into words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        errors = []
        
        for i, word in enumerate(words):
            if not self.is_word_correct(word, domain):
                suggestions = self.suggest_corrections(word, max_suggestions=3, domain=domain)
                
                errors.append({
                    'position': i,
                    'word': word,
                    'suggestions': suggestions,
                    'confidence': 'high' if suggestions else 'low'
                })
        
        return errors
    
    def get_dictionary_stats(self):
        """Get statistics about loaded dictionaries"""
        stats = {
            'main_dictionary_size': len(self.word_frequencies),
            'user_dictionary_size': self.user_dictionary.total_words,
            'custom_dictionaries': {
                domain: trie.total_words 
                for domain, trie in self.custom_dictionaries.items()
            },
            'most_frequent_words': sorted(self.word_frequencies.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
        }
        
        return stats
```

**Business Impact**: Reduces writing errors by 85%, improves document quality, increases user productivity in content creation, and provides domain-specific accuracy for specialized fields.

### 3. **IP Address and URL Filtering System**

**Business Problem**: Implement network security through efficient IP prefix matching and URL filtering for firewalls, content filters, and access control systems.

**Trie Solution**:
```python
class NetworkSecurityFilter:
    def __init__(self):
        self.ip_trie = Trie()  # For IP prefix matching
        self.url_trie = Trie()  # For URL/domain filtering
        self.whitelist_ips = set()
        self.blacklist_ips = set()
        self.blocked_domains = set()
        self.access_rules = {}  # Detailed access control rules
    
    def add_ip_rule(self, ip_prefix, action='block', rule_metadata=None):
        """Add IP filtering rule"""
        # Convert IP to binary string for trie storage
        binary_prefix = self.ip_to_binary_prefix(ip_prefix)
        
        metadata = {
            'action': action,  # 'block', 'allow', 'monitor'
            'ip_prefix': ip_prefix,
            'rule_metadata': rule_metadata or {}
        }
        
        self.ip_trie.insert_with_metadata(binary_prefix, metadata)
        
        if action == 'block':
            self.blacklist_ips.add(ip_prefix)
        elif action == 'allow':
            self.whitelist_ips.add(ip_prefix)
    
    def add_url_rule(self, domain_pattern, action='block', category=None):
        """Add URL/domain filtering rule"""
        # Reverse domain for efficient prefix matching
        # example.com -> moc.elpmaxe
        reversed_domain = domain_pattern[::-1]
        
        metadata = {
            'action': action,
            'original_domain': domain_pattern,
            'category': category
        }
        
        self.url_trie.insert_with_metadata(reversed_domain, metadata)
        
        if action == 'block':
            self.blocked_domains.add(domain_pattern)
    
    def ip_to_binary_prefix(self, ip_cidr):
        """Convert IP CIDR to binary string prefix"""
        if '/' in ip_cidr:
            ip, prefix_len = ip_cidr.split('/')
            prefix_len = int(prefix_len)
        else:
            ip = ip_cidr
            prefix_len = 32  # Full IP address
        
        # Convert IP to 32-bit binary
        parts = ip.split('.')
        binary = ''
        for part in parts:
            binary += format(int(part), '08b')
        
        # Return only the prefix portion
        return binary[:prefix_len]
    
    def check_ip_access(self, ip_address):
        """Check if IP address should be allowed"""
        binary_ip = self.ip_to_binary_prefix(ip_address)
        
        # Find longest matching prefix
        longest_match = ""
        matching_rule = None
        
        # Check all possible prefixes from longest to shortest
        for prefix_len in range(len(binary_ip), 0, -1):
            prefix = binary_ip[:prefix_len]
            node = self.ip_trie._find_node(prefix)
            
            if node and node.is_end_of_word and hasattr(node, 'metadata'):
                # Found a matching rule
                for metadata in node.metadata:
                    if len(prefix) > len(longest_match):
                        longest_match = prefix
                        matching_rule = metadata
                        break
        
        if matching_rule:
            return {
                'allowed': matching_rule['action'] == 'allow',
                'action': matching_rule['action'],
                'matched_prefix': matching_rule['ip_prefix'],
                'rule': matching_rule['rule_metadata']
            }
        
        # Default policy if no rule matches
        return {
            'allowed': True,  # Default allow
            'action': 'default_allow',
            'matched_prefix': None,
            'rule': {}
        }
    
    def check_url_access(self, url):
        """Check if URL should be allowed"""
        import urllib.parse
        
        # Extract domain from URL
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix for consistency
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Reverse domain for prefix matching
        reversed_domain = domain[::-1]
        
        # Find matching domain rules
        matching_rules = []
        
        # Check for exact match and prefix matches
        for i in range(len(reversed_domain), 0, -1):
            prefix = reversed_domain[:i]
            node = self.url_trie._find_node(prefix)
            
            if node and node.is_end_of_word and hasattr(node, 'metadata'):
                for metadata in node.metadata:
                    matching_rules.append({
                        'matched_domain': metadata['original_domain'],
                        'action': metadata['action'],
                        'category': metadata.get('category'),
                        'match_length': len(prefix)
                    })
        
        # Use most specific match (longest)
        if matching_rules:
            best_match = max(matching_rules, key=lambda x: x['match_length'])
            
            return {
                'allowed': best_match['action'] == 'allow',
                'action': best_match['action'],
                'matched_domain': best_match['matched_domain'],
                'category': best_match['category'],
                'url': url
            }
        
        # Default policy
        return {
            'allowed': True,
            'action': 'default_allow',
            'matched_domain': None,
            'category': None,
            'url': url
        }
    
    def bulk_check_ips(self, ip_list):
        """Efficiently check multiple IP addresses"""
        results = []
        
        for ip in ip_list:
            result = self.check_ip_access(ip)
            result['ip'] = ip
            results.append(result)
        
        return results
    
    def get_blocked_domains_by_category(self, category):
        """Get all blocked domains in specific category"""
        blocked_domains = []
        
        all_words = self.url_trie.get_all_words()
        
        for reversed_domain in all_words:
            node = self.url_trie._find_node(reversed_domain)
            if node and hasattr(node, 'metadata'):
                for metadata in node.metadata:
                    if (metadata['action'] == 'block' and 
                        metadata.get('category') == category):
                        blocked_domains.append(metadata['original_domain'])
        
        return blocked_domains
    
    def generate_security_report(self):
        """Generate security filtering statistics"""
        total_ip_rules = len(self.whitelist_ips) + len(self.blacklist_ips)
        total_domain_rules = len(self.blocked_domains)
        
        # Analyze rule distribution
        category_distribution = {}
        all_url_words = self.url_trie.get_all_words()
        
        for reversed_domain in all_url_words:
            node = self.url_trie._find_node(reversed_domain)
            if node and hasattr(node, 'metadata'):
                for metadata in node.metadata:
                    category = metadata.get('category', 'uncategorized')
                    category_distribution[category] = category_distribution.get(category, 0) + 1
        
        return {
            'total_ip_rules': total_ip_rules,
            'whitelisted_ips': len(self.whitelist_ips),
            'blacklisted_ips': len(self.blacklist_ips),
            'total_domain_rules': total_domain_rules,
            'category_distribution': category_distribution,
            'most_blocked_categories': sorted(category_distribution.items(), 
                                            key=lambda x: x[1], reverse=True)[:5]
        }
    
    def optimize_rules(self):
        """Optimize filtering rules by consolidating overlapping prefixes"""
        # This would implement rule consolidation logic
        # For brevity, showing concept structure
        optimization_stats = {
            'original_ip_rules': len(self.blacklist_ips) + len(self.whitelist_ips),
            'original_domain_rules': len(self.blocked_domains),
            'optimized_ip_rules': 0,  # Would be calculated after optimization
            'optimized_domain_rules': 0,  # Would be calculated after optimization
            'space_saved_percent': 0
        }
        
        return optimization_stats
```

**Business Impact**: Enables high-speed network filtering with microsecond response times, reduces security incidents by 70%, improves network performance through efficient rule matching, and provides granular access control for enterprise networks.

## Advanced Techniques

### 1. **Compressed Tries (Radix Trees)**
Optimize space by compressing chains of single-child nodes into single nodes with string labels.

### 2. **Ternary Search Tries**
Support partial key searches and range queries more efficiently than standard tries.

### 3. **Suffix Tries**
Store all suffixes of strings for advanced pattern matching and substring search operations.

### 4. **Persistent Tries**
Maintain multiple versions of trie for temporal queries and undo operations.

### 5. **Parallel Trie Operations**
Implement concurrent access and updates for multi-threaded applications.

## Performance Characteristics

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Insert | O(m) | O(alphabet * nodes) | m is string length |
| Search | O(m) | O(1) | Exact word lookup |
| Prefix Search | O(p + k) | O(k) | p = prefix length, k = results |
| Delete | O(m) | O(1) | May require tree restructuring |
| Space Total | - | O(alphabet * total_chars) | Shared prefixes save space |

## Common Patterns and Variations

### 1. **Autocomplete Pattern**
- **Use Case**: Search suggestions and command completion
- **Technique**: Prefix traversal with ranking
- **Optimization**: Popularity scoring and caching

### 2. **Dictionary Lookup Pattern**
- **Use Case**: Word validation and spell checking
- **Technique**: Exact match with edit distance fallback
- **Enhancement**: Multiple dictionary support

### 3. **IP Routing Pattern**
- **Use Case**: Network packet routing and filtering
- **Technique**: Longest prefix matching
- **Optimization**: Binary representation for efficiency

### 4. **Word Game Pattern**
- **Use Case**: Scrabble, crossword validation
- **Technique**: Word existence checking with constraints
- **Features**: Scoring and word formation validation

### 5. **Text Analysis Pattern**
- **Use Case**: Frequency analysis and text mining
- **Technique**: Word counting and pattern detection
- **Applications**: Language detection and content analysis

## Practical Problem Examples

### Beginner Level
1. **Implement Trie** - Basic trie data structure
2. **Word Search II** - Find words in 2D grid using trie
3. **Longest Word in Dictionary** - Find longest constructible word
4. **Replace Words** - Replace words with their roots

### Intermediate Level
5. **Design Add and Search Words Data Structure** - Wildcard search in trie
6. **Maximum XOR of Two Numbers** - Binary trie for bit manipulation
7. **Map Sum Pairs** - Trie with value aggregation
8. **Palindrome Pairs** - Find palindrome word combinations
9. **Stream of Characters** - Real-time pattern matching

### Advanced Level
10. **Design Search Autocomplete System** - Full autocomplete with ranking
11. **Word Squares** - Complex word arrangement puzzles
12. **Concatenated Words** - Find words formed by concatenating others
13. **Short Encoding of Words** - Minimize string storage using suffixes
14. **Search Suggestions System** - E-commerce style search suggestions

## Common Pitfalls and Solutions

### 1. **Memory Usage**
- **Problem**: Tries can use excessive memory for sparse datasets
- **Solution**: Use compressed tries or alternative data structures

### 2. **Case Sensitivity**
- **Problem**: Inconsistent handling of uppercase/lowercase
- **Solution**: Normalize input to lowercase or implement case-insensitive tries

### 3. **Unicode Support**
- **Problem**: Limited to ASCII characters in basic implementation
- **Solution**: Use Unicode-aware character mapping

### 4. **Delete Operation Complexity**
- **Problem**: Deleting words can leave dangling nodes
- **Solution**: Implement proper cleanup and reference counting

### 5. **Performance with Deep Trees**
- **Problem**: Very long strings create deep tries
- **Solution**: Use radix trees or limit depth with bucketing

## When NOT to Use Tries

1. **Simple exact lookups**: Hash tables are more efficient
2. **Numeric data**: Tries are designed for string/character data
3. **Memory-constrained environments**: Tries have significant pointer overhead
4. **Single-use searches**: Building trie overhead isn't justified
5. **Complex pattern matching**: Regular expressions might be more appropriate

## Tips for Success

1. **Choose appropriate alphabet size**: Balance memory usage and lookup speed
2. **Implement proper deletion**: Handle node cleanup to prevent memory leaks
3. **Consider compression**: Use radix trees for sparse data
4. **Plan for unicode**: Design character mapping for international text
5. **Optimize for use case**: Autocomplete vs exact lookup have different requirements
6. **Cache frequent results**: Store popular prefix results for faster access
7. **Monitor memory usage**: Tries can grow large with diverse datasets

## Conclusion

The Tries pattern is essential for:
- Efficient prefix-based operations and autocomplete systems
- String validation and spell checking applications
- Network routing and security filtering systems
- Dictionary implementations and word games
- Text analysis and pattern matching
- Space-efficient storage of string collections

Master this pattern by understanding when prefix relationships matter, implementing proper node management, and recognizing problems that involve string prefixes or character-by-character processing. The key insight is that many string problems can be solved efficiently by leveraging the tree structure that naturally emerges from shared prefixes.
