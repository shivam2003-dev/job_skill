# Tree Breadth-First Search (BFS) Pattern

## Pattern Overview

The **Tree Breadth-First Search (BFS)** pattern is a level-order traversal technique that explores all nodes at the current depth before moving to nodes at the next depth level. It's particularly effective for problems requiring level-by-level processing, finding shortest paths, or when solutions are likely to be found near the root.

## When to Use This Pattern

Your problem matches this pattern if **any** of these conditions is fulfilled:

### ✅ Use Tree BFS When:

1. **Tree data structure**: The input data is in the form of a tree, or the cost of transforming it into a tree is low.

2. **Not a wide tree**: The tree being searched is not very wide, as BFS may become prohibitive for extremely wide trees due to memory requirements.

3. **Level-by-level traversal**: The solution dictates traversing the tree one level at a time, for example, to find the level order traversal of the nodes of a tree or a variant of this ordering.

4. **Solution near the root**: We have reason to believe that the solution is near the root of the tree.

### ❌ When BFS Might Not Be Optimal:

- Solution likely near leaves (DFS is better)
- Very wide trees (memory intensive)
- Deep trees with limited memory
- Path-dependent problems requiring backtracking

## Tree BFS Fundamentals

### Core Concepts
- **Queue-based**: Uses FIFO (First In, First Out) data structure
- **Level-order**: Processes nodes level by level
- **Shortest Path**: Finds shortest path in unweighted trees
- **Memory Trade-off**: Uses more memory but guarantees shortest path

### Tree Node Definition
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Enhanced node with next pointer for space optimization
class TreeNodeWithNext:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
```

## Implementation Templates

### Basic BFS Template
```python
from collections import deque

def bfs_basic(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        # Add children to queue
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    
    return result
```

### Level-by-Level BFS Template
```python
def bfs_level_order(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

### BFS with Level Tracking Template
```python
def bfs_with_levels(root):
    if not root:
        return []
    
    result = []
    queue = deque([(root, 0)])  # (node, level)
    
    while queue:
        node, level = queue.popleft()
        
        # Extend result if new level
        if level >= len(result):
            result.append([])
        
        result[level].append(node.val)
        
        # Add children with incremented level
        if node.left:
            queue.append((node.left, level + 1))
        if node.right:
            queue.append((node.right, level + 1))
    
    return result
```

### BFS for Tree Properties Template
```python
def bfs_tree_properties(root):
    if not root:
        return {"height": 0, "width": 0, "nodes": 0}
    
    queue = deque([root])
    height = 0
    max_width = 0
    total_nodes = 0
    
    while queue:
        level_size = len(queue)
        max_width = max(max_width, level_size)
        height += 1
        total_nodes += level_size
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return {
        "height": height,
        "width": max_width,
        "nodes": total_nodes
    }
```

### Right Side View Template
```python
def right_side_view(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Last node in level is rightmost
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
```

## Common Problem Categories

### 1. **Level Order Traversals**
- Basic level order traversal
- Reverse level order (bottom-up)
- Zigzag level order traversal
- Level order with level separation

### 2. **Tree Properties**
- Minimum/maximum depth
- Tree width calculation
- Level with maximum sum
- Complete tree node counting

### 3. **View Problems**
- Right side view
- Left side view
- Bottom view
- Top view

### 4. **Path Finding**
- Shortest path to target
- Minimum steps to reach node
- Level of target node
- Distance between nodes

### 5. **Tree Modification**
- Connect nodes at same level
- Populate next pointers
- Level-based transformations
- Tree serialization by levels

### 6. **Tree Analysis**
- Check if tree is complete
- Find cousins of a node
- Level-wise statistics
- Tree comparison by levels

## Advanced BFS Techniques

### 1. **Two-Way BFS (Bidirectional)**
```python
def bidirectional_bfs(root, target):
    if not root:
        return -1
    
    if root.val == target:
        return 0
    
    # Forward BFS from root
    forward_queue = deque([(root, 0)])
    forward_visited = {root.val: 0}
    
    # Backward BFS from target nodes
    backward_queue = deque()
    backward_visited = {}
    
    # Find all target nodes first
    def find_targets(node, level):
        if not node:
            return
        if node.val == target:
            backward_queue.append((node, level))
            backward_visited[node.val] = level
        
        find_targets(node.left, level + 1)
        find_targets(node.right, level + 1)
    
    find_targets(root, 0)
    
    while forward_queue and backward_queue:
        # Expand smaller frontier
        if len(forward_queue) <= len(backward_queue):
            # Expand forward
            for _ in range(len(forward_queue)):
                node, dist = forward_queue.popleft()
                
                if node.val in backward_visited:
                    return dist + backward_visited[node.val]
                
                for child in [node.left, node.right]:
                    if child and child.val not in forward_visited:
                        forward_visited[child.val] = dist + 1
                        forward_queue.append((child, dist + 1))
        else:
            # Similar expansion for backward
            pass
    
    return -1
```

### 2. **Space-Optimized BFS with Next Pointers**
```python
def connect_next_pointers(root):
    if not root:
        return root
    
    leftmost = root
    
    while leftmost.left:  # While not leaf level
        head = leftmost
        
        while head:
            # Connect children of current node
            head.left.next = head.right
            
            # Connect to next node's left child
            if head.next:
                head.right.next = head.next.left
            
            head = head.next
        
        leftmost = leftmost.left
    
    return root
```

### 3. **BFS with Multiple Sources**
```python
def multi_source_bfs(root, sources):
    if not root or not sources:
        return []
    
    queue = deque()
    visited = set()
    
    # Add all source nodes to queue
    def add_sources(node, level):
        if not node:
            return
        
        if node.val in sources:
            queue.append((node, level))
            visited.add(node.val)
        
        add_sources(node.left, level + 1)
        add_sources(node.right, level + 1)
    
    add_sources(root, 0)
    
    result = []
    while queue:
        node, level = queue.popleft()
        result.append((node.val, level))
        
        for child in [node.left, node.right]:
            if child and child.val not in visited:
                visited.add(child.val)
                queue.append((child, level + 1))
    
    return result
```

### 4. **BFS with Custom Ordering**
```python
def bfs_custom_order(root, comparator):
    if not root:
        return []
    
    import heapq
    
    # Use priority queue instead of regular queue
    pq = [(0, 0, root)]  # (priority, insertion_order, node)
    insertion_order = 0
    result = []
    
    while pq:
        _, _, node = heapq.heappop(pq)
        result.append(node.val)
        
        # Add children with custom priority
        for child in [node.left, node.right]:
            if child:
                priority = comparator(child)
                insertion_order += 1
                heapq.heappush(pq, (priority, insertion_order, child))
    
    return result
```

## Step-by-Step Problem-Solving Approach

### 1. **Identify the Problem Type**
- Is level-by-level processing required?
- Do you need shortest path in tree?
- Are you looking for tree properties?
- Is the solution likely near the root?

### 2. **Choose the Right BFS Variant**
- **Basic BFS**: Simple traversal needs
- **Level-by-level**: When level information matters
- **With tracking**: When you need additional state
- **Space-optimized**: For memory constraints

### 3. **Determine Queue Contents**
- Nodes only
- Nodes with level information
- Nodes with additional state
- Multiple pieces of information per entry

### 4. **Handle Level Processing**
- Process entire level at once
- Track level boundaries
- Maintain level-specific data
- Handle level transitions

### 5. **Optimize if Needed**
- Use next pointers for space efficiency
- Early termination conditions
- Bidirectional search for shortest paths
- Custom priority for specific ordering

## Real-World Applications

### 1. **File System Analysis**
Directory structures represented as trees:
- Each directory is a node, files are leaf nodes
- Root represents starting directory
- BFS traverses to analyze dependencies or find shortest path to files

```python
class DirectoryNode:
    def __init__(self, name, is_directory=True):
        self.name = name
        self.is_directory = is_directory
        self.children = []  # subdirectories and files
        self.size = 0

def analyze_file_system(root_dir, target_file):
    if not root_dir:
        return None
    
    queue = deque([(root_dir, 0, [])])  # (node, depth, path)
    
    while queue:
        current, depth, path = queue.popleft()
        current_path = path + [current.name]
        
        # Found target file
        if current.name == target_file:
            return {
                "path": "/".join(current_path),
                "depth": depth,
                "is_directory": current.is_directory
            }
        
        # Add children to explore
        if current.is_directory:
            for child in current.children:
                queue.append((child, depth + 1, current_path))
    
    return None  # File not found
```

### 2. **Version Control Systems (Git)**
BFS traverses file system tree to identify changes:
- Track revisions and manage branches
- Identify modified files and directories
- Merge changes efficiently

```python
class GitNode:
    def __init__(self, path, file_type, last_modified):
        self.path = path
        self.type = file_type  # 'file' or 'directory'
        self.last_modified = last_modified
        self.children = []
        self.status = 'unchanged'  # 'added', 'modified', 'deleted'

def detect_changes(old_tree, new_tree):
    changes = []
    
    # BFS through both trees simultaneously
    queue = deque([(old_tree, new_tree, "")])
    
    while queue:
        old_node, new_node, path = queue.popleft()
        
        # Compare nodes
        if not old_node and new_node:
            changes.append(("added", path + "/" + new_node.path))
        elif old_node and not new_node:
            changes.append(("deleted", path + "/" + old_node.path))
        elif old_node and new_node:
            if old_node.last_modified != new_node.last_modified:
                changes.append(("modified", path + "/" + new_node.path))
            
            # Compare children
            old_children = {child.path: child for child in old_node.children}
            new_children = {child.path: child for child in new_node.children}
            
            all_paths = set(old_children.keys()) | set(new_children.keys())
            for child_path in all_paths:
                old_child = old_children.get(child_path)
                new_child = new_children.get(child_path)
                queue.append((old_child, new_child, path + "/" + (new_node.path if new_node else old_node.path)))
    
    return changes
```

### 3. **Genealogy and Evolutionary Trees**
BFS analyzes biological relationships:
- Start from specific organism/species
- Traverse ancestral connections breadth-first
- Reconstruct evolutionary histories and genetic relationships

```python
class OrganismNode:
    def __init__(self, species_name, generation, traits):
        self.species = species_name
        self.generation = generation
        self.traits = traits
        self.ancestors = []  # parent species
        self.descendants = []  # child species

def analyze_evolutionary_tree(root_species, target_traits):
    if not root_species:
        return []
    
    related_species = []
    queue = deque([(root_species, 0)])  # (species, generation_distance)
    visited = {root_species.species}
    
    while queue:
        current, distance = queue.popleft()
        
        # Check if species has target traits
        trait_similarity = calculate_trait_similarity(current.traits, target_traits)
        if trait_similarity > 0.7:  # 70% similarity threshold
            related_species.append({
                "species": current.species,
                "generation_distance": distance,
                "similarity": trait_similarity
            })
        
        # Explore descendants (evolutionary tree)
        for descendant in current.descendants:
            if descendant.species not in visited:
                visited.add(descendant.species)
                queue.append((descendant, distance + 1))
    
    return sorted(related_species, key=lambda x: x["similarity"], reverse=True)

def calculate_trait_similarity(traits1, traits2):
    # Simple trait comparison
    common_traits = set(traits1) & set(traits2)
    total_traits = set(traits1) | set(traits2)
    return len(common_traits) / len(total_traits) if total_traits else 0
```

### 4. **Traversing DOM Tree**
Navigate HTML document structure level by level:
- HTML represented as tree structure
- Children of HTML tag become children of tree nodes
- Each level can have any number of nodes

```python
class DOMNode:
    def __init__(self, tag_name, attributes=None, text_content=""):
        self.tag = tag_name
        self.attributes = attributes or {}
        self.text = text_content
        self.children = []
        self.parent = None

def traverse_dom_tree(root_html):
    if not root_html:
        return []
    
    traversal_result = []
    queue = deque([root_html])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            
            # Process current DOM element
            element_info = {
                "tag": node.tag,
                "attributes": node.attributes,
                "text": node.text.strip(),
                "children_count": len(node.children)
            }
            current_level.append(element_info)
            
            # Add children for next level
            for child in node.children:
                queue.append(child)
        
        traversal_result.append(current_level)
    
    return traversal_result

def find_elements_by_class(root, class_name):
    """Find all elements with specific class using BFS"""
    if not root:
        return []
    
    matching_elements = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        
        # Check if node has the target class
        if "class" in node.attributes:
            classes = node.attributes["class"].split()
            if class_name in classes:
                matching_elements.append(node)
        
        # Continue BFS
        for child in node.children:
            queue.append(child)
    
    return matching_elements
```

### 5. **Space-Efficient DOM Traversal with Next Pointers**
Create shadow tree with next pointers to avoid queue memory usage:
```python
def create_shadow_tree_with_next(root):
    """Create shadow tree where each node points to next node at same level"""
    if not root:
        return None
    
    # Start with root level
    leftmost = root
    
    while leftmost:
        current = leftmost
        leftmost = None  # Will be set to first node of next level
        
        while current:
            # Connect children of current node
            if current.children:
                # Connect siblings
                for i in range(len(current.children) - 1):
                    current.children[i].next = current.children[i + 1]
                
                # Set leftmost for next level
                if not leftmost:
                    leftmost = current.children[0]
                
                # Connect to next node's children
                if current.next:
                    # Find next node with children
                    next_with_children = current.next
                    while next_with_children and not next_with_children.children:
                        next_with_children = next_with_children.next
                    
                    if next_with_children and next_with_children.children:
                        current.children[-1].next = next_with_children.children[0]
            
            current = current.next
    
    return root

def traverse_with_next_pointers(root):
    """Traverse tree using next pointers without queue"""
    if not root:
        return []
    
    result = []
    leftmost = root
    
    while leftmost:
        current = leftmost
        current_level = []
        leftmost = None
        
        while current:
            current_level.append(current.tag)
            
            # Set leftmost for next level
            if not leftmost and current.children:
                leftmost = current.children[0]
            
            current = current.next
        
        if current_level:
            result.append(current_level)
    
    return result
```

## Practice Problems

### Beginner Level
1. **Binary Tree Level Order Traversal** - Basic BFS implementation
2. **Maximum Depth of Binary Tree** - Tree height using BFS
3. **Minimum Depth of Binary Tree** - Shortest path to leaf
4. **Average of Levels** - Level-wise calculations
5. **Binary Tree Right Side View** - Rightmost nodes per level

### Intermediate Level
6. **Binary Tree Zigzag Level Order** - Alternating direction traversal
7. **Populating Next Right Pointers** - Connect nodes at same level
8. **Find Bottom Left Tree Value** - Leftmost node in last level
9. **Binary Tree Level Order Traversal II** - Bottom-up level order
10. **Cousins in Binary Tree** - Same level, different parents
11. **Maximum Width of Binary Tree** - Calculate tree width
12. **Even Odd Tree** - Level-based value validation

### Advanced Level
13. **Serialize and Deserialize Binary Tree** - Level-order serialization
14. **Complete Binary Tree Inserter** - Maintain complete tree property
15. **Maximum Level Sum** - Find level with maximum sum
16. **Vertical Order Traversal** - Column-wise tree traversal
17. **Binary Tree Cameras** - Minimum cameras for full coverage
18. **All Nodes Distance K** - Find nodes at distance K

## Common Patterns and Optimizations

### 1. **Level Boundary Detection**
```python
def process_level_boundaries(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)  # Detect level boundary
        level_sum = 0
        
        for i in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            
            # First node in level
            if i == 0:
                # Process first node
                pass
            
            # Last node in level
            if i == level_size - 1:
                # Process last node
                pass
            
            # Add children
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_sum)
    
    return result
```

### 2. **Early Termination**
```python
def bfs_with_early_termination(root, target):
    if not root:
        return -1
    
    if root.val == target:
        return 0
    
    queue = deque([(root, 0)])
    
    while queue:
        node, level = queue.popleft()
        
        if node.val == target:
            return level  # Early termination
        
        for child in [node.left, node.right]:
            if child:
                queue.append((child, level + 1))
    
    return -1
```

### 3. **Memory Optimization**
```python
def memory_optimized_bfs(root):
    """Use generators to reduce memory usage"""
    if not root:
        return
    
    current_level = [root]
    
    while current_level:
        # Yield current level
        yield [node.val for node in current_level]
        
        # Prepare next level
        next_level = []
        for node in current_level:
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        current_level = next_level
```

### 4. **Parallel Level Processing**
```python
def parallel_level_processing(root, process_func):
    """Process multiple nodes at same level in parallel"""
    if not root:
        return []
    
    from concurrent.futures import ThreadPoolExecutor
    
    queue = deque([root])
    results = []
    
    while queue:
        level_nodes = []
        level_size = len(queue)
        
        # Collect all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Process level nodes in parallel
        with ThreadPoolExecutor() as executor:
            level_results = list(executor.map(process_func, level_nodes))
            results.append(level_results)
    
    return results
```

## Time and Space Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Basic BFS | O(n) | O(w) | w = maximum width of tree |
| Level Order | O(n) | O(w) | Store entire level in queue |
| Tree Properties | O(n) | O(w) | Visit all nodes once |
| Shortest Path | O(n) | O(w) | Guaranteed shortest in unweighted |
| Next Pointers | O(n) | O(1) | Space-optimized approach |
| Bidirectional BFS | O(n) | O(w) | Can be faster than regular BFS |

Where:
- n = number of nodes
- w = maximum width of tree
- Best case w = 1 (linear tree)
- Worst case w = n/2 (complete binary tree)

## Common Pitfalls and Solutions

### 1. **Queue Memory Overflow**
```python
# Problem: Very wide trees consume too much memory
# Solution: Use generators or process level by level
def memory_safe_bfs(root):
    if not root:
        return
    
    current_level = [root]
    
    while current_level:
        next_level = []
        
        # Process current level
        for node in current_level:
            process(node)
            
            # Collect next level
            if node.left:
                next_level.append(node.left)
            if node.right:
                next_level.append(node.right)
        
        current_level = next_level
```

### 2. **Incorrect Level Handling**
```python
# Problem: Not properly tracking level boundaries
# Solution: Use level size to process complete levels
def correct_level_processing(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)  # Critical: capture size before loop
        current_level = []
        
        for _ in range(level_size):  # Process exactly this many nodes
            node = queue.popleft()
            current_level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result
```

### 3. **Missing Null Checks**
```python
# Problem: Adding null nodes to queue
# Solution: Always validate before adding
def safe_bfs(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        # Safe addition to queue
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    
    return result
```

### 4. **Infinite Loops with Cycles**
```python
# Problem: Cycles in tree structure cause infinite loops
# Solution: Use visited set (though trees shouldn't have cycles)
def cycle_safe_bfs(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    visited = {id(root)}  # Use object id for uniqueness
    
    while queue:
        node = queue.popleft()
        result.append(node.val)
        
        for child in [node.left, node.right]:
            if child and id(child) not in visited:
                visited.add(id(child))
                queue.append(child)
    
    return result
```

## Tips for Success

1. **Master Level Processing**: Understand how to process complete levels
2. **Queue Management**: Learn to manage queue size and memory
3. **Early Termination**: Stop when target is found
4. **Level Tracking**: Use level information when needed
5. **Memory Awareness**: Consider space complexity for wide trees
6. **Null Handling**: Always validate nodes before processing
7. **Use Collections.deque**: More efficient than list for queue operations
8. **Consider Generators**: For memory-efficient processing

## When NOT to Use BFS

- **Deep Solutions**: When solution is likely near leaves
- **Path Dependencies**: When solution depends on specific path
- **Memory Constraints**: When tree width exceeds memory limits
- **DFS Advantages**: When recursive structure matches problem naturally

## Conclusion

Tree BFS is essential for:
- Level-order tree processing
- Shortest path problems in trees
- Tree property calculations
- DOM and hierarchy traversal
- Version control and file system analysis

Master this pattern by understanding the queue-based level-by-level processing and practicing different variations. The key is recognizing when breadth-first exploration provides the most efficient solution for your tree-based problems.
