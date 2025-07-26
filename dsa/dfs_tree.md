# Tree Depth-First Search (DFS) Pattern

## Pattern Overview

The **Tree Depth-First Search (DFS)** pattern is a fundamental tree traversal technique that explores as far as possible along each branch before backtracking. It's particularly effective for problems involving path exploration, hierarchical structures, and scenarios where solutions are likely to be found deeper in the tree.

## When to Use This Pattern

Your problem matches this pattern if **any** of these conditions is fulfilled:

### ✅ Use Tree DFS When:

1. **Tree data structure**: The input data is in the form of a tree, or the cost of transforming it into a tree is low.

2. **Balanced/low branching factor**: The tree is balanced or has a low branching factor, where DFS provides efficient exploration due to less recursion or backtracking.

3. **Hierarchical structures**: We deal with hierarchical structures like organizational charts or family trees, where traversing from parent to child nodes or vice versa is essential for problem-solving.

4. **Solution near the leaves**: We have reason to believe that the solution is near the leaves of the tree.

5. **Traversal along paths**: Components of the solution are listed along paths from the root to the leaves, and finding the optimal solution requires traversal along these paths. A classic example of this is finding the height of a given tree.

6. **Explore all possible paths**: The problem requires exploring all possible paths in the tree to find a solution or enumerate all valid solutions.

### ❌ When DFS Might Not Be Optimal:

- Finding shortest paths (BFS is better)
- Level-order processing requirements
- Unbalanced trees with very deep paths
- Memory constraints with deep recursion

## Tree DFS Fundamentals

### Core Concepts
- **Recursion**: Natural fit for tree structures
- **Stack-based**: Uses call stack (recursive) or explicit stack (iterative)
- **Path Tracking**: Maintains current path from root to current node
- **Backtracking**: Returns to parent after exploring children

### Tree Node Definition
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

## DFS Traversal Types

### 1. **Preorder Traversal (Root → Left → Right)**
- Process root first, then children
- Used for: Tree copying, prefix expression evaluation
```python
def preorder(root):
    if not root:
        return []
    
    result = [root.val]
    result.extend(preorder(root.left))
    result.extend(preorder(root.right))
    return result
```

### 2. **Inorder Traversal (Left → Root → Right)**
- Process left subtree, root, then right subtree
- Used for: BST sorted order, expression trees
```python
def inorder(root):
    if not root:
        return []
    
    result = []
    result.extend(inorder(root.left))
    result.append(root.val)
    result.extend(inorder(root.right))
    return result
```

### 3. **Postorder Traversal (Left → Right → Root)**
- Process children first, then root
- Used for: Tree deletion, calculating sizes, dependency resolution
```python
def postorder(root):
    if not root:
        return []
    
    result = []
    result.extend(postorder(root.left))
    result.extend(postorder(root.right))
    result.append(root.val)
    return result
```

## Implementation Templates

### Basic Recursive DFS Template
```python
def dfs_recursive(root):
    if not root:
        return  # Base case
    
    # Process current node (preorder)
    process(root.val)
    
    # Recursively explore children
    dfs_recursive(root.left)
    dfs_recursive(root.right)
    
    # Optional: process after children (postorder)
    # cleanup(root.val)
```

### Iterative DFS Template
```python
def dfs_iterative(root):
    if not root:
        return
    
    stack = [root]
    
    while stack:
        node = stack.pop()
        
        # Process current node
        process(node.val)
        
        # Add children to stack (right first for left-to-right processing)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
```

### Path Tracking Template
```python
def dfs_with_path(root, target, path=[]):
    if not root:
        return False
    
    # Add current node to path
    path.append(root.val)
    
    # Check if target found
    if root.val == target:
        return True
    
    # Search in children
    if (dfs_with_path(root.left, target, path) or 
        dfs_with_path(root.right, target, path)):
        return True
    
    # Backtrack: remove current node from path
    path.pop()
    return False
```

### Tree Height Template
```python
def max_depth(root):
    if not root:
        return 0
    
    left_depth = max_depth(root.left)
    right_depth = max_depth(root.right)
    
    return 1 + max(left_depth, right_depth)
```

### Path Sum Template
```python
def has_path_sum(root, target_sum):
    if not root:
        return False
    
    # Leaf node check
    if not root.left and not root.right:
        return root.val == target_sum
    
    # Recursive check on children
    remaining = target_sum - root.val
    return (has_path_sum(root.left, remaining) or 
            has_path_sum(root.right, remaining))
```

## Common Problem Categories

### 1. **Tree Validation**
- Validate binary search tree
- Check if tree is balanced
- Verify tree symmetry
- Detect identical trees

### 2. **Path Problems**
- Root-to-leaf path sum
- Maximum path sum
- All root-to-leaf paths
- Path with given sum

### 3. **Tree Construction**
- Build tree from traversals
- Clone/copy trees
- Merge trees
- Convert to other structures

### 4. **Tree Modification**
- Invert/mirror tree
- Flatten tree to linked list
- Prune tree
- Insert/delete nodes

### 5. **Tree Queries**
- Lowest common ancestor
- Distance between nodes
- Kth smallest/largest element
- Tree diameter

### 6. **Tree Serialization**
- Serialize to string
- Deserialize from string
- Encode/decode trees

## Advanced DFS Techniques

### 1. **Morris Traversal (O(1) Space)**
```python
def morris_inorder(root):
    result = []
    current = root
    
    while current:
        if not current.left:
            result.append(current.val)
            current = current.right
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right
            
            if not predecessor.right:
                # Make thread
                predecessor.right = current
                current = current.left
            else:
                # Remove thread
                predecessor.right = None
                result.append(current.val)
                current = current.right
    
    return result
```

### 2. **DFS with Memoization**
```python
def dfs_memo(root, memo={}):
    if not root:
        return 0
    
    if root in memo:
        return memo[root]
    
    # Compute result
    result = compute_result(root)
    memo[root] = result
    
    return result
```

### 3. **Multi-way DFS (N-ary Trees)**
```python
class NaryNode:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children or []

def dfs_nary(root):
    if not root:
        return
    
    # Process current node
    process(root.val)
    
    # Recursively process all children
    for child in root.children:
        dfs_nary(child)
```

### 4. **DFS with State Passing**
```python
def dfs_with_state(root, current_sum=0, max_sum=[float('-inf')]):
    if not root:
        return
    
    current_sum += root.val
    
    # Update global state
    if not root.left and not root.right:  # Leaf
        max_sum[0] = max(max_sum[0], current_sum)
    
    # Continue DFS
    dfs_with_state(root.left, current_sum, max_sum)
    dfs_with_state(root.right, current_sum, max_sum)
```

## Step-by-Step Problem-Solving Approach

### 1. **Identify the Problem Type**
- Is it a tree traversal problem?
- Do you need to find paths?
- Are you modifying the tree?
- Do you need to validate something?

### 2. **Choose the Right DFS Type**
- **Preorder**: For copying, prefix operations
- **Inorder**: For BST operations, sorted output
- **Postorder**: For cleanup, size calculation, dependency resolution

### 3. **Determine Base Cases**
- Empty tree (null root)
- Leaf nodes
- Single node trees

### 4. **Design the Recursion**
- What to do at current node?
- How to combine results from children?
- What state to pass down?

### 5. **Handle Edge Cases**
- Null inputs
- Single node
- Unbalanced trees
- Deep recursion limits

### 6. **Optimize if Needed**
- Iterative implementation for deep trees
- Memoization for overlapping subproblems
- Early termination conditions

## Real-World Applications

### 1. **Find Products in a Price Range**
Convert product prices into a binary search tree and perform preorder traversal:
- Start at root and check if value lies in range
- If value ≥ lower bound, traverse left child
- If value ≤ upper bound, traverse right child
- Add valid products to result array

```python
def find_products_in_range(root, low, high, result):
    if not root:
        return
    
    # Check if current product is in range
    if low <= root.price <= high:
        result.append(root.product)
    
    # Traverse left if root.price >= low
    if root.price >= low:
        find_products_in_range(root.left, low, high, result)
    
    # Traverse right if root.price <= high
    if root.price <= high:
        find_products_in_range(root.right, low, high, result)
```

### 2. **Dependency Resolution**
In software project dependency graphs:
- Each module is a node
- Dependencies are directed edges
- DFS produces topological ordering
- Ensures modules appear before their dependencies

```python
def resolve_dependencies(dependency_graph):
    visited = set()
    result = []
    
    def dfs(module):
        if module in visited:
            return
        
        visited.add(module)
        
        # Visit all dependencies first
        for dependency in dependency_graph.get(module, []):
            dfs(dependency)
        
        # Add current module after its dependencies
        result.append(module)
    
    for module in dependency_graph:
        dfs(module)
    
    return result
```

### 3. **Syntax Tree Analysis**
In compilers and interpreters for code analysis:
- Source code represented as syntax trees
- DFS traverses for code generation, optimization, analysis
- Natural recursive structure matches code hierarchy

```python
class ASTNode:
    def __init__(self, node_type, value=None, children=None):
        self.type = node_type
        self.value = value
        self.children = children or []

def analyze_syntax_tree(ast_node):
    if not ast_node:
        return
    
    # Process current node based on type
    if ast_node.type == "FUNCTION":
        analyze_function(ast_node)
    elif ast_node.type == "VARIABLE":
        analyze_variable(ast_node)
    elif ast_node.type == "EXPRESSION":
        analyze_expression(ast_node)
    
    # Recursively analyze children
    for child in ast_node.children:
        analyze_syntax_tree(child)
```

### 4. **File System Traversal**
Navigate directory structures:
```python
def traverse_filesystem(directory):
    # Process current directory
    process_directory(directory)
    
    # Recursively traverse subdirectories
    for subdirectory in directory.subdirectories:
        traverse_filesystem(subdirectory)
    
    # Process files in current directory
    for file in directory.files:
        process_file(file)
```

### 5. **Organizational Hierarchy**
Navigate company organizational charts:
```python
def analyze_org_chart(employee):
    # Process current employee
    analyze_employee_data(employee)
    
    # Recursively analyze all direct reports
    for report in employee.direct_reports:
        analyze_org_chart(report)
```

## Practice Problems

### Beginner Level
1. **Maximum Depth of Binary Tree** - Basic DFS traversal
2. **Invert Binary Tree** - Tree modification
3. **Same Tree** - Tree comparison
4. **Symmetric Tree** - Tree validation
5. **Path Sum** - Root-to-leaf path problems

### Intermediate Level
6. **Binary Tree Maximum Path Sum** - Advanced path problems
7. **Lowest Common Ancestor** - Tree queries
8. **Serialize and Deserialize Binary Tree** - Tree serialization
9. **Validate Binary Search Tree** - Tree validation
10. **Binary Tree Right Side View** - Modified traversal
11. **Count Good Nodes in Binary Tree** - Path-dependent counting
12. **Sum Root to Leaf Numbers** - Path value calculation

### Advanced Level
13. **Binary Tree Cameras** - Optimization with DFS
14. **Distribute Coins in Binary Tree** - Resource distribution
15. **Binary Tree Maximum Width** - Level-based with DFS
16. **Recover Binary Search Tree** - Tree correction
17. **House Robber III** - Dynamic programming on trees
18. **Binary Tree Coloring Game** - Game theory with trees

## Common Patterns and Optimizations

### 1. **Early Termination**
```python
def find_target_dfs(root, target):
    if not root:
        return False
    
    if root.val == target:
        return True  # Early termination
    
    # Continue search only if not found
    return (find_target_dfs(root.left, target) or 
            find_target_dfs(root.right, target))
```

### 2. **Result Accumulation**
```python
def collect_leaves(root, leaves):
    if not root:
        return
    
    if not root.left and not root.right:
        leaves.append(root.val)
        return
    
    collect_leaves(root.left, leaves)
    collect_leaves(root.right, leaves)
```

### 3. **Parent Tracking**
```python
def dfs_with_parent(root, parent=None):
    if not root:
        return
    
    # Process node with parent information
    process_node(root, parent)
    
    # Continue DFS with current node as parent
    dfs_with_parent(root.left, root)
    dfs_with_parent(root.right, root)
```

### 4. **Level Tracking**
```python
def dfs_with_level(root, level=0):
    if not root:
        return
    
    # Process node at current level
    process_at_level(root, level)
    
    # Continue to next level
    dfs_with_level(root.left, level + 1)
    dfs_with_level(root.right, level + 1)
```

## Time and Space Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Basic Traversal | O(n) | O(h) | h = height of tree |
| Path Finding | O(n) | O(h) | Worst case: visit all nodes |
| Tree Height | O(n) | O(h) | Recursive call stack |
| Tree Validation | O(n) | O(h) | Check all nodes |
| Morris Traversal | O(n) | O(1) | Constant space |
| Iterative DFS | O(n) | O(h) | Explicit stack |

Where:
- n = number of nodes
- h = height of tree
- Best case h = log(n) (balanced tree)
- Worst case h = n (skewed tree)

## Common Pitfalls and Solutions

### 1. **Stack Overflow in Deep Trees**
```python
# Problem: Deep recursion causes stack overflow
# Solution: Use iterative approach
def dfs_iterative_safe(root):
    if not root:
        return
    
    stack = [root]
    while stack:
        node = stack.pop()
        process(node)
        
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
```

### 2. **Incorrect Base Case Handling**
```python
# Problem: Not handling null nodes properly
# Solution: Always check for null first
def dfs_safe(root):
    if not root:  # Always check first
        return default_value
    
    # Process node
    return compute_result(root)
```

### 3. **Modifying Tree During Traversal**
```python
# Problem: Changing tree structure while traversing
# Solution: Collect nodes first, then modify
def safe_tree_modification(root):
    nodes_to_modify = []
    
    def collect_nodes(node):
        if node:
            if should_modify(node):
                nodes_to_modify.append(node)
            collect_nodes(node.left)
            collect_nodes(node.right)
    
    collect_nodes(root)
    
    # Now safely modify collected nodes
    for node in nodes_to_modify:
        modify_node(node)
```

### 4. **Incorrect Path Tracking**
```python
# Problem: Path state not properly maintained
# Solution: Use backtracking correctly
def path_tracking_correct(root, path, all_paths):
    if not root:
        return
    
    path.append(root.val)  # Add to path
    
    if not root.left and not root.right:  # Leaf
        all_paths.append(path[:])  # Copy current path
    
    path_tracking_correct(root.left, path, all_paths)
    path_tracking_correct(root.right, path, all_paths)
    
    path.pop()  # Backtrack: remove from path
```

## Tips for Success

1. **Master the Basics**: Understand all three traversal orders
2. **Practice Recursion**: Get comfortable with recursive thinking
3. **Visualize**: Draw trees to understand the problem
4. **Handle Nulls**: Always check for null nodes first
5. **Think Base Cases**: Identify stopping conditions clearly
6. **Use Helper Functions**: Pass additional parameters when needed
7. **Consider Iterative**: For very deep trees, use iterative approach
8. **Track State**: Pass down or accumulate state as needed

## When NOT to Use DFS

- **Shortest Path**: Use BFS for unweighted shortest paths
- **Level Processing**: When you need to process nodes level by level
- **Memory Constraints**: When recursion depth might be too large
- **Broad Trees**: When tree is very wide but shallow

## Conclusion

Tree DFS is essential for:
- Hierarchical data processing
- Path-based problem solving
- Tree validation and modification
- Recursive algorithm design
- Dependency resolution
- Syntax analysis

Master this pattern by understanding the recursive nature of trees and practicing different traversal strategies. The key is recognizing when depth-first exploration provides the most efficient solution to your tree-based problems.
