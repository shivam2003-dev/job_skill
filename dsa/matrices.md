# Matrix Pattern

## Pattern Overview

The **Matrix** pattern involves solving problems that work with 2D arrays (matrices). This pattern is fundamental in computer science and has applications ranging from image processing to machine learning algorithms.

## When to Use This Pattern

Your problem matches this pattern if the following condition is fulfilled:

### ✅ Use Matrix Pattern When:

**2D array input**: The input data is given as a 2D array. However, some exceptions to this could be problems that have a 2D array as an input, but are solved with some other pattern, e.g., graphs, dynamic programming, etc.

### Common Indicators:
- Input is explicitly a 2D array or matrix
- Problem involves grid-based operations
- Requires row, column, or diagonal processing
- Involves coordinate-based transformations
- Needs spatial relationship analysis

## Matrix Fundamentals

### Basic Properties
- **Dimensions**: m × n (m rows, n columns)
- **Indexing**: Usually 0-based (matrix[i][j])
- **Memory Layout**: Row-major order in most languages
- **Access Pattern**: O(1) for direct access

### Common Matrix Types
1. **Square Matrix**: m = n (equal rows and columns)
2. **Rectangular Matrix**: m ≠ n 
3. **Sparse Matrix**: Most elements are zero
4. **Dense Matrix**: Most elements are non-zero
5. **Identity Matrix**: Diagonal elements are 1, others are 0
6. **Symmetric Matrix**: matrix[i][j] = matrix[j][i]

## Common Problem Categories

### 1. **Matrix Traversal**
- Spiral matrix traversal
- Diagonal traversal
- Zigzag traversal
- Layer-by-layer processing

### 2. **Matrix Transformation**
- Rotate matrix (90°, 180°, 270°)
- Transpose matrix
- Flip matrix (horizontal/vertical)
- Matrix multiplication

### 3. **Search and Find**
- Search in sorted matrix
- Find peaks in matrix
- Locate specific patterns
- Matrix element queries

### 4. **Path Finding**
- Shortest path in grid
- Unique paths counting
- Obstacle avoidance
- Path with constraints

### 5. **Matrix Manipulation**
- Set matrix zeros
- Reshape matrix
- Merge intervals in 2D
- Matrix region operations

### 6. **Mathematical Operations**
- Matrix addition/subtraction
- Matrix multiplication
- Determinant calculation
- Matrix inversion

## Implementation Templates

### Basic Matrix Operations
```python
class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if matrix else 0
    
    def get(self, i, j):
        if 0 <= i < self.rows and 0 <= j < self.cols:
            return self.matrix[i][j]
        return None
    
    def set(self, i, j, value):
        if 0 <= i < self.rows and 0 <= j < self.cols:
            self.matrix[i][j] = value
    
    def is_valid(self, i, j):
        return 0 <= i < self.rows and 0 <= j < self.cols
```

### Matrix Traversal Templates
```python
# Row-wise traversal
def traverse_rows(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            process(matrix[i][j])

# Column-wise traversal
def traverse_columns(matrix):
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            process(matrix[i][j])

# Diagonal traversal (main diagonal)
def traverse_main_diagonal(matrix):
    n = min(len(matrix), len(matrix[0]))
    for i in range(n):
        process(matrix[i][i])

# Spiral traversal
def spiral_traverse(matrix):
    if not matrix or not matrix[0]:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Traverse right
        for j in range(left, right + 1):
            result.append(matrix[top][j])
        top += 1
        
        # Traverse down
        for i in range(top, bottom + 1):
            result.append(matrix[i][right])
        right -= 1
        
        # Traverse left (if valid)
        if top <= bottom:
            for j in range(right, left - 1, -1):
                result.append(matrix[bottom][j])
            bottom -= 1
        
        # Traverse up (if valid)
        if left <= right:
            for i in range(bottom, top - 1, -1):
                result.append(matrix[i][left])
            left += 1
    
    return result
```

### Matrix Transformation Templates
```python
# Rotate matrix 90 degrees clockwise
def rotate_90_clockwise(matrix):
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Reverse each row
    for i in range(n):
        matrix[i].reverse()

# Transpose matrix
def transpose(matrix):
    rows, cols = len(matrix), len(matrix[0])
    transposed = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    
    return transposed

# Set zeros
def set_zeros(matrix):
    rows, cols = len(matrix), len(matrix[0])
    zero_rows, zero_cols = set(), set()
    
    # Find zeros
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 0:
                zero_rows.add(i)
                zero_cols.add(j)
    
    # Set rows to zero
    for i in zero_rows:
        for j in range(cols):
            matrix[i][j] = 0
    
    # Set columns to zero
    for j in zero_cols:
        for i in range(rows):
            matrix[i][j] = 0
```

### Search in Matrix Templates
```python
# Search in row-wise and column-wise sorted matrix
def search_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    
    row, col = 0, len(matrix[0]) - 1
    
    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return True
        elif matrix[row][col] > target:
            col -= 1
        else:
            row += 1
    
    return False

# Binary search in sorted matrix
def search_sorted_matrix(matrix, target):
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = (left + right) // 2
        mid_value = matrix[mid // cols][mid % cols]
        
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False
```

## Direction Vectors and Navigation

### Common Direction Arrays
```python
# 4-directional movement (up, right, down, left)
directions_4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

# 8-directional movement (including diagonals)
directions_8 = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),           (0, 1),
                (1, -1),  (1, 0),  (1, 1)]

# Knight moves in chess
knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2),  (1, 2),  (2, -1),  (2, 1)]

def get_neighbors(matrix, row, col, directions=directions_4):
    neighbors = []
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if (0 <= new_row < len(matrix) and 
            0 <= new_col < len(matrix[0])):
            neighbors.append((new_row, new_col))
    return neighbors
```

## Step-by-Step Problem-Solving Approach

### 1. **Understand the Matrix Structure**
- Determine dimensions (m × n)
- Identify if it's square or rectangular
- Check for any special properties (sorted, symmetric, etc.)

### 2. **Analyze the Problem Requirements**
- What type of operation is needed?
- Is it traversal, transformation, or search?
- Are there constraints on time/space complexity?

### 3. **Choose the Right Approach**
- **Brute Force**: Simple nested loops
- **Optimized**: Use matrix properties for efficiency
- **Divide and Conquer**: For large matrices
- **Dynamic Programming**: For optimization problems

### 4. **Handle Edge Cases**
- Empty matrix
- Single row or single column
- Single element matrix
- Out-of-bounds access

### 5. **Optimize if Needed**
- In-place operations to save space
- Early termination conditions
- Use mathematical properties

## Advanced Techniques

### 1. **Layer-by-Layer Processing**
```python
def process_layers(matrix):
    n = len(matrix)
    for layer in range(n // 2):
        first, last = layer, n - 1 - layer
        
        # Process the layer
        for i in range(first, last):
            offset = i - first
            
            # Save top element
            top = matrix[first][i]
            
            # Top = Left
            matrix[first][i] = matrix[last - offset][first]
            
            # Left = Bottom
            matrix[last - offset][first] = matrix[last][last - offset]
            
            # Bottom = Right
            matrix[last][last - offset] = matrix[i][last]
            
            # Right = Top
            matrix[i][last] = top
```

### 2. **Matrix Exponentiation**
```python
def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Cannot multiply matrices")
    
    result = [[0] * cols_B for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def matrix_power(matrix, n):
    if n == 0:
        size = len(matrix)
        return [[1 if i == j else 0 for j in range(size)] 
                for i in range(size)]
    
    if n == 1:
        return matrix
    
    if n % 2 == 0:
        half_power = matrix_power(matrix, n // 2)
        return matrix_multiply(half_power, half_power)
    else:
        return matrix_multiply(matrix, matrix_power(matrix, n - 1))
```

### 3. **Sparse Matrix Operations**
```python
class SparseMatrix:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = {}  # (row, col): value
    
    def set(self, row, col, value):
        if value != 0:
            self.data[(row, col)] = value
        elif (row, col) in self.data:
            del self.data[(row, col)]
    
    def get(self, row, col):
        return self.data.get((row, col), 0)
    
    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Cannot multiply matrices")
        
        result = SparseMatrix(self.rows, other.cols)
        
        for (i, k), val_a in self.data.items():
            for j in range(other.cols):
                val_b = other.get(k, j)
                if val_b != 0:
                    current = result.get(i, j)
                    result.set(i, j, current + val_a * val_b)
        
        return result
```

## Real-World Applications

### 1. **Image Processing**
Matrices represent images where each pixel's color values are stored:
- **Scaling**: Resize images by matrix interpolation
- **Rotation**: Apply rotation matrices to transform images
- **Translation**: Move images using translation matrices
- **Affine Transformations**: Combined scaling, rotation, and translation
- **Filtering**: Apply convolution matrices for blur, sharpen, edge detection

### 2. **Computer Graphics and Gaming**
Matrices are fundamental for 3D transformations:
- **Object Transformation**: Translate, rotate, and scale 3D objects
- **Camera Projections**: Convert 3D coordinates to 2D screen coordinates
- **Vertex Transformations**: Process vertices in 3D graphics pipelines
- **Lighting Calculations**: Compute realistic lighting and shadows
- **Animation**: Interpolate between transformation matrices

### 3. **Data Analysis and Statistics**
Matrices represent and analyze data sets:
- **Linear Regression**: Use matrix operations for least squares fitting
- **Principal Component Analysis (PCA)**: Dimensionality reduction
- **Factor Analysis**: Identify underlying factors in data
- **Multivariate Analysis**: Handle multiple variables simultaneously
- **Covariance Matrices**: Measure relationships between variables
- **Correlation Matrices**: Quantify statistical dependencies

### 4. **Machine Learning**
Matrices are central to ML algorithms:
- **Linear Regression**: Weight matrices and feature vectors
- **Logistic Regression**: Sigmoid function with matrix operations
- **Support Vector Machines (SVM)**: Kernel matrices and optimization
- **Neural Networks**: Weight matrices and activation functions
- **Dimensionality Reduction**: PCA, t-SNE, and manifold learning
- **Deep Learning**: Tensor operations and backpropagation

### 5. **Additional Applications**
- **Cryptography**: Matrix-based encryption algorithms
- **Network Analysis**: Adjacency matrices for graph algorithms
- **Economics**: Input-output models and optimization
- **Physics Simulations**: Finite element analysis and modeling
- **Signal Processing**: Fourier transforms and filtering

## Practice Problems

### Beginner Level
1. **Transpose Matrix** - Basic matrix manipulation
2. **Reshape Matrix** - Array reshaping operations
3. **Toeplitz Matrix** - Pattern recognition in matrices
4. **Flipping an Image** - Simple transformations
5. **Island Perimeter** - Basic grid traversal

### Intermediate Level
6. **Spiral Matrix I & II** - Complex traversal patterns
7. **Rotate Image** - In-place matrix rotation
8. **Set Matrix Zeroes** - Conditional matrix modification
9. **Search a 2D Matrix** - Efficient searching techniques
10. **Number of Islands** - Connected components (DFS/BFS)
11. **Valid Sudoku** - Constraint validation
12. **Game of Life** - State transition simulation

### Advanced Level
13. **Largest Rectangle in Histogram** - Using matrix properties
14. **Maximal Rectangle** - Dynamic programming with matrices
15. **Dungeon Game** - Optimization with constraints
16. **Cherry Pickup** - Advanced dynamic programming
17. **Shortest Path in Binary Matrix** - Pathfinding algorithms
18. **Robot Room Cleaner** - Exploration with limited information

## Common Patterns and Optimizations

### 1. **In-Place Operations**
```python
# Rotate matrix in place
def rotate_in_place(matrix):
    n = len(matrix)
    
    # Process layer by layer
    for layer in range(n // 2):
        first, last = layer, n - 1 - layer
        for i in range(first, last):
            offset = i - first
            
            # Save top
            top = matrix[first][i]
            
            # Rotate elements
            matrix[first][i] = matrix[last - offset][first]
            matrix[last - offset][first] = matrix[last][last - offset]
            matrix[last][last - offset] = matrix[i][last]
            matrix[i][last] = top
```

### 2. **Space-Efficient Algorithms**
```python
# Set zeros using first row and column as markers
def set_zeros_optimized(matrix):
    rows, cols = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(cols))
    first_col_zero = any(matrix[i][0] == 0 for i in range(rows))
    
    # Use first row and column as markers
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0
    
    # Set zeros based on markers
    for i in range(1, rows):
        for j in range(1, cols):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0
    
    # Handle first row and column
    if first_row_zero:
        for j in range(cols):
            matrix[0][j] = 0
    
    if first_col_zero:
        for i in range(rows):
            matrix[i][0] = 0
```

### 3. **Boundary Handling**
```python
def safe_access(matrix, row, col, default=0):
    if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]):
        return matrix[row][col]
    return default

def process_with_padding(matrix, kernel):
    rows, cols = len(matrix), len(matrix[0])
    k_size = len(kernel)
    pad = k_size // 2
    
    result = [[0] * cols for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            value = 0
            for ki in range(k_size):
                for kj in range(k_size):
                    mi, mj = i + ki - pad, j + kj - pad
                    value += safe_access(matrix, mi, mj) * kernel[ki][kj]
            result[i][j] = value
    
    return result
```

## Time and Space Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Access Element | O(1) | O(1) | Direct indexing |
| Traverse Matrix | O(m×n) | O(1) | Visit each element once |
| Matrix Multiplication | O(m×n×p) | O(m×p) | Standard algorithm |
| Transpose | O(m×n) | O(m×n) | New matrix needed |
| Rotate 90° | O(n²) | O(1) | In-place for square matrix |
| Search (Sorted) | O(m+n) | O(1) | Start from corner |
| Binary Search | O(log(m×n)) | O(1) | Treat as 1D array |

## Tips for Success

1. **Understand Indexing**: Master 0-based vs 1-based indexing
2. **Visualize**: Draw small examples to understand patterns
3. **Check Bounds**: Always validate array access
4. **Consider In-Place**: Look for space-efficient solutions
5. **Use Direction Arrays**: Simplify neighbor traversal
6. **Think Layer-wise**: For rotation and spiral problems
7. **Mathematical Insights**: Use matrix properties when possible
8. **Handle Edge Cases**: Empty, single element, single row/column

## Common Pitfalls and Solutions

### 1. **Index Out of Bounds**
```python
# Problem: Accessing invalid indices
# Solution: Always validate bounds
def safe_get(matrix, i, j):
    if 0 <= i < len(matrix) and 0 <= j < len(matrix[0]):
        return matrix[i][j]
    return None
```

### 2. **Modifying While Iterating**
```python
# Problem: Changing matrix while traversing
# Solution: Use separate result matrix or mark changes
result = [[matrix[i][j] for j in range(cols)] for i in range(rows)]
# Apply changes to result
```

### 3. **Incorrect Dimension Handling**
```python
# Problem: Assuming square matrix
# Solution: Handle rectangular matrices properly
rows, cols = len(matrix), len(matrix[0]) if matrix else 0
```

### 4. **Memory Optimization Mistakes**
```python
# Problem: Creating unnecessary copies
# Solution: Reuse existing space when possible
# In-place operations save O(n) space
```

## Conclusion

The Matrix pattern is fundamental for:
- 2D data structure manipulation
- Image and graphics processing
- Mathematical computations
- Machine learning algorithms
- Grid-based problem solving

Master this pattern by practicing different traversal methods, transformation techniques, and optimization strategies. Understanding how to efficiently work with 2D arrays is crucial for many algorithmic challenges and real-world applications.
