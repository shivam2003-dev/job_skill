# Stack Pattern

## Pattern Overview

The **Stack** pattern is a fundamental data structure that follows the **Last In, First Out (LIFO)** principle. It's used to solve problems that require reverse order processing, state tracking, or handling nested structures.

## When to Use This Pattern

Your problem matches this pattern if **any** of these conditions is fulfilled:

### ✅ Use Stack When:

1. **Reverse order processing**: The problem involves processing elements in reverse order or requires the last element added to be processed first.

2. **Nested structures handling**: The problem involves nested structures, like parentheses, brackets, or nested function calls.

3. **State tracking**: The problem requires keeping track of previous states or undoing operations.

4. **Expression evaluation**: The problem involves evaluating expressions.

### ❌ Don't Use Stack When:

1. **Order dependence**: The problem requires either a different order dependence than Last In, First Out (LIFO) or there is no order dependency at all.

2. **Random access**: The problem involves frequent access or modification of elements at arbitrary positions and not just from the end.

3. **Need for searching**: The problem requires efficient searching for elements based on values or properties.

## Core Operations

| Operation | Description | Time Complexity | Space Complexity |
|-----------|-------------|----------------|------------------|
| `push(item)` | Add element to top | O(1) | O(1) |
| `pop()` | Remove and return top element | O(1) | O(1) |
| `peek()/top()` | View top element without removing | O(1) | O(1) |
| `isEmpty()` | Check if stack is empty | O(1) | O(1) |
| `size()` | Get number of elements | O(1) | O(1) |

## Common Problem Categories

### 1. **Parentheses and Bracket Matching**
- Valid parentheses
- Generate parentheses
- Remove invalid parentheses
- Longest valid parentheses

### 2. **Expression Evaluation**
- Infix to postfix conversion
- Evaluate postfix expression
- Basic calculator
- Evaluate reverse Polish notation

### 3. **Monotonic Stack Problems**
- Next greater element
- Daily temperatures
- Largest rectangle in histogram
- Trapping rain water

### 4. **Function Call Simulation**
- Decode strings
- Binary tree traversal (iterative)
- Backtracking problems
- Recursive to iterative conversion

### 5. **State Management**
- Undo/redo operations
- Browser history
- Game state tracking
- Parsing nested structures

## Implementation Templates

### Basic Stack Implementation
```python
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()
    
    def peek(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
```

### Parentheses Validation Template
```python
def is_valid_parentheses(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:  # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:  # Opening bracket
            stack.append(char)
    
    return not stack
```

### Monotonic Stack Template
```python
def next_greater_elements(nums):
    stack = []
    result = [-1] * len(nums)
    
    for i in range(len(nums)):
        while stack and nums[i] > nums[stack[-1]]:
            index = stack.pop()
            result[index] = nums[i]
        stack.append(i)
    
    return result
```

### Expression Evaluation Template
```python
def evaluate_postfix(tokens):
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            b = stack.pop()
            a = stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            elif token == '/':
                stack.append(int(a / b))
        else:
            stack.append(int(token))
    
    return stack[0]
```

## Step-by-Step Problem-Solving Approach

### 1. **Identify the Pattern**
- Does the problem require LIFO processing?
- Are there nested structures?
- Do you need to track previous states?
- Is reverse order processing needed?

### 2. **Choose the Right Stack Type**
- **Simple Stack**: For basic LIFO operations
- **Monotonic Stack**: For finding next/previous greater/smaller elements
- **Stack with Additional Operations**: For problems requiring min/max tracking

### 3. **Design the Algorithm**
- Determine what to push onto the stack
- Identify when to pop from the stack
- Plan the processing logic for each element

### 4. **Handle Edge Cases**
- Empty stack operations
- Stack overflow (if using fixed-size stack)
- Invalid input handling

### 5. **Optimize if Needed**
- Space optimization techniques
- Early termination conditions
- Combining operations

## Common Patterns and Techniques

### 1. **Monotonic Stack**
Used for finding next/previous greater/smaller elements:
```python
# Increasing monotonic stack
while stack and current > stack[-1]:
    # Process the popped element
    stack.pop()
stack.append(current)

# Decreasing monotonic stack
while stack and current < stack[-1]:
    # Process the popped element
    stack.pop()
stack.append(current)
```

### 2. **Stack with Min/Max Tracking**
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self):
        if self.stack.pop() == self.min_stack[-1]:
            self.min_stack.pop()
    
    def get_min(self):
        return self.min_stack[-1]
```

### 3. **Two-Stack Technique**
Used for queue implementation or complex operations:
```python
class QueueUsingStacks:
    def __init__(self):
        self.input_stack = []
        self.output_stack = []
    
    def enqueue(self, item):
        self.input_stack.append(item)
    
    def dequeue(self):
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
        return self.output_stack.pop()
```

## Real-World Applications

### 1. **Function Call Stack**
Programming languages use stacks to manage function calls:
- When a function is called, its context is pushed onto the stack
- When the function completes, it is popped off the stack
- Enables proper return flow and variable scoping

### 2. **Text Editor Undo/Redo Feature**
Stacks are commonly used for undo/redo functionality:
- Each edit operation is pushed onto the undo stack
- Undo operation pops from undo stack and pushes to redo stack
- Allows users to revert to previous states

### 3. **Browser Back and Forward Buttons**
Web browsers use stacks for navigation:
- Visited pages are stored in a stack
- Back button pops the current page
- Forward button pushes pages back onto the stack

### 4. **Call History in Smartphones**
Smartphones maintain call history using stack principles:
- Most recent calls appear first
- LIFO access pattern for call management
- Easy retrieval of recent contacts

### 5. **Expression Evaluation**
Mathematical expression parsing and evaluation:
- Example: In `2 + 3 × 7`, stack ensures `×` operator (higher precedence) is performed before `+` operator
- Handles operator precedence correctly
- Converts infix to postfix notation

### 6. **Transaction History in Banking Apps**
Banking applications use stacks for transaction management:
- Each transaction is pushed onto the stack
- Most recent transactions retrieved first
- Preserves integrity of transaction history
- Enables easy rollback of operations

### 7. **Compiler Design**
- Syntax analysis and parsing
- Symbol table management
- Code generation
- Error handling and recovery

### 8. **Game Development**
- Game state management
- Save/load functionality
- Undo moves in strategy games
- Menu navigation systems

## Practice Problems

### Beginner Level
1. **Valid Parentheses** - Basic bracket matching
2. **Implement Stack using Queues** - Understanding stack operations
3. **Min Stack** - Stack with additional functionality
4. **Baseball Game** - Simple stack operations

### Intermediate Level
5. **Daily Temperatures** - Monotonic stack pattern
6. **Next Greater Element I & II** - Circular array with stack
7. **Evaluate Reverse Polish Notation** - Expression evaluation
8. **Decode String** - Nested structure handling
9. **Asteroid Collision** - Stack-based simulation
10. **Remove All Adjacent Duplicates** - String manipulation with stack

### Advanced Level
11. **Largest Rectangle in Histogram** - Complex monotonic stack
12. **Trapping Rain Water** - Advanced monotonic stack
13. **Basic Calculator I, II, III** - Complex expression evaluation
14. **Exclusive Time of Functions** - Stack for simulation
15. **Valid Parenthesis String** - Advanced bracket matching

## Common Pitfalls and Solutions

### 1. **Empty Stack Operations**
```python
# Problem: Calling pop() on empty stack
if not stack:
    # Handle empty stack case
    return default_value
result = stack.pop()

# Solution: Always check before popping
```

### 2. **Forgetting to Process Remaining Elements**
```python
# Problem: Elements left in stack after main loop
while stack:
    # Process remaining elements
    element = stack.pop()
    # Handle element
```

### 3. **Incorrect Bracket Pairing**
```python
# Problem: Not handling all bracket types
mapping = {')': '(', '}': '{', ']': '['}
# Make sure all opening brackets are in values
```

### 4. **Index vs Value Confusion**
```python
# Be clear about what you're storing
stack.append(i)        # Storing index
stack.append(nums[i])  # Storing value
```

## Time and Space Complexity Analysis

| Problem Type | Time Complexity | Space Complexity | Notes |
|--------------|----------------|------------------|-------|
| Basic Operations | O(1) | O(1) | Per operation |
| Bracket Matching | O(n) | O(n) | Worst case: all opening brackets |
| Expression Evaluation | O(n) | O(n) | Linear scan with stack |
| Monotonic Stack | O(n) | O(n) | Each element pushed/popped once |
| Two Stack Queue | O(1) amortized | O(n) | Worst case O(n) for dequeue |

## Stack vs Other Data Structures

| Feature | Stack | Queue | Array | Linked List |
|---------|-------|-------|--------|-------------|
| Access Pattern | LIFO | FIFO | Random | Sequential |
| Top/Front Access | O(1) | O(1) | O(1) | O(1) |
| Middle Access | O(n) | O(n) | O(1) | O(n) |
| Insertion | O(1) | O(1) | O(n) | O(1) |
| Deletion | O(1) | O(1) | O(n) | O(1) |

## Tips for Success

1. **Understand LIFO**: Always remember Last In, First Out principle
2. **Draw It Out**: Visualize stack operations for complex problems
3. **Check Empty Stack**: Always validate before pop/peek operations
4. **Use Appropriate Data**: Store indices vs values based on need
5. **Consider Monotonic**: For min/max problems, consider monotonic stack
6. **Practice Patterns**: Master common patterns like bracket matching
7. **Think Recursively**: Many stack problems have recursive nature

## Advanced Techniques

### 1. **Stack with Lazy Propagation**
For problems requiring bulk operations:
```python
class LazyStack:
    def __init__(self):
        self.stack = []
        self.lazy_increment = 0
    
    def push(self, val):
        self.stack.append(val - self.lazy_increment)
    
    def pop(self):
        if not self.stack:
            return None
        return self.stack.pop() + self.lazy_increment
    
    def increment(self, k, val):
        if k >= len(self.stack):
            self.lazy_increment += val
        else:
            # Apply increment to specific range
            pass
```

### 2. **Multi-Stack in Single Array**
Implementing multiple stacks in one array:
```python
class MultiStack:
    def __init__(self, stack_count, total_size):
        self.stack_count = stack_count
        self.array = [0] * total_size
        self.sizes = [0] * stack_count
    
    def push(self, stack_num, value):
        if self.is_full(stack_num):
            raise Exception("Stack is full")
        
        index = self.index_of_top(stack_num) + 1
        self.array[index] = value
        self.sizes[stack_num] += 1
```

## Conclusion

The Stack pattern is essential for:
- Handling nested structures and recursive problems
- Managing state and undo operations
- Expression parsing and evaluation
- Algorithm optimization with monotonic stacks

Master this pattern by understanding the LIFO principle and practicing with different problem types. The key is recognizing when reverse order processing or state tracking is needed in your problem.
