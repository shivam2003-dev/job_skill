# Bitwise Manipulation Pattern

## Pattern Overview

The **Bitwise Manipulation** pattern leverages binary operations to solve problems efficiently at the bit level. This pattern exploits the binary representation of numbers and uses fundamental bitwise operations (AND, OR, XOR, NOT, shifts) to achieve solutions that are often more efficient than traditional approaches. It's particularly powerful for problems involving unique elements, missing numbers, duplicates, and certain mathematical computations.

## When to Use This Pattern

Your problem matches this pattern if **both** of these conditions are fulfilled:

### ✅ Use Bitwise Manipulation When:

1. **Binary representation**: The input data can be usefully manipulated at the level of the primitive bitwise logical operations in order to compute some portion or all of the solution.

2. **Efficient sorting**: The input data is unsorted, and the answer seems to require sorting, but we want to do better than O(n log n).

### Additional Use Cases:
- Finding unique elements among duplicates
- Detecting missing or extra numbers
- Checking if a number is a power of 2
- Counting set bits or performing bit-level arithmetic
- Space optimization using bit arrays
- Fast mathematical operations (multiplication/division by powers of 2)

### ❌ Don't Use Bitwise Manipulation When:

1. **Complex data structures**: Working with objects, strings, or nested structures where bit operations don't apply
2. **Floating-point arithmetic**: Bitwise operations are primarily for integers
3. **Readability is crucial**: Bitwise code can be harder to understand and maintain
4. **Small datasets**: The optimization may not be worth the complexity

## Core Concepts

### Fundamental Bitwise Operations

**AND (&)**: Returns 1 only if both bits are 1
- Use case: Checking if specific bits are set, clearing bits

**OR (|)**: Returns 1 if at least one bit is 1
- Use case: Setting specific bits, combining flags

**XOR (^)**: Returns 1 if bits are different
- Use case: Finding unique elements, toggling bits, encryption

**NOT (~)**: Inverts all bits
- Use case: Bit masking, complement operations

**Left Shift (<<)**: Moves bits left, equivalent to multiplying by 2^n
- Use case: Fast multiplication, creating powers of 2

**Right Shift (>>)**: Moves bits right, equivalent to dividing by 2^n
- Use case: Fast division, extracting high-order bits

### Key Properties and Tricks

**XOR Properties**:
- a ^ a = 0 (any number XOR with itself is 0)
- a ^ 0 = a (any number XOR with 0 is itself)
- XOR is commutative and associative

**Power of 2 Detection**:
- n & (n-1) == 0 (true only for powers of 2)

**Bit Counting**:
- Brian Kernighan's algorithm: n & (n-1) removes rightmost set bit

## Essential Implementation Templates

### Basic Bit Operations
```python
# Check if bit at position i is set
def is_bit_set(num, i):
    return (num & (1 << i)) != 0

# Set bit at position i
def set_bit(num, i):
    return num | (1 << i)

# Clear bit at position i
def clear_bit(num, i):
    return num & ~(1 << i)

# Toggle bit at position i
def toggle_bit(num, i):
    return num ^ (1 << i)

# Count number of set bits
def count_set_bits(num):
    count = 0
    while num:
        count += 1
        num &= num - 1  # Remove rightmost set bit
    return count
```

### Find Single Number (XOR Pattern)
```python
def single_number(nums):
    """Find the single number that appears once while others appear twice"""
    result = 0
    for num in nums:
        result ^= num
    return result
```

### Missing Number Pattern
```python
def missing_number(nums):
    """Find missing number in array [0, n]"""
    n = len(nums)
    expected_xor = 0
    actual_xor = 0
    
    # XOR all numbers from 0 to n
    for i in range(n + 1):
        expected_xor ^= i
    
    # XOR all numbers in array
    for num in nums:
        actual_xor ^= num
    
    return expected_xor ^ actual_xor
```

### Power of Two Check
```python
def is_power_of_two(n):
    """Check if number is power of 2"""
    return n > 0 and (n & (n - 1)) == 0
```

## Problem Categories

### 1. **Single Number Problems**
- Find unique element among duplicates
- Single number variations (II, III)
- Missing number detection
- Extra number identification

### 2. **Bit Counting and Manipulation**
- Count set bits (Hamming weight)
- Number of 1 bits in range
- Bit reversal operations
- Gray code generation

### 3. **Mathematical Operations**
- Fast multiplication/division by powers of 2
- Add two numbers without arithmetic operators
- Subtract using bitwise operations
- Power calculation using bit manipulation

### 4. **Array Processing**
- Maximum XOR in array
- Subarray XOR queries
- Duplicate detection using bits
- Bit-based sorting algorithms

### 5. **Optimization Problems**
- Space-efficient data structures using bit arrays
- Fast set operations
- Bit masking for state representation
- Dynamic programming with bitmasks

## Real-World Applications

### 1. **Compression Algorithms**

**Business Problem**: Efficiently store and transmit large amounts of data while minimizing storage space and bandwidth usage.

**Bitwise Solution**: Compression techniques like Huffman coding use bitwise algorithms for efficient encoding and decoding of data at the bit level. They facilitate compact representation of variable-length codes by concatenating bits, optimizing storage and transmission.

**Implementation Strategy**:
```python
def huffman_decode_bit(encoded_data, bit_position):
    """Extract bit at specific position for Huffman decoding"""
    byte_index = bit_position // 8
    bit_index = bit_position % 8
    return (encoded_data[byte_index] >> (7 - bit_index)) & 1

def pack_bits(bit_array):
    """Pack array of bits into bytes for compression"""
    result = []
    for i in range(0, len(bit_array), 8):
        byte_val = 0
        for j in range(8):
            if i + j < len(bit_array) and bit_array[i + j]:
                byte_val |= (1 << (7 - j))
        result.append(byte_val)
    return result
```

**Business Impact**: Significantly reduces storage costs, improves transmission speeds, and enables efficient data handling in resource-constrained environments.

### 2. **Status Register Management in Computer Processors**

**Business Problem**: Efficiently track and manage various processor states and flags using minimal memory while providing fast access and updates.

**Bitwise Solution**: In the status register of a computer processor, each bit conveys a distinct significance. For instance, the initial bit of the status register denotes whether the outcome of an arithmetic operation is zero, known as the zero flag. This bit's value can be inspected, altered, or cleared by using a mask of identical length.

**Implementation Strategy**:
```python
class StatusRegister:
    def __init__(self):
        self.register = 0
        # Define flag positions
        self.ZERO_FLAG = 0
        self.CARRY_FLAG = 1
        self.OVERFLOW_FLAG = 2
        self.NEGATIVE_FLAG = 3
    
    def set_flag(self, flag_position):
        """Set specific flag to 1"""
        self.register |= (1 << flag_position)
    
    def clear_flag(self, flag_position):
        """Clear specific flag to 0"""
        self.register &= ~(1 << flag_position)
    
    def check_flag(self, flag_position):
        """Check if specific flag is set"""
        return (self.register & (1 << flag_position)) != 0
    
    def update_arithmetic_flags(self, result):
        """Update flags based on arithmetic operation result"""
        # Clear all flags first
        self.register = 0
        
        # Set zero flag if result is 0
        if result == 0:
            self.set_flag(self.ZERO_FLAG)
        
        # Set negative flag if result is negative
        if result < 0:
            self.set_flag(self.NEGATIVE_FLAG)
```

**Business Impact**: Enables efficient processor state management, reduces memory usage, and provides fast flag operations crucial for system performance.

### 3. **Cryptography and Security**

**Business Problem**: Implement secure encryption and decryption algorithms that can withstand various attack methods while maintaining performance.

**Bitwise Solution**: Cyclic shifts are commonly employed in cryptographic algorithms to introduce confusion and diffusion, enhancing security. By applying cyclic shifts, the relationship between the input and output data becomes complex, making it harder for attackers to decipher the original information.

**Implementation Strategy**:
```python
def left_rotate(value, amount, width=32):
    """Perform left circular rotation"""
    amount %= width
    return ((value << amount) | (value >> (width - amount))) & ((1 << width) - 1)

def right_rotate(value, amount, width=32):
    """Perform right circular rotation"""
    amount %= width
    return ((value >> amount) | (value << (width - amount))) & ((1 << width) - 1)

def simple_encryption(data, key):
    """Simple encryption using XOR and bit rotation"""
    encrypted = []
    for i, byte in enumerate(data):
        # XOR with key
        encrypted_byte = byte ^ key
        # Apply rotation based on position
        encrypted_byte = left_rotate(encrypted_byte, i % 8, 8)
        encrypted.append(encrypted_byte)
    return encrypted

def avalanche_test(function, input_data):
    """Test avalanche effect - small input change causes large output change"""
    original_output = function(input_data)
    
    # Flip one bit in input
    modified_input = input_data ^ 1
    modified_output = function(modified_input)
    
    # Count differing bits in output
    diff = original_output ^ modified_output
    return count_set_bits(diff)
```

**Business Impact**: Provides robust security mechanisms, protects sensitive data, ensures compliance with security standards, and maintains user trust.

### 4. **Hash Functions and Data Integrity**

**Business Problem**: Quickly verify data integrity and detect corruption or tampering in large datasets and network transmissions.

**Bitwise Solution**: Hash functions use bitwise operations to compute checksums like Cyclic Redundancy Check (CRC) and Adler-32. These checksums are used for error detection and data integrity verification.

**Implementation Strategy**:
```python
def crc32_simple(data, polynomial=0xEDB88320):
    """Simplified CRC32 implementation"""
    crc = 0xFFFFFFFF
    
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ polynomial
            else:
                crc >>= 1
    
    return crc ^ 0xFFFFFFFF

def adler32(data):
    """Adler-32 checksum implementation"""
    a, b = 1, 0
    MOD_ADLER = 65521
    
    for byte in data:
        a = (a + byte) % MOD_ADLER
        b = (b + a) % MOD_ADLER
    
    return (b << 16) | a

def verify_data_integrity(original_data, received_data):
    """Verify data integrity using checksums"""
    original_crc = crc32_simple(original_data)
    received_crc = crc32_simple(received_data)
    
    return original_crc == received_crc
```

**Business Impact**: Ensures reliable data transmission, prevents data corruption, enables fast error detection, and maintains system reliability.

### 5. **Network Protocol Optimization**

**Business Problem**: Efficiently process network packets, manage routing tables, and implement fast lookup mechanisms in high-performance networking equipment.

**Bitwise Solution**: Use bit manipulation for IP address processing, subnet calculations, and routing table optimizations.

**Implementation Strategy**:
```python
def ip_to_int(ip_string):
    """Convert IP address string to integer"""
    parts = ip_string.split('.')
    return (int(parts[0]) << 24) | (int(parts[1]) << 16) | (int(parts[2]) << 8) | int(parts[3])

def int_to_ip(ip_int):
    """Convert integer back to IP address string"""
    return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"

def subnet_contains_ip(network_ip, subnet_mask, test_ip):
    """Check if IP address belongs to subnet using bitwise operations"""
    network_int = ip_to_int(network_ip)
    test_int = ip_to_int(test_ip)
    mask_int = (0xFFFFFFFF << (32 - subnet_mask)) & 0xFFFFFFFF
    
    return (network_int & mask_int) == (test_int & mask_int)

def longest_prefix_match(routing_table, destination_ip):
    """Find longest prefix match in routing table"""
    dest_int = ip_to_int(destination_ip)
    best_match = None
    longest_prefix = -1
    
    for network, prefix_len, next_hop in routing_table:
        network_int = ip_to_int(network)
        mask = (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF
        
        if (dest_int & mask) == (network_int & mask) and prefix_len > longest_prefix:
            longest_prefix = prefix_len
            best_match = next_hop
    
    return best_match
```

**Business Impact**: Enables high-speed routing decisions, optimizes network performance, reduces latency, and supports scalable network infrastructure.

## Advanced Techniques

### 1. **Bit Manipulation with DP**
Use bitmasks to represent states in dynamic programming problems, particularly useful for subset problems and traveling salesman variations.

### 2. **Trie with Bit Manipulation**
Implement tries for binary representations to solve maximum XOR problems efficiently.

### 3. **Segment Trees with Lazy Propagation**
Use bitwise operations for range updates and queries on binary properties.

### 4. **Rolling Hash with Bit Operations**
Implement efficient string matching and pattern recognition using bitwise rolling hashes.

### 5. **Parallel Bit Operations**
Leverage SIMD instructions and parallel processing for bulk bit manipulation operations.

## Performance Characteristics

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Basic bit ops | O(1) | O(1) | AND, OR, XOR, shifts |
| Count set bits | O(log n) | O(1) | Using Brian Kernighan's algorithm |
| Find single number | O(n) | O(1) | XOR-based approach |
| Power of 2 check | O(1) | O(1) | Using n & (n-1) |
| Bit reversal | O(log n) | O(1) | Reverse bits in number |

## Common Patterns and Variations

### 1. **XOR Pattern**
- **Use Case**: Finding unique elements, missing numbers
- **Key Property**: a ^ a = 0, a ^ 0 = a
- **Applications**: Single number problems, duplicate detection

### 2. **Bit Masking Pattern**
- **Use Case**: State representation, subset enumeration
- **Technique**: Use bits to represent presence/absence
- **Applications**: Dynamic programming, combinatorial problems

### 3. **Power of 2 Pattern**
- **Use Case**: Efficient mathematical operations
- **Key Check**: n & (n-1) == 0
- **Applications**: Array sizing, hash table implementations

### 4. **Bit Counting Pattern**
- **Use Case**: Hamming weight, population count
- **Technique**: Brian Kernighan's algorithm
- **Applications**: Graph algorithms, optimization problems

### 5. **Bit Reversal Pattern**
- **Use Case**: Mirror operations, FFT algorithms
- **Technique**: Swap bits systematically
- **Applications**: Signal processing, cryptography

## Practical Problem Examples

### Beginner Level
1. **Single Number** - Find unique element using XOR
2. **Number of 1 Bits** - Count set bits in binary representation
3. **Power of Two** - Check if number is power of 2
4. **Missing Number** - Find missing number in sequence

### Intermediate Level
5. **Single Number II** - Find unique among numbers appearing thrice
6. **Bitwise AND of Numbers Range** - Range bitwise operations
7. **Reverse Bits** - Reverse binary representation
8. **Sum of Two Integers** - Add without arithmetic operators
9. **Maximum XOR** - Find maximum XOR of two numbers

### Advanced Level
10. **Maximum XOR of Two Numbers in Array** - Using trie structure
11. **Counting Bits** - Count bits for range of numbers
12. **Gray Code** - Generate Gray code sequence
13. **Minimum XOR Sum** - Optimize XOR operations
14. **Bitwise ORs of Subarrays** - Complex bit manipulation

## Common Pitfalls and Solutions

### 1. **Integer Overflow**
- **Problem**: Bit shifts can cause overflow in fixed-width integers
- **Solution**: Use appropriate data types and bounds checking

### 2. **Signed vs Unsigned**
- **Problem**: Right shift behavior differs for signed numbers
- **Solution**: Be explicit about signed/unsigned operations

### 3. **Endianness Issues**
- **Problem**: Bit order varies across systems
- **Solution**: Use standardized bit manipulation libraries

### 4. **Readability**
- **Problem**: Bitwise code can be cryptic
- **Solution**: Add clear comments and use meaningful variable names

### 5. **Platform Dependencies**
- **Problem**: Bit operations may behave differently across platforms
- **Solution**: Test on target platforms and use portable code

## When NOT to Use Bitwise Manipulation

1. **Floating-point numbers**: Bitwise operations don't apply to float arithmetic
2. **Complex data types**: Objects, strings, and structures need different approaches
3. **Maintainability critical**: When code clarity is more important than performance
4. **Small datasets**: Optimization overhead may not be worthwhile
5. **Platform portability**: When consistent behavior across systems is crucial

## Tips for Success

1. **Master XOR properties**: Essential for many bit manipulation problems
2. **Understand two's complement**: Important for signed number operations
3. **Practice bit pattern recognition**: Common patterns appear frequently
4. **Use bit manipulation utilities**: Many languages provide built-in functions
5. **Test edge cases**: Zero, negative numbers, and boundary values
6. **Document bit layouts**: Clear documentation for complex bit structures
7. **Consider readability**: Balance optimization with code maintainability

## Conclusion

The Bitwise Manipulation pattern is essential for:
- High-performance computing and optimization
- System-level programming and embedded systems
- Cryptography and security applications
- Data compression and encoding algorithms
- Network protocol implementations
- Mathematical computations and algorithms

Master this pattern by understanding fundamental bitwise operations, recognizing common bit manipulation patterns, and practicing with problems that leverage binary representations. The key insight is that many problems can be solved more efficiently by working directly with the binary representation of data rather than using traditional arithmetic or logical approaches.
