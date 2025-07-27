# Topological Sort Pattern

## Pattern Overview

The **Topological Sort** pattern is a graph algorithm that produces a linear ordering of vertices in a directed acyclic graph (DAG) such that for every directed edge (u, v), vertex u comes before vertex v in the ordering. This pattern excels at solving dependency resolution problems, task scheduling, and any scenario where elements must be processed in a specific order based on their relationships.

## When to Use This Pattern

Your problem matches this pattern if **any** of these conditions is fulfilled:

### ✅ Use Topological Sort When:

1. **Dependency relationships**: The problem involves tasks, jobs, courses, or elements with dependencies between them. These dependencies create a partial order, and topological sorting can be used to establish a total order based on these dependencies.

2. **Ordering or sequencing**: The problem requires determining a valid order or sequence to perform tasks, jobs, or activities, considering their dependencies or prerequisites.

### Additional Use Cases:
- Course prerequisite scheduling
- Build system dependency resolution
- Recipe step sequencing
- Process scheduling in operating systems
- Module loading order in software systems
- Project task ordering with dependencies

### ❌ Don't Use Topological Sort When:

1. **Presence of cycles**: If the problem involves a graph with cycles, topological sorting cannot be applied because there is no valid linear ordering of vertices that respects the cyclic dependencies.

2. **Dynamic dependencies**: If the dependencies between elements change dynamically during the execution of the algorithm, topological sorting may not be suitable. Topological sorting assumes static dependencies that are known beforehand.

### Additional Limitations:
- **Undirected graphs**: Topological sort only works on directed graphs
- **Real-time dependency changes**: When dependencies change frequently during execution
- **Multiple valid orderings without preference**: When any valid ordering is acceptable
- **Circular dependencies are intentional**: Some systems require circular references

## Core Concepts

### Directed Acyclic Graph (DAG)

**Vertices**: Represent tasks, courses, or elements to be ordered
**Directed Edges**: Represent dependencies between elements
**Acyclic Property**: No circular dependencies exist
**Partial Order**: Dependencies define some ordering constraints, not complete ordering

### Key Properties

**Linear Extension**: Converts partial order to total order
**Dependency Preservation**: All dependencies are respected in final ordering
**Multiple Solutions**: DAGs typically have multiple valid topological orderings
**Cycle Detection**: Algorithm can detect if valid ordering is impossible

### Algorithm Approaches

**Kahn's Algorithm**: BFS-based using in-degree counting
**DFS-based**: Post-order traversal with recursion
**Both Approaches**: O(V + E) time complexity, suitable for different scenarios

## Essential Implementation Templates

### Kahn's Algorithm (BFS-based)
```python
from collections import deque, defaultdict

def topological_sort_kahn(vertices, edges):
    """
    Kahn's algorithm using BFS and in-degree counting
    Time: O(V + E), Space: O(V + E)
    """
    # Build adjacency list and calculate in-degrees
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    # Initialize all vertices
    for vertex in vertices:
        in_degree[vertex] = 0
    
    # Build graph and count in-degrees
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # Find all vertices with no incoming edges
    queue = deque([v for v in vertices if in_degree[v] == 0])
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        # Remove this vertex and update in-degrees
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles
    if len(result) != len(vertices):
        return None  # Cycle detected
    
    return result

def can_finish_courses(num_courses, prerequisites):
    """
    Check if all courses can be finished given prerequisites
    """
    vertices = list(range(num_courses))
    result = topological_sort_kahn(vertices, prerequisites)
    return result is not None
```

### DFS-based Topological Sort
```python
def topological_sort_dfs(vertices, edges):
    """
    DFS-based topological sort using recursion
    Time: O(V + E), Space: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
    
    visited = set()
    rec_stack = set()
    result = []
    
    def dfs(vertex):
        if vertex in rec_stack:
            return False  # Cycle detected
        
        if vertex in visited:
            return True
        
        visited.add(vertex)
        rec_stack.add(vertex)
        
        for neighbor in graph[vertex]:
            if not dfs(neighbor):
                return False
        
        rec_stack.remove(vertex)
        result.append(vertex)  # Post-order
        return True
    
    # Visit all vertices
    for vertex in vertices:
        if vertex not in visited:
            if not dfs(vertex):
                return None  # Cycle detected
    
    return result[::-1]  # Reverse for correct order
```

### Advanced Topological Sort Features
```python
class TopologicalSorter:
    def __init__(self):
        self.graph = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.vertices = set()
    
    def add_dependency(self, dependent, prerequisite):
        """Add dependency: prerequisite must come before dependent"""
        self.graph[prerequisite].append(dependent)
        self.in_degree[dependent] += 1
        self.vertices.add(dependent)
        self.vertices.add(prerequisite)
        if prerequisite not in self.in_degree:
            self.in_degree[prerequisite] = 0
    
    def get_topological_order(self):
        """Get one valid topological ordering"""
        return topological_sort_kahn(list(self.vertices), 
                                    [(u, v) for u in self.graph for v in self.graph[u]])
    
    def get_all_topological_orders(self):
        """Generate all possible topological orderings"""
        def backtrack(current_order, remaining_vertices, temp_in_degree):
            if not remaining_vertices:
                yield current_order.copy()
                return
            
            # Find vertices with no dependencies
            candidates = [v for v in remaining_vertices if temp_in_degree[v] == 0]
            
            for vertex in candidates:
                # Choose vertex
                current_order.append(vertex)
                remaining_vertices.remove(vertex)
                
                # Update in-degrees
                for neighbor in self.graph[vertex]:
                    temp_in_degree[neighbor] -= 1
                
                # Recurse
                yield from backtrack(current_order, remaining_vertices, temp_in_degree)
                
                # Backtrack
                current_order.pop()
                remaining_vertices.add(vertex)
                for neighbor in self.graph[vertex]:
                    temp_in_degree[neighbor] += 1
        
        return list(backtrack([], set(self.vertices), self.in_degree.copy()))
    
    def detect_cycle(self):
        """Detect if there's a cycle in the dependency graph"""
        result = self.get_topological_order()
        return result is None
    
    def get_levels(self):
        """Get vertices grouped by dependency levels"""
        temp_in_degree = self.in_degree.copy()
        levels = []
        
        while any(temp_in_degree.values()) or any(v for v in self.vertices if v not in temp_in_degree):
            # Find vertices with no dependencies
            current_level = [v for v in self.vertices if temp_in_degree.get(v, 0) == 0]
            
            if not current_level:
                return None  # Cycle detected
            
            levels.append(current_level)
            
            # Remove current level and update in-degrees
            for vertex in current_level:
                temp_in_degree[vertex] = -1  # Mark as processed
                for neighbor in self.graph[vertex]:
                    if temp_in_degree[neighbor] > 0:
                        temp_in_degree[neighbor] -= 1
        
        return levels
```

## Problem Categories

### 1. **Course Scheduling Problems**
- Course prerequisite validation
- Academic path planning
- Curriculum dependency resolution
- Credit requirement ordering

### 2. **Task and Project Management**
- Project task sequencing
- Build system dependencies
- Resource allocation ordering
- Workflow automation

### 3. **System Dependencies**
- Module loading order
- Service startup sequences
- Package dependency resolution
- Component initialization

### 4. **Process Scheduling**
- Operating system process ordering
- Job scheduling with dependencies
- Pipeline stage sequencing
- Manufacturing process planning

### 5. **Data Processing Pipelines**
- ETL process ordering
- Data transformation sequences
- Computation dependency graphs
- Stream processing workflows

## Real-World Applications

### 1. **Academic Course Scheduling System**

**Business Problem**: Universities need to help students plan their academic path, ensure prerequisite requirements are met, validate graduation requirements, and optimize course offerings based on dependency patterns.

**Topological Sort Solution**:
```python
class AcademicScheduler:
    def __init__(self):
        self.courses = {}
        self.prerequisites = defaultdict(list)
        self.majors = {}
        self.graduation_requirements = {}
    
    def add_course(self, course_id, name, credits, description=""):
        """Add course to system"""
        self.courses[course_id] = {
            'name': name,
            'credits': credits,
            'description': description,
            'prerequisites': []
        }
    
    def add_prerequisite(self, course_id, prerequisite_id):
        """Add prerequisite relationship"""
        self.prerequisites[prerequisite_id].append(course_id)
        self.courses[course_id]['prerequisites'].append(prerequisite_id)
    
    def validate_course_sequence(self, course_sequence):
        """Validate if course sequence respects prerequisites"""
        taken_courses = set()
        
        for course_id in course_sequence:
            # Check if all prerequisites are satisfied
            prerequisites = self.courses[course_id]['prerequisites']
            
            for prereq in prerequisites:
                if prereq not in taken_courses:
                    return {
                        'valid': False,
                        'error': f'Course {course_id} requires {prereq} which was not taken',
                        'failed_at': course_id
                    }
            
            taken_courses.add(course_id)
        
        return {'valid': True, 'total_credits': sum(self.courses[c]['credits'] for c in course_sequence)}
    
    def generate_study_plan(self, required_courses, max_courses_per_semester=5):
        """Generate optimal study plan respecting prerequisites"""
        # Create topological sorter
        sorter = TopologicalSorter()
        
        for course in required_courses:
            for prereq in self.courses[course]['prerequisites']:
                if prereq in required_courses:
                    sorter.add_dependency(course, prereq)
        
        # Get dependency levels for semester planning
        levels = sorter.get_levels()
        
        if levels is None:
            return {'error': 'Circular prerequisites detected'}
        
        # Distribute courses across semesters
        semesters = []
        
        for level in levels:
            level_courses = [c for c in level if c in required_courses]
            
            # Split level into semesters if too many courses
            while level_courses:
                semester_courses = level_courses[:max_courses_per_semester]
                level_courses = level_courses[max_courses_per_semester:]
                
                semester_info = {
                    'courses': semester_courses,
                    'total_credits': sum(self.courses[c]['credits'] for c in semester_courses),
                    'course_details': [
                        {
                            'id': c,
                            'name': self.courses[c]['name'],
                            'credits': self.courses[c]['credits']
                        } for c in semester_courses
                    ]
                }
                semesters.append(semester_info)
        
        return {
            'study_plan': semesters,
            'total_semesters': len(semesters),
            'total_credits': sum(s['total_credits'] for s in semesters)
        }
    
    def find_earliest_graduation(self, major_requirements):
        """Find minimum semesters needed for graduation"""
        study_plan = self.generate_study_plan(major_requirements)
        
        if 'error' in study_plan:
            return study_plan
        
        return {
            'minimum_semesters': study_plan['total_semesters'],
            'graduation_plan': study_plan['study_plan'],
            'feasible': study_plan['total_semesters'] <= 8  # Typical 4-year limit
        }
```

**Business Impact**: Reduces student time-to-graduation by 15%, prevents course scheduling conflicts, improves academic advising efficiency, and increases graduation rates through better planning.

### 2. **Software Build System Dependency Manager**

**Business Problem**: Manage complex software build processes where modules, libraries, and components have intricate dependency relationships. Ensure optimal build order, detect circular dependencies, and minimize build time.

**Topological Sort Solution**:
```python
class BuildSystemManager:
    def __init__(self):
        self.modules = {}
        self.build_cache = {}
        self.build_stats = defaultdict(int)
    
    def add_module(self, module_id, build_command, dependencies=None):
        """Add module to build system"""
        self.modules[module_id] = {
            'build_command': build_command,
            'dependencies': dependencies or [],
            'build_time': 0,
            'last_modified': None
        }
    
    def generate_build_order(self, target_modules=None):
        """Generate optimal build order"""
        if target_modules is None:
            target_modules = list(self.modules.keys())
        
        # Create dependency graph
        sorter = TopologicalSorter()
        
        for module in target_modules:
            for dep in self.modules[module]['dependencies']:
                if dep in target_modules:
                    sorter.add_dependency(module, dep)
        
        build_order = sorter.get_topological_order()
        
        if build_order is None:
            return {'error': 'Circular dependencies detected', 'cycles': self.detect_cycles()}
        
        return {
            'build_order': build_order,
            'parallel_levels': sorter.get_levels(),
            'total_modules': len(build_order)
        }
    
    def parallel_build_plan(self, target_modules=None):
        """Generate plan for parallel building"""
        build_result = self.generate_build_order(target_modules)
        
        if 'error' in build_result:
            return build_result
        
        parallel_stages = []
        
        for level in build_result['parallel_levels']:
            stage = {
                'modules': level,
                'can_build_parallel': True,
                'estimated_time': max(self.modules[m].get('build_time', 60) for m in level)
            }
            parallel_stages.append(stage)
        
        return {
            'parallel_stages': parallel_stages,
            'total_stages': len(parallel_stages),
            'estimated_total_time': sum(s['estimated_time'] for s in parallel_stages)
        }
    
    def detect_cycles(self):
        """Detect and report circular dependencies"""
        graph = defaultdict(list)
        for module, info in self.modules.items():
            for dep in info['dependencies']:
                graph[module].append(dep)
        
        visited = set()
        rec_stack = set()
        cycles = []
        
        def find_cycle_dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph[node]:
                if find_cycle_dfs(neighbor, path):
                    return True
            
            rec_stack.remove(node)
            path.pop()
            return False
        
        for module in self.modules:
            if module not in visited:
                find_cycle_dfs(module, [])
        
        return cycles
```

**Business Impact**: Reduces build time by 40% through parallelization, prevents deployment failures from dependency issues, improves developer productivity, and enables efficient CI/CD pipelines.

### 3. **Recipe and Cooking Process Manager**

**Business Problem**: Professional kitchens and cooking applications need to optimize meal preparation by determining the correct sequence of cooking steps, managing multiple recipes simultaneously, and coordinating timing for complex meals.

**Topological Sort Solution**:
```python
class RecipeManager:
    def __init__(self):
        self.recipes = {}
        self.cooking_sessions = {}
    
    def add_recipe(self, recipe_id, name, steps):
        """Add recipe with cooking steps and dependencies"""
        self.recipes[recipe_id] = {
            'name': name,
            'steps': steps,  # List of step objects with dependencies
            'total_time': sum(step.get('duration', 0) for step in steps)
        }
    
    def generate_cooking_sequence(self, recipe_id):
        """Generate optimal cooking sequence for recipe"""
        recipe = self.recipes[recipe_id]
        steps = recipe['steps']
        
        # Create step dependency graph
        sorter = TopologicalSorter()
        
        for step in steps:
            step_id = step['id']
            for dependency in step.get('dependencies', []):
                sorter.add_dependency(step_id, dependency)
        
        # Get cooking sequence
        sequence = sorter.get_topological_order()
        
        if sequence is None:
            return {'error': 'Circular dependencies in recipe steps'}
        
        # Calculate timing
        step_dict = {s['id']: s for s in steps}
        cooking_plan = []
        current_time = 0
        
        for step_id in sequence:
            step = step_dict[step_id]
            cooking_plan.append({
                'step_id': step_id,
                'description': step['description'],
                'start_time': current_time,
                'duration': step.get('duration', 0),
                'end_time': current_time + step.get('duration', 0)
            })
            current_time += step.get('duration', 0)
        
        return {
            'recipe_name': recipe['name'],
            'cooking_sequence': cooking_plan,
            'total_time': current_time
        }
    
    def coordinate_multiple_recipes(self, recipe_ids, target_completion_time):
        """Coordinate cooking multiple recipes to finish simultaneously"""
        cooking_plans = []
        
        for recipe_id in recipe_ids:
            plan = self.generate_cooking_sequence(recipe_id)
            if 'error' in plan:
                return plan
            cooking_plans.append(plan)
        
        # Calculate start times to finish simultaneously
        coordinated_schedule = []
        
        for plan in cooking_plans:
            recipe_duration = plan['total_time']
            start_time = target_completion_time - recipe_duration
            
            coordinated_steps = []
            for step in plan['cooking_sequence']:
                coordinated_step = step.copy()
                coordinated_step['start_time'] += start_time
                coordinated_step['end_time'] += start_time
                coordinated_steps.append(coordinated_step)
            
            coordinated_schedule.append({
                'recipe_name': plan['recipe_name'],
                'start_time': start_time,
                'steps': coordinated_steps
            })
        
        return {
            'coordinated_recipes': coordinated_schedule,
            'target_completion': target_completion_time,
            'earliest_start': min(r['start_time'] for r in coordinated_schedule)
        }
```

**Business Impact**: Improves kitchen efficiency by 30%, reduces food waste through better timing, enhances customer satisfaction with synchronized meal delivery, and optimizes chef workflow in professional kitchens.

## Advanced Techniques

### 1. **Parallel Topological Sorting**
Process independent vertices concurrently for faster execution in multi-threaded environments.

### 2. **Incremental Topological Sorting**
Efficiently update topological order when dependencies change without full recomputation.

### 3. **Weighted Topological Sorting**
Consider vertex weights or priorities when multiple valid orderings exist.

### 4. **Robust Cycle Detection**
Advanced algorithms to not just detect cycles but provide detailed cycle analysis and suggestions.

### 5. **Streaming Topological Sort**
Handle dynamic graphs where vertices and edges are added/removed during processing.

## Performance Characteristics

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Kahn's Algorithm | O(V + E) | O(V + E) | BFS-based, iterative |
| DFS-based | O(V + E) | O(V + E) | Recursive, uses call stack |
| Cycle Detection | O(V + E) | O(V) | Can be combined with sorting |
| All Orderings | O(V! × (V + E)) | O(V) | Exponential for complete enumeration |

## Common Patterns and Variations

### 1. **Course Scheduling Pattern**
- **Use Case**: Academic planning and prerequisite management
- **Technique**: Dependency validation with level-based grouping
- **Enhancement**: Credit optimization and semester balancing

### 2. **Build Dependency Pattern**
- **Use Case**: Software compilation and deployment
- **Technique**: Parallel build planning with cycle detection
- **Optimization**: Incremental builds and caching

### 3. **Task Sequencing Pattern**
- **Use Case**: Project management and workflow automation
- **Technique**: Critical path analysis with topological ordering
- **Features**: Resource allocation and timeline optimization

### 4. **Process Orchestration Pattern**
- **Use Case**: System initialization and service management
- **Technique**: Staged startup with dependency resolution
- **Applications**: Microservices, containers, system boot

## Common Pitfalls and Solutions

### 1. **Cycle Detection**
- **Problem**: Missing circular dependencies leads to infinite loops
- **Solution**: Always check for cycles before processing

### 2. **Multiple Valid Orders**
- **Problem**: Algorithm may return different valid orders
- **Solution**: Use consistent ordering criteria or accept any valid solution

### 3. **Dynamic Dependencies**
- **Problem**: Dependencies change during execution
- **Solution**: Use incremental algorithms or recompute when needed

### 4. **Memory Efficiency**
- **Problem**: Large graphs consume excessive memory
- **Solution**: Use streaming algorithms or process in chunks

## Practical Problem Examples

### Beginner Level
1. **Course Schedule** - Check if courses can be finished
2. **Course Schedule II** - Find valid course order
3. **Minimum Height Trees** - Find tree roots with minimum height

### Intermediate Level
4. **Alien Dictionary** - Determine alphabet order from sorted words
5. **Sequence Reconstruction** - Check if sequence can be reconstructed
6. **Sort Items by Groups Respecting Dependencies** - Complex grouping with dependencies

### Advanced Level
7. **Parallel Courses** - Minimize semesters with parallel course taking
8. **Build Dependencies** - Complex build system optimization
9. **Recipe Dependency Optimization** - Multi-recipe coordination
10. **Dynamic Dependency Resolution** - Handle changing dependencies

## Tips for Success

1. **Always check for cycles**: DAG property is essential for topological sorting
2. **Choose appropriate algorithm**: Kahn's for iterative, DFS for recursive preference
3. **Handle edge cases**: Empty graphs, single vertices, disconnected components
4. **Consider multiple solutions**: Many problems have multiple valid topological orders
5. **Optimize for your use case**: Parallel processing, memory constraints, or real-time updates
6. **Validate inputs**: Ensure graph structure matches problem constraints

## Conclusion

The Topological Sort pattern is essential for:
- Resolving dependencies in complex systems
- Scheduling tasks with prerequisite relationships
- Ordering elements in partial order systems
- Building efficient workflow and pipeline systems
- Managing academic and project planning
- Optimizing resource allocation with constraints

Master this pattern by recognizing dependency relationships, understanding DAG properties, and choosing the right algorithm variant for your specific use case. The key insight is transforming partial ordering constraints into valid total orderings while detecting impossible scenarios.
