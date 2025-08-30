import heapq

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost from this node to end
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self): # Added for set/dict usage
        return hash(self.position)

def astar(maze, start, end):
    """
    Finds the shortest path from start to end in a maze using the A* algorithm.

    Args:
        maze (list of list of int): A 2D grid where 0 represents a walkable path
                                    and 1 represents an obstacle.
        start (tuple): The starting coordinates (row, col).
        end (tuple): The ending coordinates (row, col).

    Returns:
        list of tuple: The path from start to end, or None if no path is found.
    """
    start_node = Node(start)
    end_node = Node(end)

    open_list = [] # Min-heap for (f_cost, node)
    open_list_positions = {start_node.position: start_node} # Dictionary to quickly access nodes in open_list by position
    closed_list_positions = set() # Set to store positions of nodes already evaluated

    heapq.heappush(open_list, (start_node.f, start_node))

    while open_list:
        f_cost, current_node = heapq.heappop(open_list)

        # If we already processed a better path to this node, skip
        if current_node.position in closed_list_positions:
            continue

        closed_list_positions.add(current_node.position)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        (x, y) = current_node.position
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

        for next_pos in neighbors:
            (nx, ny) = next_pos

            if not (0 <= nx < len(maze) and 0 <= ny < len(maze[0])):
                continue

            if maze[nx][ny] != 0:
                continue

            if next_pos in closed_list_positions:
                continue

            new_g = current_node.g + 1
            new_h = abs(nx - end_node.position[0]) + abs(ny - end_node.position[1])
            new_f = new_g + new_h

            # If the neighbor is already in open_list_positions and we found a worse path, skip
            if next_pos in open_list_positions and new_g >= open_list_positions[next_pos].g:
                continue

            # This is either a new node or a better path to an existing node in open_list
            new_node = Node(next_pos, current_node)
            new_node.g = new_g
            new_node.h = new_h
            new_node.f = new_f

            heapq.heappush(open_list, (new_node.f, new_node))
            open_list_positions[next_pos] = new_node # Update or add the node with the better path

    return None  # No path found

if __name__ == "__main__":
    # Example Maze: 0 = walkable, 1 = obstacle
    maze = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    end = (9, 9)

    print(f"Finding path from {start} to {end} in the maze:")
    path = astar(maze, start, end)

    if path:
        print("Path found:")
        for r, c in path:
            print(f"({r}, {c})", end=" -> ")
        print("End")

        # Visualize the path on the maze
        path_maze = [row[:] for row in maze] # Create a copy
        for r, c in path:
            if (r, c) != start and (r, c) != end:
                path_maze[r][c] = '*'
        path_maze[start[0]][start[1]] = 'S'
        path_maze[end[0]][end[1]] = 'E'

        print("\nMaze with path:")
        for row in path_maze:
            print(" ".join(map(str, row)))
    else:
        print("No path found!")

    print("\n--- Example 2: No Path ---")
    maze_no_path = [
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 0, 1]
    ]
    start_no_path = (0, 0)
    end_no_path = (3, 3)
    print(f"Finding path from {start_no_path} to {end_no_path} in the maze:")
    path_no_path = astar(maze_no_path, start_no_path, end_no_path)
    if path_no_path:
        print("Path found:", path_no_path)
    else:
        print("No path found!")
