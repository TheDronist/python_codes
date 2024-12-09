import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import heapq
from scipy.ndimage import binary_dilation

class AStar3D:
    def __init__(self, grid, start, goal):
        # Initialize the A* search algorithm with the grid, start, and goal positions.
        self.grid = grid
        self.start = start
        self.goal = goal
        self.path = None

    @staticmethod
    def heuristic(a, b):
        # Calculate the Euclidean distance heuristic between two points.
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)

    def find_path(self):
        # Perform the A* search algorithm to find the shortest path.
        # Define 26 possible neighbors including diagonals
        neighbors = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1),
                     (1, 1, 0), (-1, 1, 0), (-1, -1, 0), (1, -1, 0), (0, 1, 1), (0, -1, 1),
                     (0, 1, -1), (0, -1, -1), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
                     (1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1), (1, 1, -1), (-1, 1, -1),
                     (-1, -1, -1), (1, -1, -1)]

        close_set = set()
        came_from = {}
        gscore = {self.start: 0}
        fscore = {self.start: self.heuristic(self.start, self.goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[self.start], self.start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if current == self.goal:
                self.path = []
                while current in came_from:
                    self.path.append(current)
                    current = came_from[current]
                self.path.append(self.start)
                self.path.reverse()
                return self.path

            close_set.add(current)

            for i, j, k in neighbors:
                neighbor = (current[0] + i, current[1] + j, current[2] + k)

                if (0 <= neighbor[0] < self.grid.shape[0] and
                        0 <= neighbor[1] < self.grid.shape[1] and
                        0 <= neighbor[2] < self.grid.shape[2] and
                        self.grid[neighbor[0], neighbor[1], neighbor[2]] == 0):  # Check if the cell is free space

                    if neighbor in close_set:
                        continue

                    tentative_g_score = gscore[current] + self.heuristic(current, neighbor)

                    if (neighbor not in gscore or tentative_g_score < gscore[neighbor]):
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return None

    @staticmethod
    def inflate_map(grid, radius):
        # Inflate obstacles in the grid by a given radius.
        size = 2 * radius + 1
        struct = np.zeros((size, size, size), dtype=bool)
        center = radius
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    if (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2 <= radius ** 2:
                        struct[x, y, z] = True

        inflated_grid = binary_dilation(grid, structure=struct).astype(int)
        return inflated_grid

    def visualize_path(self):
        # Visualize the path in the 3D grid.
        if self.path is None:
            print("No path found to visualize.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the obstacles
        obstacle_indices = np.argwhere(self.grid == 1)
        ax.scatter(obstacle_indices[:, 0], obstacle_indices[:, 1], obstacle_indices[:, 2], c='black', s=1)

        # Plot the path
        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], c='blue', linewidth=2)

        # Start and goal
        ax.scatter(*self.start, c='green', s=100, label='Start')
        ax.scatter(*self.goal, c='red', s=100, label='Goal')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()

def main():
    # Example of 10x10x10 occupancy grid map to test the 3D Astar algorithm
    shape = (10, 10, 10)
    map = np.zeros(shape, dtype=int)
    map[2:4, 2:4, 0:10] = 1
    map[4:6, 5:7, 0:10] = 1

    # Start and goal positions
    start = (1, 1, 1)
    goal = (7, 7, 7)
    
    astar_solver = AStar3D(map, start, goal)
    # Plan the 3D path
    path = astar_solver.find_path()
    # Visualize the occupancy grid with the path
    astar_solver.visualize_path() 

if __name__ == '__main__':
    main()