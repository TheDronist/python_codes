import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AStar2D:
    def __init__(self, matrix, start, goal):
        self.matrix = matrix
        self.start = start
        self.goal = goal
        self.path = []

    @staticmethod
    def heuristic(a, b):
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def find_path(self):
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        close_set = set()
        came_from = {}
        gscore = {self.start: 0}
        fscore = {self.start: self.heuristic(self.start, self.goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[self.start], self.start))

        while oheap:
            current = heapq.heappop(oheap)[1]
            if current == self.goal:
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(self.start)  # Include start point
                self.path = data[::-1]  # Reverse to get path from start to goal
                return True

            close_set.add(current)
            for i, j in neighbors:
                neighbor = current[0] + i, current[1] + j
                tentative_g_score = gscore[current] + self.heuristic(current, neighbor)

                if 0 <= neighbor[0] < self.matrix.shape[0] and 0 <= neighbor[1] < self.matrix.shape[1]:
                    if self.matrix[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    continue

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue

                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        self.path = []
        return False

    def visualize_path(self):
        if not self.path:
            print("No path to visualize.")
            return

        matris_with_path = np.copy(self.matrix)
        for point in self.path:
            matris_with_path[point[0], point[1]] = 2  # Assign a unique value for the path

        cmap = ListedColormap(['white', 'black', 'red'])  # 0: white, 1: black, 2: red
        plt.imshow(matris_with_path, cmap=cmap, interpolation='nearest')
        plt.title('Matrix and Path')
        plt.show()


def main():
    # Example of 10x10 occupancy grid map to test the 2D Astar algorithm
    shape = (10, 10)
    map = np.zeros(shape, dtype=int)
    map[2:4, 2:4] = 1
    map[4:6, 5:7] = 1

    # Start and goal positions
    start = (0, 0)
    goal = (8, 8)

    astar_solver = AStar2D(map, start, goal)
    if astar_solver.find_path():
        print("Path found:")
        astar_solver.visualize_path()
        print(astar_solver.path)
    else:
        print("No path found")


if __name__ == '__main__':
    main()