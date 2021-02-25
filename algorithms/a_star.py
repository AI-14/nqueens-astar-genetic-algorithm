from typing import Optional, List, Tuple
import random, copy, time, heapq
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'figure.max_open_warning': 0})


class State:
    """Class that holds the state of a board. 
    It holds queen's positions on the board, fitness, and total cost.
    """

    def __init__(self, N: Optional[int] = 4) -> None:
        self.queen_positions: List[int] = [random.randint(1,N) for i in range(N)]
        self.heuristic: int  = self.calculate_heuristic()
        self.g: int = 0
        self.total_cost: int = self.heuristic + self.g

    def calculate_heuristic(self) -> int:
        """Function to calculate the fitness of the state.
        Fitness is calculated as => Number of pairs of queens attacking each other. 
        """

        h: int = 0
        N: int = len(self.queen_positions)
        max_fitness: int = N*(N-1) // 2
        for i in range(N):
            fixed_queen_pos: Tuple[int, int] = (self.queen_positions[i], i) # (row, column)
            for j in range(i+1, N):
                curr_queen_pos: Tuple[int, int] = (self.queen_positions[j], j) # (row, column)
                
                diagonal_collision_row: int = abs(fixed_queen_pos[0]-curr_queen_pos[0])
                diagonal_collision_column: int = abs(fixed_queen_pos[1]-curr_queen_pos[1])

                # If a queen pair is not in the same row or diagonal => they are not attacking each other.
                if fixed_queen_pos[0] != curr_queen_pos[0] and diagonal_collision_row != diagonal_collision_column:
                    h += 1
        return max_fitness - h  
    
    def update_total_cost(self) -> None:
        """Function to update total cost of a state."""

        self.g += 1
        self.total_cost = self.heuristic + self.g

    def __lt__(self, other):
        return self.total_cost < other.total_cost
    
    def __eq__(self, other):
        return self.queen_positions == other.queen_positions

    def __str__(self) -> str:
        return f'Queen Positions: {self.queen_positions} | h(n): {self.heuristic} | g(n): {self.g} | F(N): {self.total_cost}'


class AStarAlgorithm:
    """Class that contains all the inner implementations (steps) involved in the A* Algorithm."""

    @staticmethod
    def generate_random_state(N: int):
        """Function that generates a random initial state."""

        return State(N=N)

    @staticmethod
    def generate_child_states(state: State, states: List[State], visited_states: List[State]) -> List[State]:
        """Function that generates children of a state.
        It moves a queen's position up or down the column by 1 move and stores it as one of its the many children states.
        """
    
        queen_pos: List[int] = state.queen_positions
        size: int = len(queen_pos)
        children = []
        for i, pos in enumerate(queen_pos):
            if pos - 1 > 0: # Lower limit of the board.
                child_state_1 = copy.deepcopy(state)
                child_state_1.queen_positions[i] = pos - 1
                child_state_1.heuristic = child_state_1.calculate_heuristic()
                child_state_1.update_total_cost()
                if child_state_1 not in states and child_state_1 not in visited_states:
                    children.append(child_state_1)
            if pos + 1 <= size: # Upper limit of the board.
                child_state_2 = copy.deepcopy(state)
                child_state_2.queen_positions[i] = pos + 1
                child_state_2.heuristic = child_state_2.calculate_heuristic()
                child_state_2.update_total_cost() 
                if child_state_2 not in states and child_state_2 not in visited_states:
                    children.append(child_state_2)
        return children
    
    @staticmethod
    def is_goal_reached(state: State):
        """Function to check if the goal is reached or not."""

        return state.heuristic == 0


class RunAStarAlgorithm:
    """Class for running the algorithm."""

    @staticmethod
    def run_a_star(N_queens: Optional[int] = 4):
        """Function that runs the algorithm in a sequential manner.
        1. Create random state and add it to the heap.
        2. Pop the heap and check if it is the goal.
        3. If not a goal, then generate child state and add all children to heap. If yes, stop.
        4. Repeat steps 2-3 until goal is reached.
        """

        states: List[State] = []
        visited_states: List[State] = []
        goal_reached: bool = False
        steps: int = 1
        
        start_time: float = time.time()

        initial_state: State = AStarAlgorithm.generate_random_state(N=N_queens)
        states.append(initial_state)
        heapq.heapify(states)

        while len(states) != 0 and not goal_reached:
            curr_state: State = heapq.heappop(states)
            print(f'Current state: {curr_state}')
            RunAStarAlgorithm._save_state_pic(curr_state.queen_positions, steps)
            visited_states.append(curr_state)

            if AStarAlgorithm.is_goal_reached(curr_state):
                print(f'GOAL STATE => {curr_state}')
                goal_reached = True
                break
            else:
                child_states: List[State] = AStarAlgorithm.generate_child_states(curr_state, states, visited_states)
                states.extend(child_states)
                heapq.heapify(states)
            steps += 1
        else:
            print('SOLUTION NOT FOUND!')

        finish_time: float = time.time()
        time_taken: float = finish_time - start_time

        # In this case steps = search_space_depth. Hence, subtracting 1.
        if not goal_reached:
            steps -= 1
        
        print('###########################################')
        print(f'Total Steps = {steps}')
        print(f'Solution = {curr_state}')
        print(f'Time taken = {time_taken} seconds')
        
        return [steps, time_taken, goal_reached] 
        
    @staticmethod
    def _save_state_pic(positions: List[int], i: int):
        """Function to plot the chess board along with the position of the queens."""
        
        N: int = len(positions)
        board = np.zeros((N,N,3))
        board += 0.5 # "Black" color. Can also be a sequence of r,g,b with values 0-1.
        board[::2, ::2] = 1 # "White" color
        board[1::2, 1::2] = 1 # "White" color

        fig, ax = plt.subplots()
        ax.imshow(board, interpolation='nearest')

        for y, x in enumerate(positions):
            ax.text(y,x-1, u'\u2655', size=20, ha='center', va='center', backgroundcolor='cornflowerblue')

        ax.set(xticks=[], yticks=[])
        ax.axis('image')
        plt.savefig(f'states_images/state{i}')
