from typing import Optional, List, Tuple
import random, copy, time
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'figure.max_open_warning': 0})


class State:
    """Class that holds the state of a board. 
    It holds queen's positions on the board as well as its fitness.
    """

    def __init__(self, N: Optional[int] = 4) -> None:
        self.queen_positions: List[int] = [random.randint(1,N) for i in range(N)]
        self.fitness: int  = self.calculate_fitness()

    def calculate_fitness(self) -> int:
        """Function to calculate the fitness of the state.
        Fitness is calculated as => Number of pairs of queens NOT attacking each other. 
        """

        fitness: int = 0
        N: int = len(self.queen_positions)
        for i in range(N):
            fixed_queen_pos: Tuple[int, int] = (self.queen_positions[i], i) # (row, column)
            for j in range(i+1, N):
                curr_queen_pos: Tuple[int, int] = (self.queen_positions[j], j) # (row, column)
                
                diagonal_collision_row: int = abs(fixed_queen_pos[0]-curr_queen_pos[0])
                diagonal_collision_column: int = abs(fixed_queen_pos[1]-curr_queen_pos[1])

                # If a queen pair is not in the same row or diagonal => they are not attacking each other.
                if fixed_queen_pos[0] != curr_queen_pos[0] and diagonal_collision_row != diagonal_collision_column:
                    fitness += 1
        return fitness

    def __str__(self) -> str:
        return f'Queen Positions: {self.queen_positions} => Fitness: {self.fitness}'


class Population:
    """Class that holds the list of states. Hence, this is our population at a given time."""

    def __init__(self, pop_size: Optional[int] = 4, N: Optional[int] = 4) -> None:
        self.list_of_states: List[State] = [State(N) for i in range(pop_size)]
    
    def __str__(self) -> str:
        for state in self.list_of_states:
            print(state)
        return ''
    

class GeneticAlgorithm:
    """Class that contains all the inner implementations (steps) involved in the Genetic Algorithm."""
    
    @staticmethod
    def get_mating_parents(population: Population, parent_selection_algo: Optional[str] ='RWS') -> Tuple[State, State]:
        """Function to get the best matching parents from the population.
        We included two ways for finding the best parents:
        1. Rank selection method
        2. Roulette wheel selection method
        """

        parent1: State = None
        parent2: State = None

        if parent_selection_algo == 'RWS':
            parent1 = GeneticAlgorithm._roulette_wheel_selection(population)
            parent2 = GeneticAlgorithm._roulette_wheel_selection(population)
        elif parent_selection_algo == 'RS':
            parent1, parent2 = GeneticAlgorithm._rank_selection(population)

        return parent1, parent2
    
    @staticmethod
    def _rank_selection(population: Population) -> Tuple[State, State]:
        """Function to perform rank selection method on a population."""

        population.list_of_states.sort(key=lambda state: state.fitness)
        return population.list_of_states[-1], population.list_of_states[-2]
        
    @staticmethod
    def _roulette_wheel_selection(population: Population) -> State:
        """Function to perform roulette wheel selection on a population."""

        states: List[State] = population.list_of_states
        sum_fitness: int = 0
        s: int = 0
        for state in states:
            sum_fitness += state.fitness
        roulette_pointer: int = random.randint(0, sum_fitness)
        for state in states:
            s += state.fitness
            if s >= roulette_pointer:
                return state
    
    @staticmethod
    def crossover(parent1: State, parent2: State, crossover_method: str) -> Tuple[State, State]:
        """Function to perform crossover between two parents.
        Two crossover methods are used:
        1. Single point 
        2. Two point
        """

        if crossover_method == 'SP':
            return GeneticAlgorithm._single_point_crossover(parent1, parent2)
        elif crossover_method == 'TP':
            return GeneticAlgorithm._two_point_crossover(parent1, parent2)

    @staticmethod
    def _single_point_crossover(parent1: State, parent2: State) -> Tuple[State, State]:
        """Function to perform single point crossover."""

        size: int = len(parent1.queen_positions)
        child1: State = copy.deepcopy(parent1)
        child2: State = copy.deepcopy(parent2)

        split_point: int = random.randint(1, size-1) 

        child1.queen_positions = parent1.queen_positions[0:split_point] + parent2.queen_positions[split_point:]
        child2.queen_positions = parent2.queen_positions[0:split_point] + parent1.queen_positions[split_point:]

        child1.fitness = child1.calculate_fitness()
        child2.fitness = child2.calculate_fitness()
        return child1, child2

    @staticmethod
    def _two_point_crossover(parent1: State, parent2: State) -> Tuple[State, State]:
        """Function to perform two point crossover."""

        size: int = len(parent1.queen_positions)
        child1: State = copy.deepcopy(parent1)
        child2: State = copy.deepcopy(parent2)

        point1: int = random.randint(1, size//2)
        point2: int = random.randint(point1+1, size-1)
        
        child1.queen_positions = parent1.queen_positions[0:point1] + parent2.queen_positions[point1:point2] + parent1.queen_positions[point2:]
        child2.queen_positions = parent2.queen_positions[0:point1] + parent1.queen_positions[point1:point2] + parent2.queen_positions[point2:]

        child1.fitness = child1.calculate_fitness()
        child2.fitness = child2.calculate_fitness()
        return child1, child2

    @staticmethod
    def mutate(child: State) -> State:
        """Function that mutates the given child."""

        size: int = len(child.queen_positions)
        random_index: int = random.randint(0,size-1)
        random_position: int = random.randint(1, size)
        child.queen_positions[random_index] = random_position
        child.fitness = child.calculate_fitness()
        return child


class RunGeneticAlgorithm:
    """Class for running the algorithm."""

    @staticmethod
    def run_ga(N_queens: Optional[int] = 4, 
              init_pop_size: Optional[int] = 10, 
              max_gen: Optional[int] = 400, 
              mutation_prob: Optional[float] = 0.8,
              crossover_method: str = 'SP', # SP or TP
              crossover_rate: float = 0.2,
              parent_selection_algo: Optional[str] = 'RWS', # RWS or RS
              elitism: str = 'N' # N or Y
              ) -> List:
        """Function that runs the algorithm in a sequential manner.
        Steps taken are as follows:
        1. Generate random population.
        2. Choose parents from the population.
        3. Crossover between the parents to produce offsprings.
        4. Mutate the children (if necessary).
        5. Check if any state among the population has reached the goal.
        6. Repeat steps 2-5 until solution is found.

        There are other criterions such as crossover_rate and mutation_rate, elitisism that
        affects the time taken to find the solution. So we can use any combination of these
        and play with the solution space.
        """

        gen: int = 1
        solution_found: bool = False
        mutation_prob: float = mutation_prob
        solution_state: Optional[State] = None
        max_fitness = N_queens*(N_queens-1)//2

        start_time: float = time.time() 

        print('---------------------------------------')
        print('Initial Population')
        population = Population(pop_size=init_pop_size, N=N_queens)
        print(population)
        print(f'Max Fitness: {max_fitness}')
        print('----------------------------------------')

        crossover_rate_population = int(crossover_rate * len(population.list_of_states))

        while gen < max_gen:
            print(f'Generation {gen}:')
            
            if elitism == 'N':
                # Selecting according to ratio of how many couples will be picked for mating (i.e. crossover_rate).
                population.list_of_states.sort(key=lambda state: state.fitness)
                population_size: int = len(population.list_of_states)
                population.list_of_states = population.list_of_states[population_size-crossover_rate_population:]
            elif elitism == 'Y':
                population.list_of_states.sort(key=lambda state: state.fitness)
                population.list_of_states = population.list_of_states[-1:-3:-1] # Taking best two individuals.
                
            parent1, parent2 = GeneticAlgorithm.get_mating_parents(population, parent_selection_algo=parent_selection_algo)
            child1, child2 = GeneticAlgorithm.crossover(parent1, parent2, crossover_method)
            
            if random.random() < mutation_prob:
                child1 = GeneticAlgorithm.mutate(child1)
                child2 = GeneticAlgorithm.mutate(child2)
            
            population.list_of_states.extend([child1, child2])
            population.list_of_states.sort(key=lambda state: state.fitness)
            best_state_so_far: State = population.list_of_states[-1]

            print(f'Best State so far: {best_state_so_far}')
            RunGeneticAlgorithm._save_state_pic(best_state_so_far.queen_positions, gen)

            if best_state_so_far.fitness == max_fitness:
                print(f'FOUND BEST => {best_state_so_far}')
                solution_state = best_state_so_far
                solution_found = True
                break
            print('--------------------------------------')
            gen += 1

        if not solution_found:
            print('MAX ITERATION ENDED! SOLUTION NOT FOUND')

        stop_time: float = time.time()
        time_taken: float = stop_time - start_time

        # In this case gen = max_gen. Hence, subtracting 1.
        if not solution_found:
            gen -= 1

        print('###########################################')
        print(f'Total Steps = {gen}')
        print(f'Solution = {solution_state}')
        print(f'Time taken = {time_taken} seconds')

        return [gen, time_taken, solution_found]

    @staticmethod
    def _save_state_pic(positions: List[int], i: int) -> None:
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
