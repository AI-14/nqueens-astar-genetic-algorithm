from typing import Optional, List, Tuple
import random, copy, time
import matplotlib.pyplot as plt
import numpy as np


class State:

    def __init__(self, N: Optional[int] = 4) -> None:
        self.queen_positions: List[int] = [random.randint(1,N) for i in range(N)]
        self.fitness: int  = self.calculate_fitness()

    def calculate_fitness(self) -> int:
        fitness: int = 0
        N: int = len(self.queen_positions)
        for i in range(N):
            fixed_queen_pos: Tuple[int, int] = (self.queen_positions[i], i) # (row, column)
            for j in range(i+1, N):
                curr_queen_pos: Tuple[int, int] = (self.queen_positions[j], j) # (row, column)
                
                diagonal_collision_row = abs(fixed_queen_pos[0]-curr_queen_pos[0])
                diagonal_collision_column = abs(fixed_queen_pos[1]-curr_queen_pos[1])

                # If a queen pair is not in the same row or diagonal => they are not attacking each other.
                if fixed_queen_pos[0] != curr_queen_pos[0] and diagonal_collision_row != diagonal_collision_column:
                    fitness += 1
        return fitness


    def __str__(self) -> str:
        return f'Queen Positions: {self.queen_positions} => Fitness: {self.fitness}'


class Population:

    def __init__(self, pop_size: Optional[int] = 4, N: Optional[int] = 4) -> None:
        self.list_of_states = [State(N) for i in range(pop_size)]
    
    def __str__(self) -> str:
        for state in self.list_of_states:
            print(state)
        return ''
    
class GeneticAlgorithm:
    
    @staticmethod
    def get_mating_parents(population: Population, parent_selection_algo='RWS'):
        parent1: State = None
        parent2: State = None

        if parent_selection_algo == 'RWS':
            parent1 = GeneticAlgorithm._roulette_wheel_selection(population)
            parent2 = GeneticAlgorithm._roulette_wheel_selection(population)
        elif parent_selection_algo == 'RS':
            parent1, parent2 = GeneticAlgorithm._rank_selection(population)

        return parent1, parent2
    
    @staticmethod
    def _rank_selection(population: Population) -> State:
        population.list_of_states.sort(key=lambda state: state.fitness)
        return population.list_of_states[-1], population.list_of_states[-2]
        

    @staticmethod
    def _roulette_wheel_selection(population: Population) -> State:
        states = population.list_of_states
        sum_fitness = 0
        s = 0
        for state in states:
            sum_fitness += state.fitness
        roulette_pointer = random.randint(0, sum_fitness)
        for state in states:
            s += state.fitness
            if s >= roulette_pointer:
                return state
    
    @staticmethod
    def crossover(parent1: State, parent2: State):
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
    def mutate(child: State) -> State:
        size = len(child.queen_positions)
        random_index = random.randint(0,size-1)
        random_position = random.randint(1, size)
        child.queen_positions[random_index] = random_position
        child.fitness = child.calculate_fitness()
        return child



class RunGeneticAlgorithm():

    @staticmethod
    def run_ga(N_queens: int = 4, 
            init_pop_size: int = 8, 
            max_gen: int = 400, 
            mutation_prob: float = 0.8,
            parent_selection_algo: str = 'RWS'):

        gen = 1
        solution_found = False
        mutation_prob = mutation_prob
        solution_state = None

        start_time = time.time() 
        print('---------------------------------------')
        print('Initial Population')
        population = Population(pop_size=init_pop_size, N=N_queens)
        max_fitness = N_queens*(N_queens-1)//2
        population.list_of_states.sort(key=lambda state: state.fitness)
        print(population)
        print(f'Max Fitness: {max_fitness}')
        print('----------------------------------------')

        while gen < max_gen:
            print(f'Generation {gen}:')
            parent1, parent2 = GeneticAlgorithm.get_mating_parents(population, parent_selection_algo=parent_selection_algo)
            child1, child2 = GeneticAlgorithm.crossover(parent1, parent2)

            if random.random() < mutation_prob:
                child1 = GeneticAlgorithm.mutate(child1)
                child2 = GeneticAlgorithm.mutate(child2)
            
            population.list_of_states.extend([child1, child2])
            population.list_of_states.sort(key=lambda state: state.fitness)
            best_state_so_far = population.list_of_states[-1]

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

        stop_time = time.time()
        time_taken = stop_time - start_time
        print('###########################################')
        print(f'Total Steps = {gen}')
        print(f'Solution = {solution_state}')
        print(f'Time taken = {time_taken} seconds')
        return [gen, time_taken, solution_found]

    @staticmethod
    def _save_state_pic(positions,i):
        N = len(positions)
        board = np.zeros((N,N,3))
        board += 0.5 # "Black" color. Can also be a sequence of r,g,b with values 0-1.
        board[::2, ::2] = 1 # "White" color
        board[1::2, 1::2] = 1 # "White" color

        fig, ax = plt.subplots()
        ax.imshow(board, interpolation='nearest')

        for y, x in enumerate(positions):
        # Use "family='font name'" to change the font
            ax.text(y,x-1, u'\u2655', size=15, ha='center', va='center')

        ax.set(xticks=[], yticks=[])
        ax.axis('image')
        plt.savefig(f'states_images/state{i}')
