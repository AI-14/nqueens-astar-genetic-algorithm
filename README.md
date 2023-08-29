# N-Queens Puzzle Solver (A* And GeneticAlgorithm) 

   Course Mini Project: ICS 381/202 - Principles of Artificial Intelligence

## Table of Contents
1. [Description](#description)
2. [Demo](#demo)
3. [Installation and Usage](#installation-usage)

## Description <a name="description"><a/>
   Design a GUI for solving N-Queens problem using A* and Genetic Algorihtms. For more details, see *202-ICS381-PA01.pdf*.
   
## Demo <a name="demo"><a/>

   GUI demo:

   ![](readme_res//demo.gif)

   **Note:** We can use the algorithms as a text-based instead of GUI. Under ````algorithms/```` folder, run the corresponding algorithms by calling static functions
   ````RunAStarAlgorithm.run_a_star()```` for A* and ````RunGeneticAlgorithm.run_ga()```` for Genetic algorithm and pass their parameters. One additional task has to be
   done if you are using text-based UI i.e. create a folder called ````states_images```` (this folder stores the images of states).

   Screenshots of A* and Genetic Algorithm (Text-based UI):
   
   <img src="readme_res//TextBasedUI-AStar.png" width="500"/>
   
   <img src="readme_res//TextBasedUI-GeneticAlgorithm.png" width="500"/>
   
   >*I recommend using text-based UI for an in-depth comparative study.*
   
## Installation and Usage <a name="installation-usage"><a/>
- Requirements
  - `python >= 3.6`
- `git clone https://github.com/AI-14/nqueens-astar-genetic-algorithm.git` - clones the repository
- `cd nqueens-astar-genetic-algorithm`
- `py -m venv yourVenvName` - creates a virtual environment
- `pip install -r requirements.txt` - installs all modules
- `py PA1.py` - runs the main file
