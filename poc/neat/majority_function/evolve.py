'''
Created on 12 juil. 2019

@author: nboutin
'''

import os
import neat
from math import pow

inputs = ((False,False,False),(False,False,True),(False,True,False),(False,True,True),
                  (True,False,False),(True,False,True),(True,True,False),(True,True,True))

def majorityFunction(a,b,c):
    '''M(A,B,C)=AB+AC+BC'''
    return a and b or a and c or b and c

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = float(len(inputs))
        net = neat.nn.FeedForwardNetwork.create(genome, config)
         
        for i in inputs:
            i_float = [float(x) for x in i] # from bool to float
            r = net.activate(i_float)
            genome.fitness -= pow(r[0] - majorityFunction(*i), 2) #*i, expand tuple
            
def run(config_file):

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)
    
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     p.add_reporter(stats)
#     p.add_reporter(neat.Checkpointer(5))
    
    # Run until a solution is found.
    winner = p.run(eval_genomes)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))
    
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for i in inputs:
        output = winner_net.activate([float(x) for x in i])
        print("  input {!r}, expected output {!r}, got {!r}".format(i, majorityFunction(*i), output))

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'majority_config')
    run(config_path)