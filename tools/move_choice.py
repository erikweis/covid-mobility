import numpy as np
import matplotlib.pyplot as plt


def get_relative_move_coordinates(num_choices, move_size_exponent,max_attempts=500):

    choices = []

    move_sizes = np.random.zipf(move_size_exponent,size=num_choices)

    for r in move_sizes:

        for _ in range(5):

            theta = 2*np.pi*np.random.random()
            x = int(round(r*np.cos(theta)))
            y = int(round(r*np.sin(theta)))

            if (x,y) in choices:
                continue
            else:
                choices.append((x,y))
                break

    #ensure origin is there
    if (0,0) not in choices:
        choices.append((0,0))

    return choices

def get_move_choice_params(income):

    max_income = 200000
    rescaled_income = income/max_income #rescaled income is number between 0 and 1

    ### move size exponent (for zipf distribution)
    min_move_size_exponent = 1.2
    max_move_size_exponent = 4
    diff_expo = max_move_size_exponent- min_move_size_exponent
    
    move_size_exponent = max_move_size_exponent-rescaled_income*diff_expo

    ### num choices
    min_num_choices = 3
    max_num_choices = 25
    diff_numchoices = max_num_choices-min_num_choices

    num_choices = int(min_num_choices + rescaled_income*diff_numchoices)

    return num_choices,move_size_exponent


def plot_example_choices():

    for move_size_exponent in np.linspace(2,7,20):
        N = 20
        x0,y0 = 10,10

        move_options = np.zeros((N,N))

        num_choices = 10
        choices = get_relative_move_coordinates(num_choices,move_size_exponent)

        # get possible locations to move
        for x,y in choices:
            move_options[(x0+x)%N][(y0+y)%N]=1

        plt.title(move_size_exponent)
        plt.imshow(move_options)
        plt.show()

def plot_example_move_distirbution_by_income():

    rows,cols = 6,6
    N = 40
    x0,y0=10,10

    incomes = np.linspace(200,200000,rows)
    
    fig, axes = plt.subplots(rows,cols,figsize=(9,9))

    for i in range(rows):
        for j in range(cols):
            
            move_options = np.zeros((N,N))

            choices = get_relative_move_coordinates(*get_move_choice_params(incomes[i]))
            for x,y in choices:
                move_options[(x0+x)%N][(y0+y)%N]=1

            axes[i][j].imshow(move_options)
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

            # if j == 0:
            #     axes[i][j].set_ylabel(int(incomes[i]))

    plt.suptitle('Example Move Possiblities by Income')
    plt.tight_layout()
    plt.savefig('move_possibilities_by_income.png')

if __name__ == '__main__':
    
    plot_example_move_distirbution_by_income()