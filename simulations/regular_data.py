import numpy as np
import sys

sys.path.append('../')
import library


def main():

    #errors are saved to a .txt file
    library.error_handling(sys.argv[0][:-3])

    #save command line arguments to variables
    args = sys.argv
    tm_type = args[1]
    n_samples = int(args[2])
    noise = args[3:]

    #assign data path
    data_path = library.data_pathways_make_directories_regular(tm_type, noise)

    #import range of theta from the library
    theta_min = library.theta_min
    theta_max = library.theta_max

    #create a toy model object with n_samples within our theta range
    tm = library.ToyModels(n_samples, theta_min, theta_max, seed=42)

    if tm_type == 'cubic':
        x, y = tm.cubic(noise)
    elif tm_type == 'singlequbit':
        x, y = tm.single_qubit(noise)
    else:
        raise Exception('Unrecognised toy model.')

    #create a noise string
    noise_str = library.noise(noise)

    #save files in destinations
    np.savetxt(f'{data_path}/x-{n_samples}{noise_str}.csv', x,
               delimiter=',')
    np.savetxt(f'{data_path}/y-{n_samples}{noise_str}.csv', y,
               delimiter=',')


if __name__ == '__main__':
    main()
