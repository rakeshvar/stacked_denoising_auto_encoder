import os
import sys
import ast
from mnist import train_x_mnist

import numpy as np
import theano as th
import dae
from stacked import Stack_DAE

sdae_args = {
        "dataset"   :None,
        "noise"     :.1,
        "n_nodes"   :(180, 42, 10),
        "learning_rate": .1,
        "n_epochs"  :100,
        "output_folder": 'plots',
        "lambda1"   :(.4, .05, .05),
}


def solo_to_tuple(val, n=3):
    if type(val) in (list, tuple):
        return val
    else:
        return (val,)*n


def main():
    if len(sys.argv) < 2:
        print("Usage:\n"
              "{0} arg1=value1 arg2=value2 ...\n"
              "Example:\n"
              "{0} n_nodes=[90,30,10] n_epochs=10 noise=.2\n"
              "(Do not give spaces or '(', ')' parentheses)\n"
              "Using defaults:\n".format(sys.argv[0]))

    for arg in sys.argv[1:]:
        k, v = arg.split('=', 1)
        sdae_args[k] = ast.literal_eval(v)

    for k in ('learning_rate', 'n_epochs', 'n_nodes', 'noise', 'lambda1'):
        sdae_args[k] = solo_to_tuple(sdae_args[k])

    print("Stacked DAE arguments: ")
    for k in sorted(sdae_args.keys()):
        print("\t{:15}: {}".format(k, sdae_args[k]))

    print("SEED: ", dae.seed)
    test_stacked_dae(**sdae_args)


def test_stacked_dae(
        dataset,
        noise,
        n_nodes,
        learning_rate,
        n_epochs,
        output_folder,
        lambda1
):
    # Cosmetics
    print("Reading Data", dataset)
    train_set_x = train_x_mnist(dataset)
    print(train_set_x.get().shape)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    file_name = '{}//{}_{}_{}_C{}_N{}_L{}_I{{}}.png' \
                ''.format(output_folder,
                        dataset if dataset else 'mnist',
                        th.config.device,
                        dae.seed, n_nodes,
                        int(100 * noise[0]),
                        int(100 * lambda1[0]))

    np.set_printoptions(precision=3)

    stk = Stack_DAE(train_set_x, n_nodes, lambda1, learning_rate, noise,
                    n_epochs)
    stk.do_all(file_name)
    stk.save_wbs_raster(file_name)


if __name__ == '__main__':
    main()