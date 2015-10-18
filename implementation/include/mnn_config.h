#ifndef MNN_CONFIG_H
#define MNN_CONFIG_H

/* Input layer size */
#define N_INPUT_LAYER 784

/* Output layer size */
#define N_OUTPUT_LAYER 10

/* Number of training images/labels according to MNIST database */
#define N_TRAINING_IMAGES 100

/* Number of test images/labels according to MNIST database */
#define N_TEST_IMAGES 100

/* Number of rows of the images according to MNIST database */
#define N_ROWS_IMAGE 28

/* Number of columns of the images according to MNIST database */
#define N_COLS_IMAGE 28

/* Number of entries for each image */
#define N_ENTRIES_IMAGE N_ROWS_IMAGE * N_COLS_IMAGE

/* Path training data */
#define PATH_TRAINING_DATA "../data/mnist-database-training/train-images-idx3-ubyte"

/* Path training labels */
#define PATH_TRAINING_LABELS "../data/mnist-database-training/train-labels-idx1-ubyte"

/* Path test data */
#define PATH_TEST_DATA "../data/mnist-database-test/t10k-images-idx3-ubyte"

/* Path test labels */
#define PATH_TEST_LABELS "../data/mnist-database-test/t10k-labels-idx1-ubyte"

/* Default path config file. User can specify own config file from the command line */
#define PATH_DEFAULT_CONFIG_FILE "default_configuration.conf"

#endif
