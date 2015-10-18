#ifndef MNN_DATA_STRUCTURE_H
#define MNN_DATA_STRUCTURE_H

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#include "mnn_utils.h"

typedef struct mnn_vector_
{
    size_t length;
    double* values;
} mnn_vector;

typedef struct mnn_matrix_
{
    size_t rows;
    size_t cols;
    double** values;
} mnn_matrix;

typedef struct mnn_network_
{
    size_t size_input_neurons;
    mnn_vector sizes_hidden_neurons; /* Possibility to use one or more hidden layer/s. Values will be casted to int */
    size_t size_output_neurons;
} mnn_network;

typedef struct mnn_network_options_
{
    char* output_file;
    char* load_file;
    char* image_file;
    char* js_file;
    int training;
} mnn_network_options;

typedef struct mnn_training_parameters_
{
    size_t size_epochs;
    size_t size_mini_batches;
    double learning_rate;
} mnn_training_parameters;

typedef struct mnn_network_configuration_
{
    mnn_network* net;
    mnn_training_parameters* training_params;
} mnn_network_configuration;

typedef struct mnn_backpropagation_data_
{
    mnn_vector* biases_;
    mnn_matrix* weights_;
} mnn_backpropagation_data;

mnn_vector allocate_vector (size_t length);

mnn_matrix allocate_matrix (size_t rows, size_t cols);

mnn_matrix matrix_dot_matrix (mnn_matrix m_1, mnn_matrix m_2);

mnn_matrix matrix_sigmoid (mnn_matrix m);

mnn_matrix matrix_sigmoid_prime (mnn_matrix m);

mnn_matrix matrix_transpose (mnn_matrix m);

void print_vector (mnn_vector v);

void print_matrix (mnn_matrix m);

void free_vector (mnn_vector v);

void free_matrix (mnn_matrix m);

#endif
