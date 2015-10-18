#ifndef MNN_NETWORK_H
#define MNN_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mnn_config.h"
#include "mnn_utils.h"
#include "mnn_file_utils.h"
#include "mnn_data_structure.h"
#include "mnn_io.h"
#include "mnn_network_utils.h"

void execute_network (mnn_network_configuration* net_configuration, mnn_network_options* net_options);

void update_mini_batch (mnn_network_configuration* net_configuration, mnn_matrix* training_data, int* training_labels,
                        mnn_matrix* weights, mnn_vector* biases, int index);

mnn_backpropagation_data backpropagation (mnn_network* net, mnn_matrix x, int y, mnn_matrix* weights, mnn_vector* biases);

int forward (mnn_network* net, mnn_matrix x, mnn_matrix* weights, mnn_vector* biases);

int evaluate (mnn_matrix m);

int test_network (mnn_network* net, mnn_matrix* test_data, int* test_labels, mnn_matrix* weights, mnn_vector* biases);

int predict_output_image (mnn_network* net, char* image_file, mnn_matrix* weights, mnn_vector* biases);

void update_best_parameters (size_t length_layers, mnn_matrix* weights, mnn_vector* biases, mnn_matrix* best_weights, mnn_vector* best_biases);

#endif
