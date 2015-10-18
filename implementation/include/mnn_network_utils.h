#ifndef MNN_NETWORK_UTILS_H
#define MNN_NETWORK_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mnn_config.h"
#include "mnn_utils.h"
#include "mnn_file_utils.h"
#include "mnn_data_structure.h"
#include "mnn_io.h"

mnn_vector* load_biases_in_the_file (mnn_network* net, int train, const char* file);

mnn_vector* load_biases_randomly (mnn_network* net);

mnn_vector* get_pointer_biases (mnn_network* net);

mnn_matrix* load_weights_in_the_file (mnn_network* net, int train, const char* file);

mnn_matrix* load_weights_randomly (mnn_network* net);

mnn_matrix* get_pointer_weights (mnn_network* net);

FILE* check_correct_format_of_the_file (mnn_network* net, int train, const char* file);

int save_weights_and_biases (mnn_network* net, mnn_matrix* weights, mnn_vector* biases, const char* file);

int save_weights_and_biases_into_js_file (mnn_network* net, mnn_matrix* weights, mnn_vector* biases, const char* file);

void print_info_network (mnn_network_configuration* net_configuration, mnn_network_options* net_options);

#endif
