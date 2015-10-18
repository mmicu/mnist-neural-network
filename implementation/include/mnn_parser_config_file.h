#ifndef MNN_PARSER_CONFIG_FILE_H
#define MNN_PARSER_CONFIG_FILE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mnn_config.h"
#include "mnn_utils.h"
#include "mnn_file_utils.h"
#include "mnn_data_structure.h"
#include "mnn_io.h"

#define N_PARAMETERS 4
/*#define DEBUG_PARSER*/

typedef struct mnn_token_
{
    char* value;
    size_t size;
} mnn_token;

/* Global variable used to track current line of the file (initialized it in "parse_config_file" function) */
int line;

int parse_config_file (mnn_network_configuration* net_configuration, const char* path_config_file);

mnn_token get_token (FILE* fp, size_t length);

void parse_open_bracket (FILE* fp);

void parse_parameters (FILE* fp, char** params, mnn_network_configuration* net_configuration);

void handle_parameter (FILE* fp, char* param_value, mnn_network_configuration* net_configuration);

void handle_hidden_layers_parameter (FILE* fp, mnn_network_configuration* net_configuration);

void handle_learning_rate_parameter (FILE* fp, mnn_network_configuration* net_configuration);

void handle_epochs_parameter (FILE* fp, mnn_network_configuration* net_configuration);

void handle_mini_batches_parameter (FILE* fp, mnn_network_configuration* net_configuration);

void parse_close_bracket (FILE* fp);

int is_number (char character);

int is_valid_alpha_character (char character);

int is_skip_character (char character);

char** setup_parameters (void);

#endif
