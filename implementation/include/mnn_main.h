#ifndef MNN_MAIN_H
#define MNN_MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mnn_utils.h"
#include "mnn_io.h"
#include "mnn_data_structure.h"
#include "mnn_parser_config_file.h"
#include "mnn_network.h"

void usage (char* argv_0);

int contains_option (char** options, int n_options, char* option);

void show_image (char* image);

#endif
