#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <stdio.h>

#include "mnn_config.h"
#include "mnn_utils.h"
#include "mnn_data_structure.h"

int* load_training_labels (void);

int* load_test_labels (void);

int* load_labels (const char* file);

mnn_matrix* load_training_data (void);

mnn_matrix* load_test_data (void);

mnn_matrix* load_data (const char* file);

int file_exists (const char* file);

#endif
