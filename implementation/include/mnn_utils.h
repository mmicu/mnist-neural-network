#ifndef MNN_UTILS_H
#define MNN_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mnn_io.h"

void __assert (char* message, int condition);

void __exit (char* message);

int reverse_int (int n);

double drand (void);

double random_normal_distribution (void);

int random_integer (int min, int max);

double sigmoid_to_number (double number);

double sigmoid_prime_to_number (double number);

#endif
