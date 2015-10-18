#ifndef MNN_IO_H
#define MNN_IO_H

#include <stdio.h>
#include <string.h>

#define ANSI_CYAN    "\x1b[36m"
#define ANSI_GREEN   "\x1b[32m"
#define ANSI_RED     "\x1b[31m"
#define ANSI_YELLOW  "\x1b[33m"
#define ANSI_RESET   "\x1b[0m"

void print_error (char* message);

void print_warning (char* message);

void print_info (char* message);

void print_success (char* message);

#endif
