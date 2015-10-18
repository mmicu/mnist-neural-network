#include "mnn_io.h"

void
print_error (char* message)
{
    if (message && strlen (message) > 0)
        printf (ANSI_RED "   [Error]  " ANSI_RESET "%s", message);
}

void
print_warning (char* message)
{
    if (message && strlen (message) > 0)
        printf (ANSI_YELLOW "   [Warning]  " ANSI_RESET "%s", message);
}

void
print_info (char* message)
{
    if (message && strlen (message) > 0)
        printf (ANSI_CYAN "   [Info]  " ANSI_RESET "%s", message);
}

void
print_success (char* message)
{
    if (message && strlen (message) > 0)
        printf (ANSI_GREEN "   [Success]  " ANSI_RESET "%s", message);
}
