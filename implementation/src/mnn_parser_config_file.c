#include "mnn_parser_config_file.h"

int
parse_config_file (mnn_network_configuration* net_configuration, const char* path_config_file)
{
    FILE* fp;
    char** params;

    /* Global variable declared in "mnn_parser_config_file.h" */
    line = 1;

    /* If specified, we try to open config file passed from the command line else the default config file */
    if (!(fp = fopen (((path_config_file) ? path_config_file : PATH_DEFAULT_CONFIG_FILE), "r"))) {
        print_error ("Impossible to open config file.\n");

        return 0;
    }

    /* Load avaiable parameters in the config file */
    if (!(params = (setup_parameters ()))) {
        print_error ("'setup_parameters' function ('mnn_parser_config_file.c'), impossible to allocate avaiable_params.\n");
        fclose (fp);

        return 0;
    }

    /* Setting training_params pointer */
    net_configuration->training_params = (mnn_training_parameters *) malloc (sizeof (mnn_training_parameters));
    if (!net_configuration->training_params) {
        print_error ("'parse_config_file' function ('mnn_parser_config_file.c'), impossible to allocate net_configuration->training_params.\n");
        fclose (fp); free (params);

        return 0;
    }

    /* Config file must start with '{' */
    parse_open_bracket (fp);

    /* Parse all config parameters and update "net_configuration" pointer */
    parse_parameters (fp, params, net_configuration);

    /* Config file must end with '};' */
    parse_close_bracket (fp);

    fclose (fp);
    free (params);

    return 1;
}

mnn_token
get_token (FILE* fp, size_t length)
{
    __assert ("'get_token' function ('mnn_parser_config_file.c'), length <= 0", length > 0);

    mnn_token token;
    char current_character, error_message[1024];
    size_t current_length;

    token.size  = length;
    token.value = (char *) malloc (sizeof (char) * (token.size + 1));
    if (!token.value) {
        sprintf (error_message, "'get_token' function ('mnn_parser_config_file.c'), impossible to allocate \"token.value\".\n");
        goto handle_error;
    }

    current_length = 0;
    while ((current_character = fgetc (fp)) != EOF) {
        if (is_skip_character (current_character)) {
            /* Handle '#' */
            if (current_character == '#') {
                while ((current_character = fgetc (fp)) != EOF) {
                    if (current_character == '\n') {
                        line++;

                        break;
                    }
                }
                if (current_character == '\n')
                    continue;
                /* End of file earlier than expected */
                sprintf (error_message, "Parsing config file. End of file earlier than expected at line %d.\n", line);
                goto handle_error;
            }
            /* Handle '\n' */
            line += (current_character == '\n') ? 1 : 0;

            continue;
        }
        token.value[current_length++] = current_character;

        if (current_length == token.size)
            break;
    }
    token.value[token.size] = '\0';

    return token;

handle_error:
    print_error (error_message);

    fclose (fp);
    exit (-1);
}

void
parse_open_bracket (FILE* fp)
{
    mnn_token token;
    char error_message[1024];

    token = get_token (fp, 1);
    if (token.value) {
        if (strcmp (token.value, "{")) {
            sprintf (error_message, "Parsing config file. Expected '{' instead of %s at line %d.\n", token.value, line);
            goto hadle_error;
        }
        free (token.value);
    }
    else {
        sprintf (error_message, "Parsing config file. Missing token at line %d.\n", line);
        goto hadle_error;
    }

    return;

hadle_error:
    print_error (error_message);

    free (token.value);
    fclose (fp);
    exit (-1);
}

void
parse_parameters (FILE* fp, char** params, mnn_network_configuration* net_configuration)
{
    mnn_token token;

    int k, found, counter;

    long int current_position;

    size_t length_string;

    /* Check if we've parsed each parameter */
    for (k = 0, counter = 0; k < N_PARAMETERS; k++)
        if (params[k] == NULL)
            counter++;

    if (counter == N_PARAMETERS)
        return;

    /* Search parameter to parse */
    for (k = 0, found = 0; k < N_PARAMETERS; k++) {
        if (params[k] != NULL) {
            length_string    = strlen (params[k]);
            current_position = ftell (fp);
            token            = get_token (fp, length_string);

            if (!strcmp (token.value, params[k])) {
                free (token.value);
                handle_parameter (fp, params[k], net_configuration);

                params[k] = NULL;
                found     = 1;

                break;
            }
            else
                fseek (fp, current_position, SEEK_SET);
        }
    }
    if (!found) {
        print_error ("Parsing config file. Valid parameter not found.\n");

        fclose (fp); exit (-1);
    }

    parse_parameters (fp, params, net_configuration);
}

void
handle_parameter (FILE* fp, char* param_value, mnn_network_configuration* net_configuration)
{
    char error_message[1024];

    if (!strcmp (param_value, "hidden-layers"))
        handle_hidden_layers_parameter (fp, net_configuration);
    else if (!strcmp (param_value, "learning-rate"))
        handle_learning_rate_parameter (fp, net_configuration);
    else if (!strcmp (param_value, "epochs"))
        handle_epochs_parameter (fp, net_configuration);
    else if (!strcmp (param_value, "mini-batches"))
        handle_mini_batches_parameter (fp, net_configuration);
    else {
        sprintf (error_message, "Parsing config file. Unknown parameter \"%s\" at line %d.\n", param_value, line);
        print_error (error_message);

        fclose (fp); exit (-1);
    }
}

void
handle_hidden_layers_parameter (FILE* fp, mnn_network_configuration* net_configuration)
{
    mnn_token token;
    int k, can_contain_space, hidden_k, index_n;
    char token_c, error_message[1024], buffer[1024];
    size_t length = 0, num_hidden_layers = 1;

    token = get_token (fp, 1);
    __assert ("'handle_hidden_layers_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (1)", token.size == 1);

    if (token.value) {
        if (token.value[0] != ':') {
            sprintf (error_message, "Expected ':' instead of %s at line %d.\n", token.value, line);
            goto handle_error;
        }
        free (token.value);

        token = get_token (fp, 1);
        __assert ("'handle_hidden_layers_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (2)", token.size == 1);

        /* Rest of the read characters must be numbers */
        if (token.value && is_number (token.value[0])) {
            buffer[length++] = token.value[0];
            free (token.value);

            #ifdef DEBUG_PARSER
                print_info ("Handle hidden-layer. It's number.\n");
            #endif

            while (1) {
                token_c = fgetc (fp);

                if (is_number (token_c))
                    buffer[length++] = token_c;
                else if (token_c == ';')
                    break;
                else {
                    sprintf (error_message, "Parsing config file. Expected number or ';' instead of \"%c\" at line %d.\n", token_c, line);
                    goto handle_error;
                }
            }
            buffer[length++] = '\0';

            #ifdef DEBUG_PARSER
                print_info ("Handle hidden-layer. It's number. Final value is:  ");
                printf ("%s\n", buffer);
            #endif
        }
        /* Array of integers */
        else if (token.value && token.value[0] == '[') {
            free (token.value);

            #ifdef DEBUG_PARSER
                print_info ("Handle hidden-layer. It's an array.\n");
            #endif

            can_contain_space = 1;
            while (1) {
                token_c = fgetc (fp);

                if (can_contain_space && (token_c == ' ' || token_c == '\t'))
                    continue;

                if (is_number (token_c)) {
                    buffer[length++]  = token_c;
                    can_contain_space = 0;
                }
                else if (token_c == ',') {
                    if (length > 0 && !is_number (buffer[length - 1])) {
                        sprintf (error_message, "Parsing config file. Expected number instead of \"%c\" at line %d.\n", token_c, line);
                        goto handle_error;
                    }
                    /* Insert separator to distinguish numbers of array */
                    buffer[length++]  = '_';
                    can_contain_space = 1;

                    num_hidden_layers++;
                }
                else if (token_c == ']') {
                    token = get_token (fp, 1);
                    __assert ("'handle_hidden_layers_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (5)", token.size == 1);

                    if (token.value && ((token.value[0] != ';') || (length == 0 || (length > 0 && !is_number (buffer[length - 1]))))) {
                        sprintf (error_message, "Parsing config file. Expected number instead of \"%s\" at line %d.\n", token.value, line);
                        free (token.value);
                        goto handle_error;
                    }
                    free (token.value);

                    break;
                }
                else {
                    sprintf (error_message, "Parsing config file. Expected number or ';' instead of \"%c\" at line %d.\n", token_c, line);
                    goto handle_error;
                }
            }
            buffer[length++] = '\0';

            #ifdef DEBUG_PARSER
                print_info ("Handle hidden-layer. It's an array. Final value is:  ");
                printf ("%s\n", buffer);
            #endif
        }
        /* First character is not number and '[' */
        else {
            sprintf (error_message, "Parsing config file. Expected number or '[...]' instead of %s at line %d.\n", token.value, line);
            goto handle_error;
        }
    }
    else {
        sprintf (error_message, "Parsing config file. Missing token at line %d.\n", line);
        goto handle_error;
    }

    /* Use "buffer" to set hidden layers */
    net_configuration->net->sizes_hidden_neurons = allocate_vector (num_hidden_layers);

    hidden_k = (num_hidden_layers - 1);
    index_n  = 0;
    for (k = (strlen (buffer) - 1); k >= 0; k--) {
        if (buffer[k] == '_') {
            hidden_k--;
            index_n = 0;
        }
        else if (is_number (buffer[k]))
            net_configuration->net->sizes_hidden_neurons.values[hidden_k] += (size_t) ((buffer[k] - '0') * pow (10, index_n++));
        else {
            fclose (fp);

            __exit ("'handle_hidden_layers_parameter' function ('mnn_parser_config_file.c'), buffer with unknown character");
        }
    }

    #ifdef DEBUG_PARSER
        print_info ("Handle hidden-layer print vector:  ");
        printf ("[");
        for (k = 0; k < net_configuration->net->sizes_hidden_neurons.length; k++)
            printf ((k != net_configuration->net->sizes_hidden_neurons.length - 1) ? "%zu, " : "%zu",
                    (size_t) net_configuration->net->sizes_hidden_neurons.values[k]);
        printf ("]\n");
    #endif

    return;

handle_error:
    print_error (error_message);

    fclose (fp);
    exit (-1);
}

void
handle_learning_rate_parameter (FILE* fp, mnn_network_configuration* net_configuration)
{
    mnn_token token;
    char token_c, error_message[1024], buffer[1024];
    char* num_after_decimal_mark;
    size_t length = 0, lenght_num_before_decimal_mark;
    int k, contain_comma = 0;

    token = get_token (fp, 1);
    __assert ("'handle_hidden_layers_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (1)", token.size == 1);

    if (token.value) {
        if (token.value[0] != ':') {
            sprintf (error_message, "Expected ':' instead of %s at line %d.\n", token.value, line);

            goto handle_error;
        }
        free (token.value);

        token = get_token (fp, 1);
        __assert ("'handle_hidden_layers_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (2)", token.size == 1);

        /* Rest of the read characters must be numbers */
        if (token.value && (is_number (token.value[0]) || token.value[0] == '.')) {
            buffer[length++] = is_number (token.value[0]) ? token.value[0] : '0';
            if (token.value[0] == '.')
                buffer[length++] = '.';

            free (token.value);

            contain_comma = (buffer[length - 1] == '.') ? 1 : 0;
            while (1) {
                token_c = fgetc (fp);

                if (is_number (token_c))
                    buffer[length++] = token_c;
                else if (token_c == '.') {
                    if (contain_comma) {
                        sprintf (error_message, "Parsing config file. Expected number or ';' instead another '.' at line %d.\n", line);

                        goto handle_error;
                    }
                    buffer[length++] = token_c;
                    contain_comma    = 1;
                }
                else if (token_c == ';')
                    break;
                else {
                    sprintf (error_message, "Parsing config file. Expected number or ';' instead of \"%c\" at line %d.\n", token_c, line);

                    goto handle_error;
                }
            }
            /* Handle 'X.' where 'X' is a number and '.' the decimal mark */
            if (contain_comma && (length > 0 && buffer[length - 1] == '.'))
                buffer[length++] = '0';
            buffer[length++] = '\0';

            #ifdef DEBUG_PARSER
                print_info ("Handle learning rate. Final value is:  ");
                printf ("%s\n", buffer);
            #endif
        }
        /* First character is not number and '.' */
        else {
            sprintf (error_message, "Parsing config file. Expected number or '.' instead of %s at line %d.\n", token.value, line);

            goto handle_error;
        }
    }
    else {
        sprintf (error_message, "Parsing config file. Missing token at line %d.\n", line);

        goto handle_error;
    }

    /* Use "buffer" to set learning rate */
    num_after_decimal_mark         = strrchr (buffer, '.');
    lenght_num_before_decimal_mark = (!num_after_decimal_mark) ? strlen (buffer) : (strlen (buffer) - strlen (num_after_decimal_mark));

    /* Before decimal mark */
    net_configuration->training_params->learning_rate = 0.0;
    for (k = 0; k < lenght_num_before_decimal_mark; k++)
        net_configuration->training_params->learning_rate += (double) ((buffer[k] - '0') * pow (10, lenght_num_before_decimal_mark - k - 1));
    /* After decimal mark. "num_after_decimal_mark", if different from NULL, include '.' as first character */
    if (num_after_decimal_mark)
        for (k = 1; k < strlen (num_after_decimal_mark); k++)
            net_configuration->training_params->learning_rate += (double) ((num_after_decimal_mark[k] - '0') * pow (10, -k));

    #ifdef DEBUG_PARSER
        print_info ("Learning rate double value:  ");
        printf ("%f\n", net_configuration->training_params->learning_rate);
    #endif

    return;

handle_error:
    print_error (error_message);

    fclose (fp);
    exit (-1);
}

void
handle_epochs_parameter (FILE* fp, mnn_network_configuration* net_configuration)
{
    mnn_token token;
    char token_c, error_message[1024], buffer[1024];
    int k;
    size_t length = 0;

    token = get_token (fp, 1);
    __assert ("'handle_epochs_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (1)", token.size == 1);

    if (token.value) {
        if (token.value[0] != ':') {
            sprintf (error_message, "Expected ':' instead of %s at line %d.\n", token.value, line);

            goto handle_error;
        }
        free (token.value);

        token = get_token (fp, 1);
        __assert ("'handle_epochs_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (2)", token.size == 1);

        /* Rest of the read characters must be numbers */
        if (token.value && is_number (token.value[0])) {
            buffer[length++] = token.value[0];
            free (token.value);

            while (1) {
                token_c = fgetc (fp);

                if (is_number (token_c))
                    buffer[length++] = token_c;
                else if (token_c == ';')
                    break;
                else {
                    sprintf (error_message, "Parsing config file. Expected number or ';' instead of \"%c\" at line %d.\n", token_c, line);

                    goto handle_error;
                }
            }
            buffer[length++] = '\0';

            #ifdef DEBUG_PARSER
                print_info ("Handle epochs. Final value is:  ");
                printf ("%s\n", buffer);
            #endif
        }
        /* First character is not number and '.' */
        else {
            sprintf (error_message, "Parsing config file. Expected number or '.' instead of %s at line %d.\n", token.value, line);

            goto handle_error;
        }
    }
    else {
        sprintf (error_message, "Parsing config file. Missing token at line %d.\n", line);

        goto handle_error;
    }

    /* Use "buffer" to set number of epochs */
    length--;
    net_configuration->training_params->size_epochs = 0.0;
    for (k = 0; k < length; k++)
        net_configuration->training_params->size_epochs += (size_t) ((buffer[k] - '0') * pow (10, length - k - 1));

    #ifdef DEBUG_PARSER
        print_info ("Epochs value:  ");
        printf ("%zu\n", net_configuration->training_params->size_epochs);
    #endif

    return;

handle_error:
    print_error (error_message);

    fclose (fp);
    exit (-1);
}

void
handle_mini_batches_parameter (FILE* fp, mnn_network_configuration* net_configuration)
{
    mnn_token token;
    char token_c, error_message[1024], buffer[1024];
    int k;
    size_t length = 0;

    token = get_token (fp, 1);
    __assert ("'handle_mini_batches_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (1)", token.size == 1);

    if (token.value) {
        if (token.value[0] != ':') {
            sprintf (error_message, "Expected ':' instead of %s at line %d.\n", token.value, line);

            goto handle_error;
        }
        free (token.value);

        token = get_token (fp, 1);
        __assert ("'handle_mini_batches_parameter' function ('mnn_parser_config_file.c'), token.size != 1 (2)", token.size == 1);

        /* Rest of the read characters must be numbers */
        if (token.value && is_number (token.value[0])) {
            buffer[length++] = token.value[0];
            free (token.value);

            while (1) {
                token_c = fgetc (fp);

                if (is_number (token_c))
                    buffer[length++] = token_c;
                else if (token_c == ';')
                    break;
                else {
                    sprintf (error_message, "Parsing config file. Expected number or ';' instead of \"%c\" at line %d.\n", token_c, line);

                    goto handle_error;
                }
            }
            buffer[length++] = '\0';

            #ifdef DEBUG_PARSER
                print_info ("Handle mini-batches. Final value is:  ");
                printf ("%s\n", buffer);
            #endif
        }
        /* First character is not number and '.' */
        else {
            sprintf (error_message, "Parsing config file. Expected number or '.' instead of %s at line %d.\n", token.value, line);

            goto handle_error;
        }
    }
    else {
        sprintf (error_message, "Parsing config file. Missing token at line %d.\n", line);

        goto handle_error;
    }

    /* Use "buffer" to set number of mini-batches */
    length--;
    net_configuration->training_params->size_mini_batches = 0.0;
    for (k = 0; k < length; k++)
        net_configuration->training_params->size_mini_batches += (size_t) ((buffer[k] - '0') * pow (10, length - k - 1));

    #ifdef DEBUG_PARSER
        print_info ("Epochs value:  ");
        printf ("%zu\n", net_configuration->training_params->size_mini_batches);
    #endif

    return;

handle_error:
    print_error (error_message);

    fclose (fp);
    exit (-1);
}

void
parse_close_bracket (FILE* fp)
{
    mnn_token token;
    char error_message[1024];

    token = get_token (fp, 2);
    if (token.value) {
        if (strcmp (token.value, "};")) {
            sprintf (error_message, "Parsing config file. Expected '};' instead of %s at line %d.\n", token.value, line);
            goto hadle_error;
        }
        free (token.value);
    }
    else {
        sprintf (error_message, "Parsing config file. Missing token at line %d.\n", line);
        goto hadle_error;
    }

    return;

hadle_error:
    print_error (error_message);

    free (token.value);
    fclose (fp);
    exit (-1);
}

int
is_number (char character)
{
    return ((character >= '0') && (character <= '9'));
}

int
is_valid_alpha_character (char character)
{
    return (((character > 'a') && (character < 'z')) || character == '-');
}

int
is_skip_character (char character)
{
    return ((character == '\n') || (character == '\t') || (character == ' ') || (character == '#'));
}

char**
setup_parameters (void)
{
    char** avaiable_params;

    if (!(avaiable_params = ((char **) malloc (sizeof (char *) * N_PARAMETERS))))
        return NULL;

    avaiable_params[0] = "hidden-layers";
    avaiable_params[1] = "learning-rate";
    avaiable_params[2] = "epochs";
    avaiable_params[3] = "mini-batches";

    return avaiable_params;
}
