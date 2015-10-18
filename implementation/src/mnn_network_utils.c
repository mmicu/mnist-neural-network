#include "mnn_network_utils.h"

mnn_vector*
load_biases_in_the_file (mnn_network* net, int train, const char* file)
{
    /* If we must train the network, each bias layer must have certain size */
    FILE* fp;
    mnn_vector* biases;
    size_t length_layers, total_biases, length_biases_k;
    double value;
    int k, res, biases_k, element_k, counter;

    if (!(fp = check_correct_format_of_the_file (net, train, file)))
        return NULL;

    /* Setting biases pointer */
    length_layers = net->sizes_hidden_neurons.length + 1;
    biases        = (mnn_vector *) malloc (sizeof (mnn_vector) * length_layers);
    if (!biases)
        __exit ("'load_biases_in_the_file' function ('mnn_network_utils.c'), impossible to allocate biases");

    /* Calculation of total number of biases and allocate vectors */
    total_biases = 0;
    for (k = 0; k < length_layers; k++) {
        length_biases_k = (k == (length_layers - 1)) ? net->size_output_neurons : (size_t) net->sizes_hidden_neurons.values[k];
        total_biases   += length_biases_k;
        biases[k]       = allocate_vector (length_biases_k);
    }

    /* Read biases */
    biases_k = element_k = counter = 0;
    length_biases_k = net->sizes_hidden_neurons.values[0];
    while ((res = fscanf (fp, "%lf;", &value)) != EOF) {
        if (res == 1) {
            biases[biases_k].values[element_k++] = value;
            counter++;

            if (element_k == length_biases_k) {
                element_k = 0;

                if (++biases_k == length_layers)
                    break;

                length_biases_k = (biases_k == (length_layers - 1)) ? net->size_output_neurons
                                                                    : (size_t) net->sizes_hidden_neurons.values[biases_k];
            }
        }
        else {
            print_error ("Expected double value for biases.\n");

            goto free_biases;
        }
    }
    if (counter != total_biases) {
        print_error ("Wrong number of read biases based on the network configuration.\n");

        goto free_biases;
    }
    fclose (fp);

    return biases;

free_biases:
    fclose (fp);
    for (k = 0; k < length_layers; k++)
        free_vector (biases[k]);
    free (biases);

    return NULL;
}

mnn_vector*
load_biases_randomly (mnn_network* net)
{
    __assert ("'load_biases_randomly' function ('mnn_network_utils.c'), length < 0", net->sizes_hidden_neurons.length > 0);

    mnn_vector* biases;
    size_t length_biases;
    int k, j;

    /* Init biases randomly */
    biases        = get_pointer_biases (net);
    length_biases = net->sizes_hidden_neurons.length + 1;
    for (k = 0; k < length_biases; k++)
        for (j = 0; j < biases[k].length; j++)
            biases[k].values[j] = random_normal_distribution ();

    return biases;
}

mnn_vector*
get_pointer_biases (mnn_network* net)
{
    mnn_vector* biases;

    size_t* sizes_layers;

    size_t length_layers;

    int k;

    /* length_layers = |biases| */
    length_layers = net->sizes_hidden_neurons.length + 1;
    /* Init sizes_layers */
    sizes_layers  = (size_t *) malloc (sizeof (size_t) * length_layers);
    if (!sizes_layers)
        __exit ("'get_pointer_biases' function ('mnn_network_utils.c'), impossible to allocate sizes_layers");

    for (k = 0; k < length_layers; k++)
        sizes_layers[k] = (k == (length_layers - 1)) ? net->size_output_neurons : (size_t) net->sizes_hidden_neurons.values[k];

    /* Init biases pointer */
    biases = (mnn_vector *) malloc (sizeof (mnn_vector) * length_layers);
    if (!biases)
        __exit ("'get_pointer_biases' function ('mnn_network_utils.c'), impossible to allocate biases");

    /* All values will be set to 0 with "allocate_vector" function */
    for (k = 0; k < length_layers; k++)
        biases[k] = allocate_vector (sizes_layers[k]);

    free (sizes_layers);

    return biases;
}

mnn_matrix*
load_weights_in_the_file (mnn_network* net, int train, const char* file)
{
    /* If we must train the network, each weight layer must have certain size */
    FILE* fp;
    mnn_matrix* weights;
    size_t length_layers, total_biases, total_weights, row_k, col_k, weight_k, num_rows_weight_k, num_cols_weight_k;
    double value;
    int k, res, counter, number_to_jump;

    /* We already check the correct format of the file since we call "load_biases_in_the_file" first */
    if (!(fp = fopen (file, "r")))
        return NULL;

    /* Setting weights pointer */
    length_layers = net->sizes_hidden_neurons.length + 1;
    weights       = (mnn_matrix *) malloc (sizeof (mnn_matrix) * length_layers);
    if (!weights)
        __exit ("'load_weights_in_the_file' function ('mnn_network_utils.c'), impossible to allocate weights");

    /* Calculation of total number of biases to jump all their values and allocation of weights */
    total_biases = total_weights = 0;
    for (k = 0; k < length_layers; k++) {
        total_biases   += (k == (length_layers - 1)) ? net->size_output_neurons : (size_t) net->sizes_hidden_neurons.values[k];

        row_k = (k == (length_layers - 1)) ? net->size_output_neurons : (size_t) net->sizes_hidden_neurons.values[k];
        col_k = (k == 0)                   ? net->size_input_neurons  : (size_t) net->sizes_hidden_neurons.values[k - 1];

        total_weights += row_k * col_k;

        weights[k] = allocate_matrix (row_k, col_k);
    }
    /* Sum the hidden layers saved in the first line */
    total_biases += length_layers;

    /* Read weights */
    number_to_jump = weight_k = row_k = col_k = counter = 0;
    num_rows_weight_k = weights[0].rows;
    num_cols_weight_k = weights[0].cols;
    while ((res = fscanf (fp, "%lf;", &value)) != EOF) {
        if (number_to_jump != total_biases) {
            number_to_jump++;
            continue;
        }

        if (res == 1) {
            weights[weight_k].values[row_k][col_k++] = value;
            counter++;

            if (col_k == num_cols_weight_k) {
                col_k = 0;
                /* Last value read */
                if (++row_k == num_rows_weight_k) {
                    row_k = col_k = 0;

                    if (++weight_k == length_layers)
                        break;

                    num_rows_weight_k = weights[weight_k].rows;
                    num_cols_weight_k = weights[weight_k].cols;
                }
            }
        }
        else {
            print_error ("Expected double value for weights.\n");

            goto free_weights;
        }
    }
    if (counter != total_weights) {
        print_error ("Wrong number of read weights based on the network configuration.\n");

        goto free_weights;
    }
    fclose (fp);

    return weights;

free_weights:
    fclose (fp);
    for (k = 0; k < length_layers; k++)
        free_matrix (weights[k]);
    free (weights);

    return NULL;
}

mnn_matrix*
load_weights_randomly (mnn_network* net)
{
    __assert ("'load_weights_randomly' function ('mnn_network_utils.c'), length < 0", net->sizes_hidden_neurons.length > 0);

    mnn_matrix* weights;
    size_t length_weights;
    int k, j, i;

    /* Init all values randomly */
    weights        = get_pointer_weights (net);
    length_weights = net->sizes_hidden_neurons.length + 1;
    for (k = 0; k < length_weights; k++)
        for (j = 0; j < weights[k].rows; j++)
            for (i = 0; i < weights[k].cols; i++)
                weights[k].values[j][i] = random_normal_distribution ();

    return weights;
}

mnn_matrix*
get_pointer_weights (mnn_network* net)
{
    mnn_matrix* weights;

    size_t* sizes_layers;

    size_t length_layers;

    int k;

    length_layers = net->sizes_hidden_neurons.length + 2;
    sizes_layers  = (size_t *) malloc (sizeof (size_t) * length_layers);
    if (!sizes_layers)
        __exit ("'load_weights_randomly' function ('mnn_network_utils.c'), impossible to allocate sizes_layers");

    /* Init sizes */
    for (k = 0; k < length_layers; k++)
        sizes_layers[k] = (k == 0) ? net->size_input_neurons :
                          (k == (length_layers - 1)) ? net->size_output_neurons : (size_t) net->sizes_hidden_neurons.values[k - 1];

    /* Init weights pointer */
    weights = (mnn_matrix *) malloc (sizeof (mnn_matrix) * (length_layers - 1));
    if (!weights)
        __exit ("'load_weights_randomly' function ('mnn_network_utils.c'), impossible to allocate weights");

    /* All values will be set to 0 with "allocate_matrix" function */
    for (k = 0; k < (length_layers - 1); k++)
        weights[k] = allocate_matrix (sizes_layers[k + 1], sizes_layers[k]);

    free (sizes_layers);

    return weights;
}

FILE*
check_correct_format_of_the_file (mnn_network* net, int train, const char* file)
{
    FILE* fp;
    size_t length_layers, read_length_layers, read_length_k;
    int k, res;

    if (!(fp = fopen (file, "r")))
        return NULL;

    length_layers = net->sizes_hidden_neurons.length;
    /* Read hidden layers values */
    if ((res = fscanf (fp, "%zu", &read_length_layers)) != 1) {
        print_error ("File format is not correct.\n");

        fclose (fp);

        return NULL;
    }
    if (read_length_layers != length_layers) {
        print_error ("Length of the layers read in the file is different from the network configuration.\n");

        fclose (fp);

        return NULL;
    }
    for (k = 0; k < length_layers; k++) {
        if ((res = fscanf (fp, "%zu", &read_length_k)) != 1) {
            print_error ("Expected number of the length of the k-th hidden layer.\n");

            fclose (fp);

            return NULL;
        }
        if (net->sizes_hidden_neurons.values[k] != read_length_k) {
            print_error ("Value of the k-th hidden layer read in the file is different from the network configuration.\n");

            fclose (fp);

            return NULL;
        }
    }

    return fp;
}

int
save_weights_and_biases (mnn_network* net, mnn_matrix* weights, mnn_vector* biases, const char* file)
{
    int k, j, i;
    FILE* fp;
    size_t length_layers;

    if (!(fp = fopen (file, "w")))
        return -1;

    /* length_layers = |biases| = |weights| */
    length_layers = net->sizes_hidden_neurons.length + 1;

    /* Write network information */
    /* Write number of hidden layers */
    fprintf (fp, "%zu ", net->sizes_hidden_neurons.length);
    for (k = 0; k < net->sizes_hidden_neurons.length; k++)
        fprintf (fp, (k != (net->sizes_hidden_neurons.length - 1)) ? "%zu " : "%zu\n", (size_t) net->sizes_hidden_neurons.values[k]);

    /* Write biases */
    for (k = 0; k < length_layers; k++) {
        for (j = 0; j < biases[k].length; j++)
            fprintf (fp, "%f;", biases[k].values[j]);

        fprintf (fp, "\n");
    }

    /* Write weights */
    for (k = 0; k < length_layers; k++) {
        for (j = 0; j < weights[k].rows; j++)
            for (i = 0; i < weights[k].cols; i++)
                fprintf (fp, "%f;", weights[k].values[j][i]);

        fprintf (fp, "\n");
    }
    fclose (fp);

    return 1;
}

int
save_weights_and_biases_into_js_file (mnn_network* net, mnn_matrix* weights, mnn_vector* biases, const char* file)
{
    int k, j, i;
    FILE* fp;
    size_t length_layers;

    if (!(fp = fopen (file, "w")))
        return -1;

    /* length_layers = |biases| = |weights| */
    length_layers = net->sizes_hidden_neurons.length + 1;

    /* Write network information */
    fprintf (fp, "// Architecture neural network:\n//   - Input layer:  %zu;\n//   - Hidden layer:  %zu, values: [",
             net->size_input_neurons, net->sizes_hidden_neurons.length);
    for (k = 0; k < net->sizes_hidden_neurons.length; k++)
        fprintf (fp, (k != (net->sizes_hidden_neurons.length - 1)) ? "%zu, " : "%zu]\n", (size_t) net->sizes_hidden_neurons.values[k]);
    fprintf (fp, "//   - Output layer:  %zu;\n", net->size_output_neurons);

    /* Write biases */
    fprintf (fp, "\n// Number of vectors: %zu. Length of each vector: [", length_layers);
    for (k = 0; k < length_layers; k++)
        fprintf (fp, k != (length_layers - 1) ? "%zu, " : "%zu].\n", biases[k].length);
    fprintf (fp, "var biases = \n[");
    for (k = 0; k < length_layers; k++) {
        fprintf (fp, "\n    $V ([");
        for (j = 0; j < biases[k].length; j++)
            fprintf (fp, j != (biases[k].length - 1) ? "%f, " : "%f])", biases[k].values[j]);

        fprintf (fp, k != (length_layers - 1) ? "," : "\n");
    }
    fprintf (fp, "];\n\n");

    /* Write weights */
    fprintf (fp, "\n// Number of matrices: %zu. Rows and columns of each matrix: [", length_layers);
    for (k = 0; k < length_layers; k++)
        fprintf (fp, k != (length_layers - 1) ? "%zux%zu, " : "%zux%zu].\n", weights[k].rows, weights[k].cols);
    fprintf (fp, "var weights = \n[");
    for (k = 0; k < length_layers; k++) {
        fprintf (fp, "\n    $M\n    (\n        [\n");
        for (j = 0; j < weights[k].rows; j++) {
            fprintf (fp, "            [");
            for (i = 0; i < weights[k].cols; i++)
                fprintf (fp, i != (weights[k].cols - 1) ? "%f, " : "%f]", weights[k].values[j][i]);

            fprintf (fp, j != (weights[k].rows - 1) ? ",\n" : "\n");
        }
        fprintf (fp, "        ]\n");
        fprintf (fp, k != (length_layers - 1) ? "    )," : "    )");
    }
    fprintf (fp, "\n];\n");

    fclose (fp);

    return 1;
}

void
print_info_network (mnn_network_configuration* net_configuration, mnn_network_options* net_options)
{
    int k;

    printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Input neurons:      %zu\n", net_configuration->net->size_input_neurons);
    printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Hidden neurons:     %zu, ", net_configuration->net->sizes_hidden_neurons.length);
    for (k = 0; k < net_configuration->net->sizes_hidden_neurons.length; k++) {
        printf ((k == 0) ? "[" : "");
        printf ((k != (net_configuration->net->sizes_hidden_neurons.length - 1)) ? "%zu, " : "%zu", (size_t) net_configuration->net->sizes_hidden_neurons.values[k]);
        printf ((k == (net_configuration->net->sizes_hidden_neurons.length - 1)) ? "]\n" : "");
    }
    printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Output neurons:     %zu\n",  net_configuration->net->size_output_neurons);
    printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Size epochs:        %zu\n",  net_configuration->training_params->size_epochs);
    printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Size mini batches:  %zu\n",  net_configuration->training_params->size_mini_batches);
    printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Learning rate:      %f\n", net_configuration->training_params->learning_rate);
    if (net_options->load_file)
        printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Load file:          %s\n", net_options->load_file);
    if (net_options->output_file)
        printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Output file:        %s\n", net_options->output_file);
    if (net_options->image_file)
        printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Image file:         %s\n", net_options->image_file);
    if (net_options->js_file)
        printf (ANSI_CYAN "   [Info]  " ANSI_RESET "Javascript file:    %s\n", net_options->js_file);
}
