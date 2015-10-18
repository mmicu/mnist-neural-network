#include "mnn_network.h"

void
execute_network (mnn_network_configuration* net_configuration, mnn_network_options* net_options)
{
    int k, j, correct_predictions, max_correct_predictions, improvement_predictions, save_params;
    size_t length_layers;
    char str_prediction[1024], str_progress[1024];

    mnn_vector* biases,
              * best_biases;

    mnn_matrix* weights,
              * best_weights,
              * training_data,
              * test_data;

    int* training_labels,
       * test_labels;

    /* Print some info about the network, training parameters and input/output file */
    print_info_network (net_configuration, net_options);

    /* Init biases and weights */
    /* If we don't specify load file, all values are random */
    /* length_layers = |biases| = |weights| */
    length_layers = net_configuration->net->sizes_hidden_neurons.length + 1;

    biases  = (!net_options->load_file) ? load_biases_randomly (net_configuration->net)
                                        : load_biases_in_the_file (net_configuration->net, net_options->training, net_options->load_file);
    weights = (!net_options->load_file) ? load_weights_randomly (net_configuration->net)
                                        : load_weights_in_the_file (net_configuration->net, net_options->training, net_options->load_file);
    if (!biases || ! weights) {
        print_error ((!biases)  ? "Impossible to load biases.\n" : "");
        print_error ((!weights) ? "Impossible to load weights.\n" : "");

        exit (-1);
    }
    print_success ((!net_options->load_file) ? "Weights and biases set randomly.\n" : "Weights and biases loaded from the file.\n");

    /* Init best_biases and best_weights */
    best_biases  = get_pointer_biases (net_configuration->net);
    best_weights = get_pointer_weights (net_configuration->net);

    /* Train the network if the option '--train' ('--save-parameters also) was specified */
    if (net_options->training) {
        /* Load data (training and test) and labels (training and test) */
        if ((training_data = (load_training_data ())))
            print_success ("Training data loaded.\n");

        if ((training_labels = (load_training_labels ())))
            print_success ("Training labels loaded.\n");

        if ((test_data = (load_test_data ())))
            print_success ("Test data loaded.\n");

        if ((test_labels = (load_test_labels ())))
            print_success ("Test labels loaded.\n");

        /* Setting "max_correct_predictions" variable to know when we must modify biases and weights */
        max_correct_predictions = (net_options->load_file) ? test_network (net_configuration->net, test_data, test_labels, weights, biases) : -1;
        /* Update values based on biases and weights */
        update_best_parameters (length_layers, weights, biases, best_weights, best_biases);
        /* If there is not an improvement of the parameters, the file won't update even if option '--save-parameters' is specified */
        save_params = 0;
        /* Train the network */
        for (k = 0; k < net_configuration->training_params->size_epochs; k++) {
            j = improvement_predictions = 0;
            while (j < N_TRAINING_IMAGES) {
                update_mini_batch (
                    net_configuration,
                    training_data,
                    training_labels,
                    weights,
                    biases,
                    j
                );

                j += net_configuration->training_params->size_mini_batches;
            }

            correct_predictions = test_network (net_configuration->net, test_data, test_labels, weights, biases);

            if (correct_predictions > max_correct_predictions) {
                max_correct_predictions = correct_predictions;

                update_best_parameters (length_layers, weights, biases, best_weights, best_biases);

                improvement_predictions = save_params = 1;
            }
            sprintf (str_progress, "Epoch [%d, %zu] complete: %d/%d", k + 1,
                                   net_configuration->training_params->size_epochs, correct_predictions, N_TEST_IMAGES);
            strcat (str_progress, (improvement_predictions) ? " (new high value).\n" : ".\n");

            print_info (str_progress);

            memset (&str_progress[0], 0, sizeof (str_progress));
        }

        for (k = 0; k < N_TRAINING_IMAGES; k++)
            free_matrix (training_data[k]);
        free (training_data);

        for (k = 0; k < N_TEST_IMAGES; k++)
            free_matrix (test_data[k]);
        free (test_data);

        free (training_labels);
        free (test_labels);
    }

    /* Save parameters */
    if (net_options->output_file) {
        if (!save_params)
            print_warning ("The file won't update since there has not been an improvement of the parameters.\n");
        else {
            if (save_weights_and_biases (net_configuration->net, best_weights, best_biases, net_options->output_file))
                print_success ("Weights and biases were saved successfully.\n");
            else
                print_error ("Impossible to save weights and biases in the output file.\n");
        }
    }

    /* Export parameters with Javascript syntax */
    if (net_options->js_file) {
        if (!save_params)
            print_warning ("The Javascript file won't update since there has not been an improvement of the parameters.\n");
        else {
            if (save_weights_and_biases_into_js_file (net_configuration->net, best_weights, best_biases, net_options->js_file))
                print_success ("Weights and biases were saved successfully in Javascript format.\n");
            else
                print_error ("Impossible to save weights and biases, in Javascript format, in the output file.\n");
        }
    }

    /* Predict output */
    if (net_options->image_file) {
        sprintf (str_prediction, "Image should contain number %d.\n",
                                 predict_output_image (net_configuration->net, net_options->image_file, weights, biases));

        print_info (str_prediction);
    }

    /* Free pointers */
    for (k = 0; k < length_layers; k++) {
        free_vector (biases[k]);
        free_matrix (weights[k]);
    }
    free (biases);
    free (weights);

    for (k = 0; k < length_layers; k++) {
        free_vector (best_biases[k]);
        free_matrix (best_weights[k]);
    }
    free (best_biases);
    free (best_weights);
}

void
update_mini_batch (mnn_network_configuration* net_configuration, mnn_matrix* training_data, int* training_labels,
                   mnn_matrix* weights, mnn_vector* biases, int index)
{
    int k, j, i, z, end_index, app_k;
    mnn_vector* b_app;
    mnn_matrix* w_app;
    mnn_backpropagation_data backprop_data;
    size_t length_layers;

    /* Init biases and weights */
    /* length_layers = |biases| = |weights| */
    length_layers = net_configuration->net->sizes_hidden_neurons.length + 1;
    /* Each value is 0 */
    b_app = get_pointer_biases (net_configuration->net);
    w_app = get_pointer_weights (net_configuration->net);

    end_index = index + net_configuration->training_params->size_mini_batches;
    for (k = index; k < end_index; k++) {
        app_k = (k >= N_TRAINING_IMAGES) ? (N_TRAINING_IMAGES - 1) : k;

        backprop_data = backpropagation (net_configuration->net, training_data[app_k], training_labels[app_k], weights, biases);

        for (j = 0; j < length_layers; j++) {
            /* Update biases */
            for (i = 0; i < b_app[j].length; i++)
                b_app[j].values[i] += backprop_data.biases_[j].values[i];

            /* Update weights */
            for (i = 0; i < w_app[j].rows; i++)
                for (z = 0; z < w_app[j].cols; z++)
                    w_app[j].values[i][z] += backprop_data.weights_[j].values[i][z];
        }

        /* Free backpropagation data */
        for (j = 0; j < length_layers; j++) {
            free_vector (backprop_data.biases_[j]);
            free_matrix (backprop_data.weights_[j]);
        }
        free (backprop_data.biases_);
        free (backprop_data.weights_);
    }

    /* Update biases and weights */
    for (k = 0; k < length_layers; k++) {
        /* Update biases */
        __assert (
            "'update_mini_batch' function ('mnn_network.c'), b_app.length != biases.length",
            b_app[k].length == biases[k].length
        );

        for (j = 0; j < biases[k].length; j++)
            biases[k].values[j] -= (double) (net_configuration->training_params->learning_rate /
                                   (double) net_configuration->training_params->size_mini_batches) * b_app[k].values[j];

        /* Update weights */
        __assert (
            "'update_mini_batch' function ('mnn_network.c'), w_app.rows != weights.rows",
            w_app[k].rows == weights[k].rows
        );
        __assert (
            "'update_mini_batch' function ('mnn_network.c'), w_app.cols != weights.cols",
            w_app[k].cols == weights[k].cols
        );

        for (j = 0; j < weights[k].rows; j++)
            for (i = 0; i < weights[k].cols; i++)
                weights[k].values[j][i] -= (double) (net_configuration->training_params->learning_rate /
                                           (double) net_configuration->training_params->size_mini_batches) * w_app[k].values[j][i];
    }

    /* Free pointers */
    for (k = 0; k < length_layers; k++) {
        free_vector (b_app[k]);
        free_matrix (w_app[k]);
    }
    free (b_app);
    free (w_app);
}

mnn_backpropagation_data
backpropagation (mnn_network* net, mnn_matrix x, int y, mnn_matrix* weights, mnn_vector* biases)
{
    mnn_backpropagation_data backprop_data;
    int k, j, i;
    size_t length_layers;
    mnn_matrix app, app_T;

    mnn_matrix* z_k,
              * a_k,
              * delta_k;

    /* Init biases and weights */
    /* length_layers = |biases| = |weights| */
    length_layers = net->sizes_hidden_neurons.length + 1;
    /* Each value is 0 */
    backprop_data.biases_  = get_pointer_biases (net);
    backprop_data.weights_ = (mnn_matrix *) malloc (sizeof (mnn_matrix) * length_layers);
    if (!backprop_data.weights_)
        __exit ("'backpropagation' function ('mnn_network.c'), impossible to allocate backprop_data.weights_");

    /* Setting z pointer */
    z_k = (mnn_matrix *) malloc (sizeof (mnn_matrix) * length_layers);
    if (!z_k)
        __exit ("'backpropagation' function ('mnn_network.c'), impossible to allocate z_k");

    /* Setting activation pointer */
    /* (length_layers + 1) because a(1) = x (where x is the input) */
    a_k = (mnn_matrix *) malloc (sizeof (mnn_matrix) * (length_layers + 1));
    if (!a_k)
        __exit ("'backpropagation' function ('mnn_network.c'), impossible to allocate a_k");
    /* Setting a(1) */
    a_k[0] = allocate_matrix (net->size_input_neurons, 1);
    for (k = 0; k < N_ROWS_IMAGE; k++)
        for (j = 0; j < N_COLS_IMAGE; j++)
            a_k[0].values[(k * N_ROWS_IMAGE) + j][0] = x.values[k][j];

    /* Setting delta pointer */
    delta_k = (mnn_matrix *) malloc (sizeof (mnn_matrix) * length_layers);
    if (!delta_k)
        __exit ("'backpropagation' function ('mnn_network.c'), impossible to allocate delta_k");

    /* Calculation of z_k and a_k => forward propagation */
    for (k = 0; k < length_layers; k++) {
        /* "matrix_dot_matrix" function allocates matrix */
        z_k[k] = matrix_dot_matrix (weights[k], a_k[k]);

        /* Sum biases of the k-th layer */
        __assert ("'backpropagation' function ('mnn_network.c'), z_k and biases", z_k[k].rows == biases[k].length);
        for (j = 0; j < z_k[k].rows; j++)
            for (i = 0; i < z_k[k].cols; i++)
                z_k[k].values[j][i] += biases[k].values[j];

        /* "matrix_sigmoid" function allocates matrix */
        a_k[k + 1] = matrix_sigmoid (z_k[k]);
    }

    /* Backpropagation */
    /* Handle the last layer */
    delta_k[length_layers - 1] = allocate_matrix (a_k[length_layers].rows, a_k[length_layers].cols);
    for (k = 0; k < delta_k[length_layers - 1].rows; k++)
        for (j = 0; j < delta_k[length_layers - 1].cols; j++)
            delta_k[length_layers - 1].values[k][j] = a_k[length_layers].values[k][j] - ((y == k) ? 1 : 0);

    __assert ("'backpropagation' function ('mnn_network.c'), delta output layer rows", delta_k[length_layers - 1].rows == z_k[length_layers - 1].rows);
    __assert ("'backpropagation' function ('mnn_network.c'), delta output layer cols", delta_k[length_layers - 1].cols == z_k[length_layers - 1].cols);
    for (k = 0; k < delta_k[length_layers - 1].rows; k++)
        for (j = 0; j < delta_k[length_layers - 1].cols; j++)
            delta_k[length_layers - 1].values[k][j] *= sigmoid_prime_to_number (z_k[length_layers - 1].values[k][j]);

    app = matrix_transpose (a_k[length_layers - 1]);
    backprop_data.weights_[length_layers - 1] = matrix_dot_matrix (delta_k[length_layers - 1], app);
    free_matrix (app);

    /* Handle others layers */
    for (k = (length_layers - 2); k >= 0; k--) {
        app   = matrix_sigmoid_prime (z_k[k]);
        app_T = matrix_transpose (weights[k + 1]);

        delta_k[k] = matrix_dot_matrix (app_T, delta_k[k + 1]);
        __assert ("'backpropagation' function ('mnn_network.c'), delta output k-th rows", app.rows == delta_k[k].rows);
        __assert ("'backpropagation' function ('mnn_network.c'), delta output k-th cols", app.cols == delta_k[k].cols);
        for (j = 0; j < delta_k[k].rows; j++)
            for (i = 0; i < delta_k[k].cols; i++)
                delta_k[k].values[j][i] *= app.values[j][i];
        free_matrix (app);
        free_matrix (app_T);

        app_T = matrix_transpose (a_k[k]);
        backprop_data.weights_[k] = matrix_dot_matrix (delta_k[k], app_T);
        free_matrix (app_T);
    }

    for (k = 0; k < length_layers; k++) {
        __assert ("'backpropagation' function ('mnn_network.c'), delta_k biases_k", backprop_data.biases_[k].length == delta_k[k].rows);

        for (j = 0; j < backprop_data.biases_[k].length; j++)
            backprop_data.biases_[k].values[j] = delta_k[k].values[j][0];
    }

    /* Free pointers */
    for (k = 0; k < length_layers; k++) {
        free_matrix (z_k[k]);
        free_matrix (a_k[k]);
        free_matrix (delta_k[k]);
    }
    free_matrix (a_k[k]);

    free (z_k);
    free (a_k);
    free (delta_k);

    return backprop_data;
}

int
forward (mnn_network* net, mnn_matrix x, mnn_matrix* weights, mnn_vector* biases)
{
    int k, j, i, res;
    mnn_matrix a_k, app, app_s;
    size_t length_layers;

    length_layers = net->sizes_hidden_neurons.length + 1;

    /* Set input matrix */
    a_k = allocate_matrix (N_ROWS_IMAGE * N_COLS_IMAGE, 1);
    for (k = 0; k < N_ROWS_IMAGE; k++)
        for (j = 0; j < N_COLS_IMAGE; j++)
            a_k.values[(k * N_ROWS_IMAGE) + j][0] = x.values[k][j];

    /* Forward */
    for (k = 0; k < length_layers; k++) {
        app = matrix_dot_matrix (weights[k], a_k);

        __assert ("'forward' function ('mnn_network.c'), biases", app.rows == biases[k].length);
        for (j = 0; j < app.rows; j++)
            for (i = 0; i < app.cols; i++)
                app.values[j][i] += biases[k].values[j];

        app_s = matrix_sigmoid (app);

        free_matrix (a_k);
        free_matrix (app);

        a_k = allocate_matrix (app_s.rows, app_s.cols);
        for (j = 0; j < a_k.rows; j++)
            for (i = 0; i < a_k.cols; i++)
                a_k.values[j][i] = app_s.values[j][i];
        free_matrix (app_s);
    }
    res = evaluate (a_k);

    free_matrix (a_k);

    return res;
}

int
evaluate (mnn_matrix m)
{
    __assert ("'evaluate' function ('mnn_network.c'), m.rows <= 0", m.rows > 0);
    __assert ("'evaluate' function ('mnn_network.c'), m.cols <= 0", m.cols > 0);

    int k, j, result;
    double max;

    result = 0;
    max    = m.values[0][0];
    for (k = 0; k < m.rows; k++) {
        for (j = 0; j < m.cols; j++) {
            if (m.values[k][j] > max) {
                max    = m.values[k][j];
                result = (k * m.cols) + j;
            }
        }
    }
    __assert ("'evaluate' function ('mnn_network.c'), result incorrect value", result >= 0 && result <= 9);

    return result;
}

int
test_network (mnn_network* net, mnn_matrix* test_data, int* test_labels, mnn_matrix* weights, mnn_vector* biases)
{
    int k, result, correct_predictions;

    for (k = 0, correct_predictions = 0; k < N_TEST_IMAGES; k++) {
        result = forward (net, test_data[k], weights, biases);

        __assert ("'test_network' function ('mnn_network.c'), result incorrect value", result >= 0 && result <= 9);

        if (test_labels[k] == result)
            correct_predictions++;
    }

    return correct_predictions;
}

int
predict_output_image (mnn_network* net, char* image_file, mnn_matrix* weights, mnn_vector* biases)
{
    FILE* fp;
    double value_pixel;
    int res_read, row_k, col_k, index, result;
    mnn_matrix x;

    x = allocate_matrix (N_ROWS_IMAGE, N_COLS_IMAGE);

    fp = fopen (image_file, "r");
    if (!fp)
        __exit ("'predict_output_image' function ('mnn_network.c'), impossibile to open the file");

    row_k = col_k = index = 0;
    while ((res_read = fscanf (fp, "%lf", &value_pixel)) != EOF) {
        if (res_read == 1) {
            if (row_k * col_k == N_ROWS_IMAGE * N_COLS_IMAGE) {
                print_error ("'predict_output_image' function ('mnn_network.c'), maximum size for the image is 28x28.\n");

                fclose (fp); exit (-1);
            }
            x.values[row_k][col_k++] = value_pixel;

            index++;

            if (col_k == N_COLS_IMAGE) {
                col_k = 0;
                row_k++;
            }
        }
        else {
            print_error ("'predict_output_image' function ('mnn_network.c'), file format is not correct.\n");

            fclose (fp); exit (-1);
        }
    }
    fclose (fp);
    __assert ("'backpropagation' function ('mnn_network.c'), index incorrect value", index == (N_ROWS_IMAGE * N_COLS_IMAGE));

    result = forward (net, x, weights, biases);

    free_matrix (x);

    return result;
}

void
update_best_parameters (size_t length_layers, mnn_matrix* weights, mnn_vector* biases, mnn_matrix* best_weights, mnn_vector* best_biases)
{
    int k, j, i;

    for (k = 0; k < length_layers; k++) {
        /* Update biases */
        for (j = 0; j < best_biases[k].length; j++)
            best_biases[k].values[j] = biases[k].values[j];

        /* Update weights */
        for (j = 0; j < best_weights[k].rows; j++)
            for (i = 0; i < best_weights[k].cols; i++)
                best_weights[k].values[j][i] = weights[k].values[j][i];
    }
}
