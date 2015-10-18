#include "mnn_main.h"

int
main (int argc, char** argv)
{
    int index_config_file, index_save_parameters, index_load_parameters, index_image, index_training, index_show_image, index_export_js;
    mnn_network* net;
    mnn_network_options* net_options;
    mnn_network_configuration* net_configuration;
    char response = 0;

    /* User specify "--help" or "--show-image" option */
    if (contains_option (argv, argc, "--help") || argc == 1)
        usage (argv[0]);
    if ((index_show_image = contains_option (argv, argc, "--show-image"))) {
        if ((index_show_image + 1) >= argc)
            __exit ("If you specify \"--show-image\" option, you must specify a file");

        if (!file_exists (argv[index_show_image + 1]))
            __exit ("If you specify \"--show-image\" option, you must specify an existing file");

        show_image (argv[index_show_image + 1]);
    }

    /* Get index of *argv based on the option specified */
    index_config_file     = contains_option (argv, argc, "--config-file");
    index_save_parameters = contains_option (argv, argc, "--save-parameters");
    index_load_parameters = contains_option (argv, argc, "--load-parameters");
    index_image           = contains_option (argv, argc, "--image");
    index_training        = contains_option (argv, argc, "--train");
    index_export_js       = contains_option (argv, argc, "--export-parameters");

    /* No valid options */
    if (!index_config_file && !index_save_parameters && !index_load_parameters && !index_image && !index_training && !index_export_js && !index_show_image)
        __exit ("No valid options. You should use '--help' option");

    /* Initialize options of the network */
    net_options = (mnn_network_options *) malloc (sizeof (mnn_network_options));
    if (!net_options)
        __exit ("'main' function ('mnn_main.c'), impossible to allocate net_options");

    /* We train the network if we specify "--train" and/or "--save-parameters" */
    net_options->training = (!index_training) ? ((index_save_parameters) ? 1 : 0) : 1;

    /* Handle '--save-parameters' option */
    net_options->output_file = NULL;
    if (index_save_parameters) {
        if ((index_save_parameters + 1) >= argc) {
            free (net_options);

            __exit ("If you specify \"--save-parameters\" option, you must specify a file");
        }

        if (file_exists (argv[index_save_parameters + 1])) {
            do {
                print_warning ("Output file already exists. Do you want to overwrite it [y/n]?   ");
                response = getchar (); getchar ();
            } while (response != 'y' && response != 'n');

            if (response == 'n' && index_training == 0) {
                free (net_options);

                exit (0);
            }
        }

        net_options->output_file = (response == 'y' || response == 0) ? argv[index_save_parameters + 1] : NULL;
    }

    /* Handle '--load-parameters' option */
    net_options->load_file = NULL;
    if (index_load_parameters) {
        if ((index_load_parameters + 1) >= argc) {
            free (net_options);

            __exit ("If you specify \"--load-parameters\" option, you must specify a file");
        }

        if (!file_exists (argv[index_load_parameters + 1])) {
            free (net_options);

            __exit ("If you specify \"--load-parameters\" option, you must specify an existing file");
        }

        net_options->load_file = argv[index_load_parameters + 1];
    }

    /* Handle '--image' option */
    net_options->image_file = NULL;
    if (index_image) {
        if ((index_image + 1) >= argc) {
            free (net_options);

            __exit ("If you specify \"--image\" option, you must specify a file");
        }

        if (!file_exists (argv[index_image + 1])) {
            free (net_options);

            __exit ("If you specify \"--image\" option, you must specify an existing file");
        }

        if (!index_load_parameters && !net_options->training)
            print_warning ("Random values for weights and biases should predict wrong number.\n");

        net_options->image_file = argv[index_image + 1];
    }

    /* Handle '--export-parameters' option */
    net_options->js_file = NULL;
    if (index_export_js) {
        if ((index_export_js + 1) >= argc) {
            free (net_options);

            __exit ("If you specify \"--export-parameters\" option, you must specify a file");
        }

        /* File exists */
        response = 0;
        if (file_exists (argv[index_export_js + 1])) {
            do {
                print_warning ("Javascript file already exists. Do you want to overwrite it [y/n]?   ");
                response = getchar (); getchar ();
            } while (response != 'y' && response != 'n');

            if (response == 'n' && net_options->training == 0) {
                free (net_options);

                exit (0);
            }
        }

        net_options->js_file = (response == 'y' || response == 0) ? argv[index_export_js + 1] : NULL;
    }

    /* Parsing of the command line terminates with success */
    /* Create the network */
    net = (mnn_network *) malloc (sizeof (mnn_network));
    if (!net) {
        free (net_options);

        __exit ("'main' function ('mnn_main.c'), impossible to allocate net");
    }

    /* mnn_config.h contains constants */
    net->size_input_neurons  = N_INPUT_LAYER;
    net->size_output_neurons = N_OUTPUT_LAYER;

    /* Create network configuration with training parameters */
    net_configuration = (mnn_network_configuration *) malloc (sizeof (mnn_network_configuration));
    if (!net_configuration) {
        free (net_options);
        free (net);

        __exit ("'main' function ('mnn_main.c'), impossible to allocate net_configuration");
    }
    /* Set "net" pointer */
    net_configuration->net = net;

    /* Set all fields of net_configuration struct */
    /* Handle '--save-parameters' option */
    if (index_config_file) {
        if ((index_config_file + 1) >= argc) {
            free (net_options);
            free (net);

            __exit ("If you specify \"--config-file\" option, you must specify a file");
        }

        if (!file_exists (argv[index_config_file + 1])) {
            free (net_options);
            free (net);

            __exit ("If you specify \"--config-file\" option, you must specify an existing file");
        }
    }

    /* Setting parameters based on the config file and execute network */
    if (parse_config_file (net_configuration, (index_config_file) ? argv[index_config_file + 1] : NULL)) {
        /* Config file has been parsed successfully, so we can train the network */
        print_success ("Config file loaded successfully.\n");

        execute_network (net_configuration, net_options);
    }

    /* Free pointers */
    free (net_options);
    free_vector (net_configuration->net->sizes_hidden_neurons);
    free (net_configuration->net);
    free (net_configuration->training_params);
    free (net_configuration);

    return 0;
}

void
usage (char* argv_0)
{
    printf ("Usage: %s [Options]\n\n", argv_0);
    printf ("Options:\n");
    printf ("   --help                               Print this help\n");
    printf ("   --config-file <file>                 Specify config file to load\n");
    printf ("                                        (if you don't specify this option, \"default_configuration.conf\" will be loaded)\n");
    printf ("   --load-parameters <file>             Load weights and biases contained in the file\n");
    printf ("                                        (you must use '--save-parameters' option in a previous execution)\n");
    printf ("   --save-parameters <file>             Save weights and biases in the file\n");
    printf ("   --image <file>                       Predict number of the image\n");
    printf ("                                        (file's name and its content must respect some constrains)\n");
    printf ("   --train                              Train the neural network\n");
    printf ("                                        (if you use \"--save-parameters\", this option will be implicit)\n");
    printf ("   --show-image <file>                  Show specified image\n");
    printf ("   --export-parameters <file>           Save weights and biases as matrices and arrays in the file with Javascript syntax\n");

    exit (0);
}

int
contains_option (char** options, int n_options, char* option)
{
    int k;

    for (k = 1; k < n_options; k++)
        if (!strcmp (options[k], option))
            return k;

    return 0;
}

void
show_image (char* image)
{
    FILE* fp;

    int res, index;

    double value;

    /* We already checked in the "main" function if file exists */
    fp = fopen (image, "r");

    index = 0;
    while ((res = fscanf (fp, "%lf", &value)) != EOF) {
        if (res == 1) {
            index++;
            printf ((value > 0) ? "#" : ".");
            printf ((index % N_COLS_IMAGE == 0) ? "\n" : "");
        }
        else {
            print_error ("'show_image' function ('mnn_main.c'), file format is not correct.\n");

            fclose (fp); exit (-1);
        }
    }
    fclose (fp);

    __assert ("'show_image' function ('mnn_main.c'), index incorrect value", index == (N_ROWS_IMAGE * N_COLS_IMAGE));
}
