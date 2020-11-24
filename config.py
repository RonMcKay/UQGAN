from os.path import abspath


class Config:
    root_save_folder = r"./experiments"
    root_metric_export_folder = abspath("./evaluation")
    root_plot_export_folder = abspath("./visualization/plots")

    # Dataset root paths
    celeba_root = r"/data/datasets/cl"
    cifar10_root = r"/data/datasets/cl/CIFAR10"
    cifar100_root = r"/data/datasets/cl/CIFAR100"
    clawa2_root = r"/data/datasets/cl/Animals_with_Attributes2"
    emnist_root = r"/data/datasets/cl"
    fmnist_root = r"/data/datasets/cl"
    lsun_root = r"/data/datasets/cl/LSUN"
    mnist_root = r"/data/datasets/cl"
    omniglot_root = r"/data/datasets/cl/Omniglot"
    svhn_root = r"/data/datasets/cl/SVHN"
    tinyimagenet_root = r"/data/datasets/cl/tinyimagenet"

    # If you want to use a MongoDB enter the required information here
    # The password should be placed on a single line in a password file
    mongo_url = r"mongodb_host:27017"
    mongo_db_name = r"your_mongodb_db_name"
    mongo_username = r"your_mongodb_username"
    mongo_password_file = r"path_to_your_mongodb_password_file"

    log_level = r"INFO"

    # Miscellaneous
    max_col_width = 50  # Maximum column width for option printing
