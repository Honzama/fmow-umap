# UMAP
umap_metrics = ["euclidean",  # 0
                "manhattan",  # 1
                "chebyshev",  # 2
                "minkowski",  # 3
                "canberra",  # 4
                "braycurtis",  # 5
                "mahalanobis",  # 6
                "wminkowski",  # 7
                "seuclidean",  # 8
                "cosine",  # 9
                "correlation",  # 10
                "haversine",  # 11
                "hamming",  # 12
                "jaccard",  # 13
                "dice",  # 14
                "russelrao",  # 15
                "kulsinski",  # 16
                "rogerstanimoto",  # 17
                "sokalmichener",  # 18
                "sokalsneath",  # 19
                "yule"]  # 20

# DATASETS
path_cifar10 = ""
path_cifar100 = ""
path_fmow_rgb = ""

str_mnist = "mnist"
str_fashion_mnist = "fashion-mnist"
str_cifar10 = "cifar-10"
str_cifar100 = "cifar-100"
str_fmow = "fmow"

# OUTPUT PATH
path_catch_files = ""
path_graphs = r""
graph_format = ".png"

# MNIST
mnist_category_names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# FASHION_MNIST
fashion_mnist_category_names = ["t-shirt/top", "trouser/pants", "pullover shirt", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

# CIFAR
cifar10_category_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
cifar100_category_names = ["apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
                           "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
                           "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
                           "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","computer_keyboard",
                           "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
                           "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
                           "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
                           "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
                           "squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor",
                           "train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm",]

# FMOW
fmow_category_names = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
                       "aquaculture", "archaeological_site", "barn", "border_checkpoint", "burial_site",
                       "car_dealership", "construction_site", "crop_field", "dam", "debris_or_rubble",
                       "educational_institution", "electric_substation", "factory_or_powerplant", "fire_station",
                       "flooded_road", "fountain", "gas_station", "golf_course", "ground_transportation_station",
                       "helipad", "hospital", "interchange", "lake_or_pond", "lighthouse", "military_facility",
                       "multi-unit_residential", "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
                       "parking_lot_or_garage", "place_of_worship", "police_station", "port", "prison", "race_track",
                       "railway_bridge", "recreational_facility", "impoverished_settlement", "road_bridge", "runway",
                       "shipyard", "shopping_mall", "single-unit_residential", "smokestack", "solar_farm",
                       "space_facility", "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
                       "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility", "wind_farm", "zoo"]
fmow_image_size = 224
