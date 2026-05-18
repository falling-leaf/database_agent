# from morphingdb_test.series_test.slice_test.morphingdb_test import slice_init_data 
# slice_init_data()
# from morphingdb_test.series_test.year_predict_test.morphingdb_test import year_predict_init_data
# year_predict_init_data()
# from morphingdb_test.series_test.swarm_test.morphingdb_test import swarm_init_data
# swarm_init_data()
# from morphingdb_test.image_test.cifar10.morphingdb_test import cifar10_init_data
# cifar10_init_data()
# from morphingdb_test.image_test.imagenet.morphingdb_test import imagenet_init_data
# imagenet_init_data()
# from morphingdb_test.image_test.stanford_dogs.morphingdb_test import stanford_dogs_init_data
# stanford_dogs_init_data()

# # Text test init data functions
# from morphingdb_test.text_test.imdb.morphingdb_test import imdb_init_data
# imdb_init_data()
# from morphingdb_test.text_test.financial_phrasebank.morphingdb_test import financial_phrasebank_init_data
# financial_phrasebank_init_data()
# from morphingdb_test.text_test.sst2.morphingdb_test import sst2_init_data
# sst2_init_data()

# Import create_model functions for the 9 test methods
from morphingdb_test.series_test.slice_test.morphingdb_test import create_model as create_slice_model
from morphingdb_test.series_test.swarm_test.morphingdb_test import create_model as create_swarm_model
from morphingdb_test.series_test.year_predict_test.morphingdb_test import create_model as create_year_predict_model
from morphingdb_test.text_test.imdb.morphingdb_test import create_model as create_imdb_model
from morphingdb_test.text_test.financial_phrasebank.morphingdb_test import create_model as create_financial_phrasebank_model
from morphingdb_test.text_test.sst2.morphingdb_test import create_model as create_sst2_model
from morphingdb_test.image_test.cifar10.morphingdb_test import create_model as create_cifar10_model
from morphingdb_test.image_test.imagenet.morphingdb_test import create_model as create_imagenet_model
from morphingdb_test.image_test.stanford_dogs.morphingdb_test import create_model as create_stanford_dogs_model

# Import all model paths
from morphingdb_test.config import (
    slice_model_path,
    swarm_model_path,
    year_predict_model_path,
    cifar10_model_path,
    imagenet_model_path,
    stanford_dogs_model_path,
    financial_phrasebank_model_path,
    imdb_model_path,
    sst2_model_path
)

def create_all_models():
    """Call create_model functions for all 9 test methods"""
    print("Creating all models...")
    create_slice_model(slice_model_path)  # slice_test expects a model path parameter
    print("Slice model created")
    create_swarm_model()
    print("Swarm model created")
    create_year_predict_model()
    print("Year predict model created")
    create_imdb_model()
    print("IMDB model created")
    create_financial_phrasebank_model()
    print("Financial phrasebank model created")
    create_sst2_model()
    print("SST2 model created")
    create_cifar10_model()
    print("CIFAR10 model created")
    create_imagenet_model()
    print("ImageNet model created")
    create_stanford_dogs_model()
    print("Stanford Dogs model created")
    print("All models created successfully!")

# Call the create_all_models function
create_all_models()
