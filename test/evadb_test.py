# evadb 
from morphingdb_test.series_test.slice_test.evadb_test import evadb_slioe_test
evadb_slioe_test()
from morphingdb_test.series_test.year_predict_test.evadb_test import evadb_year_predict_test
evadb_year_predict_test()

from morphingdb_test.image_test.cifar10.evadb_test import evadb_cifar_test
evadb_cifar_test()
from morphingdb_test.image_test.imagenet.evadb_test import evadb_imagenet_test
evadb_imagenet_test()
from morphingdb_test.image_test.stanford_dogs.evadb_test import evadb_stanford_dogs_test
evadb_stanford_dogs_test()

from morphingdb_test.text_test.financial_phrasebank.evadb_test import evadb_financial_phrasebank_test
evadb_financial_phrasebank_test()
from morphingdb_test.text_test.imdb.evadb_test import evadb_imdb_test
evadb_imdb_test()
from morphingdb_test.text_test.sst2.evadb_test import evadb_sst2_test
evadb_sst2_test()

from morphingdb_test.muti_query.evadb_test import evadb_muti_query_test
evadb_muti_query_test()