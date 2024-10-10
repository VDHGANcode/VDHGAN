## Data preprocessing
Fristly, use this version of joern to parse the function:

1.run ```VDHGAN-main/data_processing/joern/parseDataSet.py``` to  convert each function into a c file and store it in a temporary folder.

2.run ```VDHGAN-main/data_processing/process.py``` to use joern to parse each function.

3.run ```VDHGAN-main/data_processing/split_json_file.py``` to partition the dataset. Change the path of the dataset to the correct path

After parsing the functions, use ```VDHGAN-main/data_processing/word2vec.py``` for training the word2vec model. 

run ```VDHGAN-main/data_processing/split_json_file.py``` to partition the dataset.

For constructing and simplifying the code structure graphs, start from ```VDHGAN-main/data_processing/cpg_generate.py```.
