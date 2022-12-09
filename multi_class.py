from utils.preprocessor import PreprocessorClass

if __name__ == '__main__':
    dm = PreprocessorClass(preprocessed_dir= "data/preprocessed",
                          batch_size= 10,
                          max_length= 100)