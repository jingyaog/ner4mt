import zipfile, os

if __name__ == "__main__":
    
    
    # dir_from = "~/lorelei_ben_representative_lang_pack/data/monolingual_text/zipped/"
    # dir_to = "~/lorelei_ben_representative_lang_pack/data/monolingual_text/unzipped/"
    dir_from = "/Users/gongjingyao/Desktop/S23_NER/lorelei_ukrainian_repr_lang_pack/data/monolingual_text/zipped/"
    dir_to = "/Users/gongjingyao/Desktop/S23_NER/lorelei_ukrainian_repr_lang_pack/data/monolingual_text/unzipped/"
    extension = "ltf.zip"

    os.chdir(dir_from) # change directory from working dir to dir with files

    for item in os.listdir(dir_from): # loop through items in dir
        if item.endswith(extension): # check for ".zip" extension
            file_name = os.path.abspath(item) # get full path of files
            zip_ref = zipfile.ZipFile(file_name) # create zipfile object
            zip_ref.extractall(dir_to) # extract file to dir
            zip_ref.close() # close file
