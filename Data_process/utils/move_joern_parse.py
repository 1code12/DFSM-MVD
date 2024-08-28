
import os
import shutil

destination_directory = ''


for i in range(): 

    source_directory = f'CPG_process/data_processing/{i}.c/tmp/{i}.c'
    destination_subdirectory = os.path.join(destination_directory, f'{i}.c')


    shutil.copytree(source_directory, destination_subdirectory)


