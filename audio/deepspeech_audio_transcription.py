from mpi4py import MPI
import numpy
import speech_recognition as sr
import os
import subprocess
import uuid
import sys
import warnings
warnings.filterwarnings("ignore")
from deepspeech.model import Model
import scipy.io.wavfile as wav

def get_uuid(truncation=8):
    return str(uuid.uuid4())[:truncation]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

file_list = []
# open file and read the content in a list
with open('audio_list.txt', 'r') as filehandle:  
    file_list = [current_file.rstrip() for current_file in filehandle.readlines()]

audio_dir = "/home/ubuntu/SoundDir"

#file_list=(['/home/ubuntu/SoundDir/chunk0_16bit.wav','/home/ubuntu/SoundDir/chunk1_16bit.wav','/home/ubuntu/SoundDir/chunk2_16bit.wav'])

print(' ')
print('File list:',file_list)

_file_size = len(file_list)/size
print('Number of files:',_file_size)
_start = rank*_file_size
_end  = (rank+1)*_file_size
print('Start:',_start,'End: ',_end)

if rank == (size-1):
    _end = len(file_list)
    
for _file_id in range(_start,_end): 
    print('rank: ',str(rank))
    # Get Audio Filename
    vf = file_list[_file_id]    
    print('file: ',vf)
    print(' ')
    file_path,file_name = os.path.split(vf)
    folder_name = audio_dir + "/rank_" + str(rank)
    try:
        os.makedirs(folder_name)
    except:
        print("Directory %s exists \n"%folder_name)
                #model location                   alphabet file
    ds = Model('/home/ubuntu/deepspeech/models/output_graph.pb', 26, 9, '/home/ubuntu/deepspeech/models/alphabet.txt', 500)
    fs, audio = wav.read(vf)
    processed_data=ds.stt(audio,fs)
    #processed_data=ds.stt(audio.flatten(),fs)

    seperate_save=str(folder_name)+'-'+str(file_name)+'-data.txt'
    with open(seperate_save,'a+') as f:
           f.write(processed_data)  # read the entire audio file    
    
    # Audio to text
    data_save='AudioData.txt'
    with open(data_save,'a+') as f:
           f.write(processed_data+'\r\r')  # read the entire audio file     
    try:
        print('\nDeepSpeech says, "...'+str(processed_data)+'..."\n\nThe data has been stored in file: '+str(data_save)+'\n')
    except:
        print("print statement didn't work")
