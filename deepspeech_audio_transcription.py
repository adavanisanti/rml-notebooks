from mpi4py import MPI
import numpy
import speech_recognition as sr
import os
import subprocess
import uuid
import sys
import warnings

from deepspeech.model import Model
import scipy.io.wavfile as wav
warnings.filterwarnings("ignore")

def get_uuid(truncation=8):
    return str(uuid.uuid4())[:truncation]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

vf = sys.argv[1]
audio_dir = "/home/ubuntu/deepspeech"
file_list = [vf for i in range(size)] #file list, specify files
#file_list=([audio_dir+'/12_sec_mono_16bit.wav',audio_dir+'/startalk_mono_16000.wav'])
print(' ')
print('File list:',file_list)


_file_size = len(file_list)/size
print('Number of files:',count(_file_size))
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
    #print('_file_id: ',_file_id)
    #print('vf: ',vf)
    file_path,file_name = os.path.split(vf)
    audio_file_name = file_name.split(".")[0] + "-" + get_uuid() + ".wav"
    #print('audio_file_name: ',audio_file_name)
    folder_name = audio_dir + "/rank_" + str(rank)
    try:
        os.makedirs(folder_name)
    except:
        print("Directory %s exists \n"%folder_name)
    #audio_file_name = folder_name + "/" + audio_file_name
    #print(audio_file_name,file_name,file_path)
    
    # Create Audio file using ffmpeg
    #av_call = "ffmpeg -i " + vf + " -acodec pcm_s16le -ac 1 -ar 16000 " + audio_file_name
    #print('av_call: ',av_call)
    #subprocess.call(av_call.split(),stderr=open(os.devnull,'wb'), stdout=open(os.devnull,'wb'))
                #model location                   alphabet file
    ds = Model('/home/ubuntu/deepspeech/models/output_graph.pb', 26, 9, '/home/ubuntu/deepspeech/models/alphabet.txt', 500)
    fs, audio = wav.read(vf)
    processed_data=ds.stt(audio,fs)
    #processed_data=ds.stt(audio.flatten(),fs)

    # Audio to text
    data_save=str(folder_name)+'-'+str(file_name)+'-data.txt'
    with open(data_save,'a') as f:
           f.write(processed_data)  # read the entire audio file
            
    try:
        print('\nDeepSpeech says, "...'+str(processed_data)+'..."\n\nThe data has been stored in file: '+str(data_save)+'\n')
    except:
        print("print statement didn't work")
