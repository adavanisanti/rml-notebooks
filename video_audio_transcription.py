from mpi4py import MPI
import numpy
import speech_recognition as sr
import os
import subprocess
import uuid
import sys

def get_uuid(truncation=8):
    return str(uuid.uuid4())[:truncation]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

vf = sys.argv[1]

file_list = [vf for i in range(size)]
audio_dir = "/tmp/jobs_folder"

recognizer = sr.Recognizer()

_file_size = len(file_list)/size
_start = rank*_file_size
_end  = (rank+1)*_file_size

if rank == (size-1):
    _end = len(file_list)
    
for _file_id in range(_start,_end):
    # Get Audio Filename
    vf = file_list[_file_id]
    file_path,file_name = os.path.split(vf)
    audio_file_name = file_name.split(".")[0] + "-" + get_uuid() + ".wav"
    folder_name = audio_dir + "/rank_" + str(rank)
    try:
        os.makedirs(folder_name)
    except:
        print("Directory %s exists"%folder_name)
    audio_file_name = folder_name + "/" + audio_file_name
    print audio_file_name,file_name,file_path
    
    # Create Audio file using ffmpeg
    av_call = "ffmpeg -i " + vf + " -acodec pcm_s16le -ac 1 -ar 16000 " + audio_file_name
    subprocess.call(av_call.split(),stderr=open(os.devnull,'wb'), stdout=open(os.devnull,'wb'))

    # Audio to text
    #with sr.AudioFile(audio_file_name) as source:
    #audio = recognizer.record(source)
