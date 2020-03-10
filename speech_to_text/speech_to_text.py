# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 11:40:57 2020

@author: Gangmin
"""

def transcribe_gcs(gcs_uri):
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()
    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='en-US')
    operation = client.long_running_recognize(config, audio)
    response = operation.result()
    return response




if __name__ == '__main__':
    response = transcribe_gcs('gs://dataproc-temp-asia-east1-415006565877-tghovk9p/14-24-30.wav')
    
    
    with open('stt_result.txt', "w") as script:
        for result in response.results:
            script.write(u'{}'.format(result.alternatives[0].transcript)+"\n")
    print("completed")