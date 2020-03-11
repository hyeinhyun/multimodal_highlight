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
    enable_word_time_offsets = True
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,#oriin -> 44100
        language_code='en-US',enable_word_time_offsets= enable_word_time_offsets)
    operation = client.long_running_recognize(config, audio)
    response = operation.result()
    return response




if __name__ == '__main__':
    response = transcribe_gcs('gs://dataproc-temp-asia-east1-415006565877-tghovk9p/14-24-30-1.wav')
    
    
    with open('stt_result_1.txt', "w") as script:
        for result in response.results:
            alternative = result.alternatives[0]
            print(alternative.transcript)
            print(alternative)
            for word in alternative.words:
                print(word.word)
                script.write(u'{}'.format(word.word)+"\n")
                #start time
                script.write(u"{} ".format(word.start_time.seconds))
                #script.write(u"End time: {} seconds ".format(word.end_time.seconds))
                script.write("\n")
    print("completed")