Highlight Detection
==================================

Extract the highlights of e-sports through multi-imodal analysis using video, audio and chat data.

### contributor 

    - HIhyun
    - GMpark

------------------------------------------------------

## video analysis

### image_data_extraction_for_OCR

    - video/lolData/frame_extraction.ipynb

### win_loss_classifier

    - data_directory : video/lolData/leagueoflegends2018~2020
    - data_preprocessing : video/lolData/lolApi2018~2019.ipynb
                           video/lolData/lolApi_without_2015.ipynb
    - win_loss_classifier_model : video/win_loss_classifier/MLP.ipynb

### highlight_point extractor

    - data_directory : video/lolData/exp_data
    - data_preprocessing : video/lolData/exp_data/2019_test_data.ipynb
                           video/lolData/exp_data/preprocessing.ipynb
    - highlight_point_extractor : video/experiment.ipynb
    - evaluation : video/evaluation.ipynb

------------------------------------------------------

## audio analysis

### energy_anaylsis_suing_emd

    - data_preprocessing : audio/audio_split.ipynb
    - emd : audio/emd.ipynb
    - feature_extraction : audio/feature extraction.ipynb
    - evaluation : evaluation.ipynb

------------------------------------------------------

## chat analisys
