# time-sync video tagging

time-sync video tagging using comments, sound and image data.


## contributor 

- HIhyun
- GMpark

### tagg extraction

- 블라블라

### speech to text

- 스트리밍 영상(트위치)에서 음성만 녹음(확장자 m4a)
- 'm4a -> wav' 로 확장자 변환 (mono 채널) + api 사용시 인코딩(LINEAR16)
- 긴 오디오파일의 경우 google cloud storage에 업로드하여 URI를 참조
- 결과 스크립트 '단어'단위로 time stamp 함께 출력. 
- '단어'단위를 다시 같은시간대로 묶어서 temp 만듦
- 코멘트 데이터와 캡션 데이터 융합

##### to do 