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

##### to do 
- 결과 스크립트(speech to text)에 시간 기록하기
- 태그 추출 알고리즘에서도 시간 추출하도록 알고리즘 수적하기