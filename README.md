![image](https://github.com/user-attachments/assets/f328716c-8655-4068-b12d-ae5db115d497)
- 2024.11.13 10:00 ~ 2024.11.28 19:00
- X-ray 이미지에서 사람의 뼈를 Segmentation

## 팀원 소개

##  :sunglasses:팀원 소개

| [![](https://avatars.githubusercontent.com/jung0228)](https://github.com/jung0228) | [![](https://avatars.githubusercontent.com/chan-note)](https://github.com/chan-note) | [![](https://avatars.githubusercontent.com/batwan01)](https://github.com/batwan01) | [![](https://avatars.githubusercontent.com/jhuni17)](https://github.com/jhuni17) |  [![](https://avatars.githubusercontent.com/u/93571932?v=4)](https://github.com/uddaniiii) | [![](https://avatars.githubusercontent.com/u/48996852?v=4)](https://github.com/min000914) | 
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------- |
| [정현우](https://github.com/jung0228)   |   [임찬혁](https://github.com/chan-note)     | [박지완](https://github.com/batwan01)          | [최재훈](https://github.com/jhuni17) | [이단유](https://github.com/uddaniiii)  | [민창기](https://github.com/min000914) |

## 대회 소개
![image](https://github.com/user-attachments/assets/4790d3af-b534-4751-b65c-a51da184ec65)


뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적입니다.

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있습니다.

질병 진단의 목적으로 뼈의 형태나 위치가 변형되거나 부러지거나 골절 등이 있을 경우, 그 부위에서 발생하는 문제를 정확하게 파악하여 적절한 치료를 시행할 수 있습니다.

수술 계획을 세우는데 도움이 됩니다. 의사들은 뼈 구조를 분석하여 어떤 종류의 수술이 필요한지, 어떤 종류의 재료가 사용될 수 있는지 등을 결정할 수 있습니다.

의료장비 제작에 필요한 정보를 제공합니다. 예를 들어, 인공 관절이나 치아 임플란트를 제작할 때 뼈 구조를 분석하여 적절한 크기와 모양을 결정할 수 있습니다.

의료 교육에서도 활용될 수 있습니다. 의사들은 병태 및 부상에 대한 이해를 높이고 수술 계획을 개발하는 데 필요한 기술을 연습할 수 있습니다.

## 사용된 데이터셋 정보

- **데이터셋 이름**: 
- **출처**: 

### 데이터셋 설명

Input : hand bone x-ray 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. segmentation annotation은 json file로 제공됩니다.

Output : 모델은 각 클래스(29개)에 대한 확률 맵을 갖는 멀티채널 예측을 수행하고, 이를 기반으로 각 픽셀을 해당 클래스에 할당합니다.

최종적으로 예측된 결과를 Run-Length Encoding(RLE) 형식으로 변환하여 csv 파일로 제출합니다.


| 커밋 유형 | 의미 |
| :-: | -|
|feat|	새로운 기능 추가|
|fix|	버그 수정|
|docs	|문서 수정|
|style|	코드 formatting, 세미콜론 누락, 코드 자체의 변경이 없는 경우|
|refactor	|코드 리팩토링|
|test|	테스트 코드, 리팩토링 테스트 코드 추가|
|chore|	패키지 매니저 수정, 그 외 기타 수정 ex) .gitignore|
|design|	CSS 등 사용자 UI 디자인 변경|
|comment	|필요한 주석 추가 및 변경|
|rename|	파일 또는 폴더 명을 수정하거나 옮기는 작업만인 경우|
|remove|	파일을 삭제하는 작업만 수행한 경우|
|!BREAKING |CHANGE	커다란 API 변경의 경우|
|!HOTFIX	|급하게 치명적인 버그를 고쳐야 하는 경우|
