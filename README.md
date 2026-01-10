## 🍀 프로젝트 개요

- **"고비용의 물리 센서 없이, 기존 CCTV만으로 가능한 스마트 교통 관제 솔루션"**
- 기존의 루프 검지기나 물리 센서는 설치 및 유지보수 비용이 높다는 단점이 있습니다. 이를 대체하기 위해, **기존 도로 CCTV 영상만으로 차량 흐름을 실시간으로 분석하고 카운팅할 수 있는 AI 비전 솔루션**을 기획하게 되었습니다.

```mermaid
graph LR
    classDef input stroke:#1565c0,stroke-width:4px,rx:10,ry:10,font-size:30px,font-weight:bold,padding:20px;
    classDef model stroke:#ef6c00,stroke-width:4px,rx:5,ry:5,font-size:30px,font-weight:bold,padding:20px;
    classDef algo stroke:#7b1fa2,stroke-width:4px,rx:5,ry:5,font-size:30px,font-weight:bold,padding:20px;
    classDef visual stroke:#2e7d32,stroke-width:4px,rx:5,ry:5,font-size:30px,font-weight:bold,padding:20px;
    
    A["CCTV Input"]:::input
    B["YOLOv8(Detect)"]:::model
    C["ByteTrack(Track)"]:::algo
    D["CCW Logic(Count)"]:::algo
    E["OpenCV(Display)"]:::visual

    A -->|Frame| B
    B -->|BBox & Conf| C
    C -->|Track ID| D
    D -->|Count Data| E
```
## 🔧 사용 기술

| **분류** | **기술 스택** | **주요 역할 및 활용 이유** |
| --- | --- | --- |
| Language | Python | 전체 시스템 로직 구현 및 라이브러리 통합  |
| AI Model | YOLOv8 | 실시간 객체 탐지  |
| Tracker | SORT, ByteTrack | 객체 추적 및 ID 관리 |
| Library | OpenCV | Native Drawing 기반 시각화 |
| Math | NumPy | CCW 기반 벡터 교차 연산 및 고속 데이터 처리 |

## 📍주요 기능

### **1. 딥러닝 기반 실시간 추적**

- **YOLOv8의 탐지 결과와** **ByteTrack** 알고리즘을 결합하여, 실시간으로 각 차량에 **고유 ID**를 부여해 이동 궤적을 추적하고 ID 스위칭을 최소화하도록 구현.

### **2. 벡터 기반 정밀 카운팅**

- 단순 좌표 비교의 한계를 극복하기 위해, **CCW(벡터 교차) 알고리즘**을 도입하여 프레임 스킵 상황에서도 누락 없는 통행량 집계 구현.

### 3. 차선별 독립 분석

- 도로 전체가 아닌 **좌/우 차선을 독립적으로 분리**하여 개별 차선의 교통량을 정밀하게 산출.

### 4. **엣지 디바이스 최적화**

- OpenCV Native Drawing을 통해 렌더링 오버헤드를 줄여, 저사양 환경에서도 실시간 FPS 방어.












