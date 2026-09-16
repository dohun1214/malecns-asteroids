# malecns-asteroids

**초파리 뇌 지도(커넥톰) 166,700 뉴런 전체를 실시간으로 돌려서 Atari Asteroids를 플레이한다.**
그리고 보는 사람이 버튼으로 특정 뉴런을 끄면, 그 자리에서 망가지는 걸 본다.

> Runs all 166,700 neurons of the *Drosophila* male CNS connectome in real time and
> lets them play Atari Asteroids. Silence two cells and watch the escape direction break.

---

## 왜 만드는가

커넥톰으로 뭔가를 돌린 프로젝트는 많다. 대부분 "그냥 규칙으로 하는 거랑 뭐가 달라?"에
답하지 못한다. 16만 개를 돌려서 나온 게 "적이 왼쪽이면 오른쪽으로"라면 그건 `if` 세 줄이다.

**답은 제목이 아니라 토글 버튼이다. 규칙 기반 프로그램에는 끌 뉴런이 없다.**

---

## 지금까지 측정된 것

### 실시간 — 7.3배

| 방식 | 프레임(66.7ms)당 계산 | 실시간 대비 |
|---|---|---|
| post-major sparse CSR, dt=0.1 (기존 방식) | 498 ms | 0.13× |
| dt=0.2 + int32 + CUDA Graph 언롤 | 118.5 ms | 0.56× |
| **pre-major 이벤트 구동 Triton + CUDA Graph** | **9.15 ms** | **7.3×** |

RTX 4060 Ti 8GB, VRAM 50MB. 병목은 커널 런치가 아니라 **메모리 대역폭**이었다
(236 GB/s = 사양의 82%). post-major CSR은 누가 발화했든 매 스텝 전체 엣지를 읽는다.
pre-major scatter는 발화한 뉴런의 출력 엣지만 읽어 트래픽이 84.2MB → 약 1MB로 떨어진다.

### 회로가 실제로 방향을 읽는다

MaleCNS에는 LC4(looming 검출 뉴런)의 시야 위치가 없다. 그런데 LC4로 들어오는
Tm2/Tm4 컬럼에는 육각 좌표가 있다 → **입력 컬럼의 시냅스 가중 무게중심으로 위치를 복원**했다.
이 좌표는 DNp02/DNp11 연결과 **무관**하므로 순환논증이 아니다.

| 검증 | 결과 |
|---|---|
| 복원된 위치 ↔ 시냅스 경사 | \|r\| = 0.850 (좌) / 0.895 (우) |
| held-out (좌반구 축을 우반구에 적용) | r = −0.888 |
| 순열검정 (최적축 탐색 자유도 포함, 2000회) | **p = 0.0005** |

Dombrovski et al. 2023의 핵심 주장을 이 데이터에서 독립적으로 재현한 것이다.

### 게이트 1 — 세 개 전부 통과

| | 질문 | 결과 |
|---|---|---|
| (a) | 위협 방위각을 쓸면 방향 채널이 단조 변하는가 | **r = −0.986 / −0.989**, 부호까지 전환 |
| (c) | DNp11만 끄면 후방 반응이 무너지는가 | **2세포로 방향별 선택 붕괴**, 음성 대조 1.00배 |
| (b) | 앞뒤 동시 위협에서 상쇄되는가 | **방향 2%로 붕괴, 강도 1.19배 유지** |

편측성 지수 **±1.000** — 좌 LC4 자극은 좌 DNp02만, 우는 우만 구동한다.

**예측 재현**: Giant Fiber의 양측 동시 looming sublinear 합산 **0.72배**
(Jang & von Reyn 2023과 일치). LPLC2 자극 시 DNp02 정확히 **0**
(연결성 표만 보고 한 예측과 일치).

### 실전 플레이

| 정책 | 목숨당 생존(프레임) |
|---|---|
| 규칙 기반 | 1,251 |
| 가만히 있기 | 822 |
| **전체 뇌** | **672 ~ 726** |
| 무작위 | 23 |

무작위 대비 **30배**. 아직 "가만히 있기"를 못 이긴다 — 판독 개선 진행 중.

---

## 검증 장치

자작 커널은 조용히 틀릴 수 있다. 세 겹으로 막는다. 전부 자동화돼 있다.

| 검사 | 내용 |
|---|---|
| `test_scatter.py` | 같은 스파이크 벡터에서 pre-major scatter와 post-major cuSPARSE gather가 **비트 단위 일치**. 발화율 0~10%, 허브 뉴런(출차수 7,749) 포함 전부 `diff = 0.0` |
| `verify.py` | 침묵 사다리 0~5 + **600 step 동안 레퍼런스와 스파이크 단위 완전 일치** (51,644 = 51,644, 불일치 0스텝) |
| `test_determinism.py` | 직접 주입·Poisson × 10/40/200/999 step × 3회 반복 전부 `\|dv\| = \|dg\| = 0` |

결정론이 필요한 이유: "같은 시작점에서 정상 뇌와 병변 뇌가 갈라지는" 비교 영상이
프로젝트의 핵심 증거다. 가중치가 정수라 int32 원자 누산으로 덧셈 순서와 무관하게
비트 단위로 재현된다. fp32 원자 덧셈이었으면 불가능했다.

---

## 구성

```
src/
  prep_graph.py       커넥톰 -> pre-major packed int32 엣지 (가중치 상위 14bit, post 하위 18bit)
  lif_rt.py           전체 뇌 LIF. Triton 막전위/scatter 커널 + 333 step 언롤 CUDA Graph
  circuit.py          도피 회로 인덱스 추출 + 문헌 표 재현 검증
  retinotopy.py       LC4 시야 위치 복원 (held-out + 순열검정)
  vision.py           화면 -> 방위각 -> LC4 자극
  decode.py           하행뉴런 -> 3채널 -> 액션
  play.py / gate2.py  폐루프와 대조군 비교
  pacing.py           60Hz 페이싱 루프 (뇌가 늦어도 게임은 안 기다린다)
  verify.py / test_*.py  검증
```

## 요구사항

- Python 3.11, PyTorch 2.11 + CUDA 12.8, `triton-windows` 3.8 (Windows에는 공식 triton 휠이 없다)
- `gymnasium[atari]` 1.3, `ale-py` 0.12.1, `ocatari` 2.2.1 (`setuptools==69.5.1`, `wheel` 선행 필요)
- NVIDIA GPU 8GB 이상 (실측 VRAM 50MB)

데이터: [MaleCNS v1.0](https://male-cns.janelia.org/) (CC-BY, Janelia / Google Research).
저장소에 포함되지 않는다 — `prep_graph.py`가 쓸 그래프 산출물을 별도로 준비해야 한다.

## 참고 문헌

- Dombrovski et al. 2023, *Nature* — [Synaptic gradients transform object location to action](https://www.nature.com/articles/s41586-022-05562-8)
- Jang & von Reyn 2023, *JEB* — [Azimuthal invariance to looming stimuli in the giant fiber escape circuit](https://journals.biologists.com/jeb/article/226/8/jeb244790/307120)
- Shiu et al. 2024, *Nature* — [전뇌 leaky integrate-and-fire 모델](https://www.nature.com/articles/s41586-024-07763-9)
- Nern et al. 2025, *Nature* — [Connectome-driven neural inventory of a complete visual system](https://www.nature.com/articles/s41586-025-08746-0)

## 라이선스

코드 MIT. 커넥톰 데이터는 CC-BY (Janelia / Google Research).
