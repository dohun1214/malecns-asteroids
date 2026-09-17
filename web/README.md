# 대시보드

```
python src/server.py          # 게임+뇌 루프 + WebSocket(8765) + 정적 서버(8766)
브라우저에서 http://localhost:8766/index.html
```

`vendor/` 의 three.js 는 **받아서 커밋해 둔다** (데모 당일 네트워크에 의존하지 않게).
받을 때 `three.module.js` 하나만 받으면 안 된다 — 그 파일이 `./three.core.js` 를
import 하므로 **두 개를 같이** 받아야 한다. OrbitControls 는 bare specifier `three` 를
import 하므로 **importmap 이 반드시 필요**하다.

| 파일 | |
|---|---|
| `index.html` | 화면 전부 (three.js Points + 4패널 + 토글) |
| `soma_xyz.bin` | 139,662 × float32 xyz (정규화됨). `src/prep_soma.py` 가 만든다 |
| `soma_grp.bin` | 139,662 × uint8 그룹 (0 기타 / 1 LC4 / 2 LPLC2 / 3 DN / 4 VNC판독) |
| `soma_meta.json` | 점 개수·중심·반경 |
| `vendor/` | three.module.js + three.core.js + OrbitControls.js |

## 프로토콜

| | |
|---|---|
| `FLY1` (binary) | `u32 magic, u32 step, u32 n` + `u16[n]` 델타 인코딩된 **점 인덱스** |
| `FLY2` (binary) | `u32 magic, u32 step, u32 (w<<16\|h)` + raw RGB |
| text (JSON) | 수치·디코더 채널·운석 박스·병변 상태 |
| 제어 (client→server) | `{cmd:"lesion",group,on}` / `{cmd:"restore"}` / `{cmd:"newgame"}` |

스파이크는 **전역 뉴런 번호가 아니라 점군 인덱스**로 보낸다 (좌표 없는 뉴런 27,038개 제외).
