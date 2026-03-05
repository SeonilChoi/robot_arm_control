# 로봇 팔 제어 (3-DOF 평면 로봇)

3자유도 평면 로봇 팔에 대해 **동역학 기반 PID**, **임피던스**, **LQR**, **MPC**, **QP** 제어기를 각각 독립 폴더에서 구현한 프로젝트이다.  
각 폴더에는 해당 제어기의 수식 정리와 **수식–코드 매칭** 설명이 포함된 README가 있다.

## 의존성

- **numpy**
- **matplotlib**

(다른 라이브러리 사용 없음.)

## 로봇 모델

- **3-DOF 평면 로봇**: 3개 회전 관절, 2D 평면에서 동작.
- 동역학: \(\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) + \mathbf{G}(\mathbf{q}) = \boldsymbol{\tau}\)  
  모든 폴더의 `dynamics.py`에서 동일한 \(\mathbf{M},\mathbf{C},\mathbf{G}\) 식을 사용한다.

## 폴더 및 실행 방법

| 폴더 | 설명 | 실행 |
|------|------|------|
| [pid/](pid/) | 동역학 기반 PID (Computed Torque) | `cd pid && python run.py` |
| [impedance/](impedance/) | 작업공간 임피던스 제어 | `cd impedance && python run.py` |
| [lqr/](lqr/) | LQR (평형점 주변 선형화) | `cd lqr && python run.py` |
| [mpc/](mpc/) | MPC (재ceding horizon, 제약 없는 QP) | `cd mpc && python run.py` |
| [qp/](qp/) | QP 기반 목표 각도 추종 | `cd qp && python run.py` |

각 폴더는 **독립 실행**이 가능하다. 해당 폴더로 이동한 뒤 `python run.py`만 실행하면 된다.  
자세한 수식과 코드 매칭은 각 폴더의 **README.md**를 참고하면 된다.

## 설치

```bash
pip install -r requirements.txt
```

이후 위 표와 같이 원하는 폴더에서 `python run.py`를 실행하면 된다.
