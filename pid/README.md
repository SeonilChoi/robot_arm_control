# 동역학 기반 PID 제어 (Computed Torque)

3-DOF 평면 로봇 팔에 대해 **Computed Torque** 방식의 PID 제어를 적용한다. 목표 관절각 \(\mathbf{q}_d\)에 대한 오차를 PID로 보정한 “원하는 가속도” \(\mathbf{u}\)를 구하고, 동역학 식을 이용해 필요한 토크 \(\boldsymbol{\tau}\)를 계산한다.

## 1. 수식 정리

### 로봇 동역학
\[
\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) + \mathbf{G}(\mathbf{q}) = \boldsymbol{\tau}
\]
- \(\mathbf{q} \in \mathbb{R}^3\): 관절각
- \(\mathbf{M}(\mathbf{q})\): 관성 행렬 (3×3, 대칭 양정치)
- \(\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\): 코리올리스/원심력 벡터
- \(\mathbf{G}(\mathbf{q})\): 중력 벡터
- \(\boldsymbol{\tau}\): 관절 토크

### 제어 법칙 (동역학 기반 PID)
목표 \(\mathbf{q}_d\), \(\dot{\mathbf{q}}_d\), \(\ddot{\mathbf{q}}_d\)에 대해 오차 \(\mathbf{e} = \mathbf{q}_d - \mathbf{q}\), \(\dot{\mathbf{e}} = \dot{\mathbf{q}}_d - \dot{\mathbf{q}}\)로
\[
\mathbf{u} = \ddot{\mathbf{q}}_d + K_p \mathbf{e} + K_d \dot{\mathbf{e}} + K_i \int \mathbf{e}\,dt
\]
\[
\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\, \mathbf{u} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) + \mathbf{G}(\mathbf{q})
\]

## 2. 수식–코드 매칭

| 수식 | 코드 위치 |
|------|-----------|
| \(\mathbf{M}(\mathbf{q})\) | `dynamics.py`: `M(q)` — 관성 행렬 반환 |
| \(\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\) | `dynamics.py`: `C_vec(q, qd)` — 코리올리스 벡터 |
| \(\mathbf{G}(\mathbf{q})\) | `dynamics.py`: `G(q)` — 중력 벡터 |
| \(\mathbf{e} = \mathbf{q}_d - \mathbf{q}\) | `run.py`: `e = Q_D - q` |
| \(\dot{\mathbf{e}} = \dot{\mathbf{q}}_d - \dot{\mathbf{q}}\) | `run.py`: `edot = QD_D - qd` |
| \(\int \mathbf{e}\,dt\) | `run.py`: `e_int = e_int + e * DT` (적분) |
| \(\mathbf{u} = \ddot{\mathbf{q}}_d + K_p \mathbf{e} + K_d \dot{\mathbf{e}} + K_i \int\mathbf{e}\) | `run.py`: `u = QDD_D + Kp @ e + Kd @ edot + Ki @ e_int` |
| \(\boldsymbol{\tau} = \mathbf{M}\mathbf{u} + \mathbf{C} + \mathbf{G}\) | `run.py`: `tau = M(q) @ u + C_vec(q, qd) + G(q)` |
| \(\ddot{\mathbf{q}} = \mathbf{M}^{-1}(\boldsymbol{\tau} - \mathbf{C} - \mathbf{G})\) | `dynamics.py`: `qdd_from_tau(q, qd, tau)` → RK4로 \(\mathbf{q},\dot{\mathbf{q}}\) 적분 |

## 3. 실행 방법

```bash
cd pid
python run.py
```

필요 패키지: `numpy`, `matplotlib`

## 4. 결과

`run.py` 실행 시 `pid_result.png`가 생성된다.  
- 세 개 서브플롯: 관절각 \(q_1, q_2, q_3\) vs 시간. 점선은 목표 \(\mathbf{q}_d\).
