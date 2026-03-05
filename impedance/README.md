# 임피던스 제어 (Impedance Control)

작업공간(엔드이펙터 위치)에서 목표 위치 \(\mathbf{x}_d\)에 대해 가상 스프링–댐퍼–질량 관계를 부여하고, 그에 맞는 관절 토크 \(\boldsymbol{\tau}\)를 계산한다.

## 1. 수식 정리

### 작업공간 동역학 관계
- \(\mathbf{x} = f_{kin}(\mathbf{q})\): 정기구학 (엔드이펙터 위치)
- \(\dot{\mathbf{x}} = J(\mathbf{q})\dot{\mathbf{q}}\), \(\ddot{\mathbf{x}} = J\ddot{\mathbf{q}} + \dot{J}\dot{\mathbf{q}}\)

### 임피던스 법칙 (가상 힘)
\[
\mathbf{F} = M_d(\ddot{\mathbf{x}}_d - \ddot{\mathbf{x}}) + B_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + K_d(\mathbf{x}_d - \mathbf{x})
\]
- \(M_d, B_d, K_d\): 목표 질량·댐핑·강성 행렬 (대각 행렬로 설정)

### 구현 방식 (가속도 기반)
원하는 가속도를
\[
\ddot{\mathbf{x}}_{des} = \ddot{\mathbf{x}}_d + M_d^{-1}\bigl[ B_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + K_d(\mathbf{x}_d - \mathbf{x}) \bigr]
\]
로 두고, \(\ddot{\mathbf{x}}_{des} = J\ddot{\mathbf{q}}_{des} + \dot{J}\dot{\mathbf{q}}\)에서
\[
\ddot{\mathbf{q}}_{des} = J^{\dagger}(\ddot{\mathbf{x}}_{des} - \dot{J}\dot{\mathbf{q}})
\]
(\(J^{\dagger}\)는 유사역행렬). 그 다음
\[
\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\ddot{\mathbf{q}}_{des} + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) + \mathbf{G}(\mathbf{q})
\]

## 2. 수식–코드 매칭

| 수식 | 코드 |
|------|------|
| \(\mathbf{x} = f_{kin}(\mathbf{q})\) | `dynamics.py`: `forward_kinematics(q)` |
| \(\dot{\mathbf{x}} = J\dot{\mathbf{q}}\) | `run.py`: `xd = jacobian(q) @ qd` |
| \(J(\mathbf{q})\) | `dynamics.py`: `jacobian(q)` — 2×3 자코비안 |
| \(\dot{J}\dot{\mathbf{q}}\) | `dynamics.py`: `jacobian_dot(q, qd)` → `J_dot @ qd` |
| \(\ddot{\mathbf{x}}_{des} = \ddot{\mathbf{x}}_d + M_d^{-1}[\cdots]\) | `run.py`: `xdd_des = XDD_D + np.linalg.solve(M_d, B_d @ (XD_D - xd) + K_d @ (X_D - x))` |
| \(\ddot{\mathbf{q}}_{des} = J^{\dagger}(\ddot{\mathbf{x}}_{des} - \dot{J}\dot{\mathbf{q}})\) | `run.py`: `J_pinv = np.linalg.pinv(J)`, `qdd_des = J_pinv @ (xdd_des - J_dot @ qd)` |
| \(\boldsymbol{\tau} = \mathbf{M}\ddot{\mathbf{q}}_{des} + \mathbf{C} + \mathbf{G}\) | `run.py`: `tau = M(q) @ qdd_des + C_vec(q, qd) + G(q)` |

## 3. 실행 방법

```bash
cd impedance
python run.py
```

## 4. 결과

`impedance_result.png`: 위—엔드이펙터 \(x,y\) vs 시간(목표 점선). 아래—관절각 \(q_1,q_2,q_3\).
