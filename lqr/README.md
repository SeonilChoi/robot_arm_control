# LQR 제어 (Linear Quadratic Regulator)

평형점 \(\mathbf{z}_e = [\mathbf{q}_e^T, \mathbf{0}^T]^T\)(\(\dot{\mathbf{q}}_e = \mathbf{0}\)) 주변에서 로봇 동역학을 선형화한 뒤, LQR로 이득 \(K\)를 구하고 \(\boldsymbol{\tau} = \boldsymbol{\tau}_e - K(\mathbf{z} - \mathbf{z}_e)\)를 적용한다.

## 1. 수식 정리

### 상태 및 평형
- \(\mathbf{z} = [\mathbf{q}^T, \dot{\mathbf{q}}^T]^T \in \mathbb{R}^6\), \(\boldsymbol{\tau}_e = \mathbf{G}(\mathbf{q}_e)\)
- \(\dot{\mathbf{z}} = \mathbf{A}\mathbf{z} + \mathbf{B}\mathbf{u}\) (\(\mathbf{u} = \boldsymbol{\tau} - \boldsymbol{\tau}_e\)로 보조 입력)

### Riccati 방정식 및 이득
\[
\mathbf{A}^T P + P\mathbf{A} - P\mathbf{B} R^{-1} \mathbf{B}^T P + Q = 0
\]
\[
K = R^{-1} \mathbf{B}^T P, \qquad \mathbf{u} = -K(\mathbf{z} - \mathbf{z}_e)
\]
\[
\boldsymbol{\tau} = \boldsymbol{\tau}_e + \mathbf{u} = \mathbf{G}(\mathbf{q}_e) - K(\mathbf{z} - \mathbf{z}_e)
\]

### 선형화
\(\mathbf{f}(\mathbf{z},\boldsymbol{\tau}) = [\dot{\mathbf{q}}^T, (\mathbf{M}^{-1}(\boldsymbol{\tau} - \mathbf{C} - \mathbf{G}))^T]^T\)에 대해
\[
\mathbf{A} = \frac{\partial \mathbf{f}}{\partial \mathbf{z}}\bigg|_{(\mathbf{z}_e,\boldsymbol{\tau}_e)}, \qquad
\mathbf{B} = \frac{\partial \mathbf{f}}{\partial \boldsymbol{\tau}}\bigg|_{(\mathbf{z}_e,\boldsymbol{\tau}_e)} = \begin{bmatrix} 0 \\ \mathbf{M}^{-1}(\mathbf{q}_e) \end{bmatrix}
\]

## 2. 수식–코드 매칭

| 수식 | 코드 |
|------|------|
| \(\mathbf{A}, \mathbf{B}\) 선형화 | `dynamics.py`: `linearize(q_e)` — 수치 미분으로 \(\partial\ddot{\mathbf{q}}/\partial\mathbf{q},\ \partial\ddot{\mathbf{q}}/\partial\dot{\mathbf{q}}\) 및 \(\mathbf{B}[3:6,:] = \mathbf{M}^{-1}(\mathbf{q}_e)\) |
| CARE \(\mathbf{A}^T P + P\mathbf{A} - P\mathbf{B}R^{-1}\mathbf{B}^T P + Q = 0\) | `run.py`: `solve_care(A, B, Q, R)` — 해밀토니안 고유벡터로 \(P\) 계산 |
| \(K = R^{-1}\mathbf{B}^T P\) | `run.py`: `K_lqr = Rinv @ B.T @ P` |
| \(\mathbf{u} = -K(\mathbf{z} - \mathbf{z}_e)\) | `run.py`: `u = -K_lqr @ (z - Z_E)` |
| \(\boldsymbol{\tau} = \boldsymbol{\tau}_e + \mathbf{u}\) | `run.py`: `tau = TAU_E + u` |

## 3. 실행 방법

```bash
cd lqr
python run.py
```

## 4. 결과

`lqr_result.png`: 위—관절각 \(q_1,q_2,q_3\)와 평형점(점선). 아래—\(\dot{q}_1,\dot{q}_2,\dot{q}_3\).
