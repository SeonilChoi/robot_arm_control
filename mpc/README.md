# MPC 제어 (Model Predictive Control)

선형화된 이산시간 모델로 구간 \(N\) 동안 비용을 최소화하는 입력 열을 구하고, 그중 첫 입력만 적용하는 재ceding horizon 방식을 사용한다. 제약 없는 QP로 풀어 **numpy**만 사용한다.

## 1. 수식 정리

### 이산시간 모델
\[
\mathbf{z}_{k+1} = A_d \mathbf{z}_k + B_d \mathbf{u}_k
\]
(\(\mathbf{z}_k = [\mathbf{q}^T, \dot{\mathbf{q}}^T]^T\), \(\mathbf{u}_k = \boldsymbol{\tau}_k - \mathbf{G}(\mathbf{q}_r)\). \(A_d = I + A\,\Delta t\), \(B_d = B\,\Delta t\) 등으로 이산화.)

### 비용 함수
\[
J = \sum_{k=0}^{N-1} \left( \|\mathbf{z}_k - \mathbf{z}_r\|_Q^2 + \|\mathbf{u}_k\|_R^2 \right) + \|\mathbf{z}_N - \mathbf{z}_r\|_{Q_N}^2
\]
(\(\mathbf{z}_r\): 목표 상태.)

### 예측 궤적
\[
\mathbf{Z} = \Phi \mathbf{z}_0 + \Psi \mathbf{U}, \qquad
\mathbf{Z} = [\mathbf{z}_1^T, \ldots, \mathbf{z}_N^T]^T, \quad \mathbf{U} = [\mathbf{u}_0^T, \ldots, \mathbf{u}_{N-1}^T]^T
\]
\(\Phi\), \(\Psi\)는 \(A_d, B_d\)로부터 구성.

### QP
\[
\min_{\mathbf{U}} \ \frac{1}{2} \mathbf{U}^T H \mathbf{U} + \mathbf{c}^T \mathbf{U}, \qquad
H = 2(\Psi^T \bar{Q} \Psi + \bar{R}), \quad \mathbf{c} = 2\Psi^T \bar{Q}(\Phi \mathbf{z}_0 - \mathbf{Z}_{ref})
\]
(\(\bar{Q} = \mathrm{blkdiag}(Q,\ldots,Q,Q_N)\), \(\bar{R} = \mathrm{blkdiag}(R,\ldots,R)\).)  
해: \(\mathbf{U}^* = -H^{-1}\mathbf{c}\), 적용: \(\boldsymbol{\tau} = \mathbf{G}(\mathbf{q}_r) + \mathbf{u}_0^*\).

## 2. 수식–코드 매칭

| 수식 | 코드 |
|------|------|
| \(A_d, B_d\) 이산화 | `run.py`: `A_d = np.eye(6) + A * DT_MPC`, `B_d = B * DT_MPC` |
| \(\Psi\) (예측 입력 행렬) | `run.py`: `build_psi(Ad, Bd, N)` — \(\Psi_{ij} = A_d^{i-j}B_d\) (\(i\ge j\)) |
| \(\Phi\) | `run.py`: `build_phi(Ad, N)` — \(\Phi_i = A_d^{i+1}\) |
| \(\bar{Q}, \bar{R}\) | `run.py`: `Q_bar` (블록 대각), `R_bar = np.kron(np.eye(N), R_mpc)` |
| \(H = 2(\Psi^T\bar{Q}\Psi + \bar{R})\), \(\mathbf{c}\) | `run.py`: `H = 2*(Psi.T @ Q_bar @ Psi + R_bar)`, `c = 2*Psi.T @ Q_bar @ (Phi @ z - Z_ref_stack)` |
| \(\mathbf{U}^* = -H^{-1}\mathbf{c}\) | `run.py`: `U = np.linalg.solve(H, -c)` |
| \(\boldsymbol{\tau} = \mathbf{G}(\mathbf{q}_r) + \mathbf{u}_0^*\) | `run.py`: `tau = G(Q_REF) + u0` |

## 3. 실행 방법

```bash
cd mpc
python run.py
```

## 4. 결과

`mpc_result.png`: 관절각 \(q_1,q_2,q_3\) vs 시간, 점선은 목표 \(\mathbf{q}_r\).
