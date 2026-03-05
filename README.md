# 6-DOF Robot Arm Control

6자유도 로봇 팔을 MuJoCo 가상 환경에서 시뮬레이션하고, 여러 제어기를 선택해 실험·학습할 수 있는 파이썬 프로젝트입니다.

## 구조

- **robot/** — 로봇 정의: MuJoCo 모델 로드, 상태(q, qd), 동역학 헬퍼(M(q), c(q,qd))
- **controllers/** — 제어기 (각 파일 독립): PID, 임피던스, LQR, MPC
- **sim/** — MuJoCo 시뮬레이션 환경 (step, get_state, reset)
- **run.py** — 실험 진입점: `--controller`로 제어기 선택 후 시뮬레이션 실행

## 설치

```bash
pip install -r requirements.txt
```

필요: `mujoco`, `numpy`, `scipy`, `matplotlib`

## 실행

```bash
# PID 제어 (기본)
python3 run.py --controller pid

# 임피던스 제어
python3 run.py --controller impedance

# LQR 제어
python3 run.py --controller lqr

# MPC 제어 (계산량 많음)
python3 run.py --controller mpc --duration 2.0
```

### 옵션

- `--controller {pid|impedance|lqr|mpc}` — 사용할 제어기 (기본: pid)
- `--model arm_6dof.xml` — MuJoCo 모델 파일명 (robot/assets 내)
- `--duration 5.0` — 시뮬레이션 시간(초)
- `--render` — MuJoCo 뷰어로 시각화

실행 후 `run_result.png`에 관절 궤적과 오차가 저장됩니다.

## 제어기 요약

| 제어기 | 설명 |
|--------|------|
| **PID** | 관절 공간 PID. 오차 = q_ref - q, 적분·미분 항으로 토크 계산. 로봇 동역학 미사용. |
| **Impedance** | 동역학 기반. M(q) u + c(q,qd) = tau, u = qdd_ref + M_d^{-1}(K_d(q_ref-q) + B_d(qd_ref-qd)). 목표 임피던스 M_d, B_d, K_d 튜닝. |
| **LQR** | 평형점(q_eq)에서 선형화 후 LQR 게인 계산. u = -K (x - x_ref). 상태 x = [q, qd]. |
| **MPC** | 짧은 구간(horizon) 동안 비선형 동역학으로 궤적 예측, 비용 최소화 후 첫 토크만 적용. scipy.optimize 사용. |

각 제어기는 서로 독립이며, `controllers/` 아래 해당 파일만 수정·추가하면 됩니다.

---

## 제어기 상세 설명 (수식과 코드 매칭)

아래에서는 각 제어기의 수식을 단계별로 풀고, **어떤 수식이 코드의 어디에 대응하는지** 짚어줍니다. 수식은 관절 각도 \(q\), 각속도 \(\dot{q}\), 토크 \(\tau\) 등으로 쓰고, 코드는 `controllers/` 안의 해당 파일을 기준으로 합니다.

---

### 1. PID 제어기

**아이디어**  
목표 관절 각도 \(q_{\text{ref}}\)와 현재 각도 \(q\)의 **오차**를 줄이기 위해, 오차에 비례(P), 오차의 적분(I), 오차의 미분(D)에 각각 게인을 곱해 토크를 만듭니다. 로봇의 질량·관성은 쓰지 않고, “오차가 크면 세게, 적분이 쌓이면 보정, 속도가 다르면 감쇠”만 합니다.

**수식 정리**

1. **위치 오차** (목표 − 현재):
   \[
   e = q_{\text{ref}} - q
   \]

2. **오차의 적분** (과거 오차가 쌓인 값, 매 스텝 갱신):
   \[
   e_{\text{int}}(t+\Delta t) = e_{\text{int}}(t) + e \cdot \Delta t
   \]
   적분이 너무 커지면 발산할 수 있으므로, 상한/하한으로 **클리핑**합니다.

3. **속도 오차** (목표 각속도 − 현재 각속도):
   \[
   \dot{e} = \dot{q}_{\text{ref}} - \dot{q}
   \]

4. **PID 토크** (P·I·D 각각 게인 \(K_p, K_i, K_d\)):
   \[
   \tau = K_p \, e + K_i \, e_{\text{int}} + K_d \, \dot{e}
   \]

**코드와의 대응** (`controllers/pid_controller.py`)

| 수식 | 코드 |
|------|------|
| \(e = q_{\text{ref}} - q\) | `err = q_ref - q` (45행) |
| \(q_{\text{ref}}, \dot{q}_{\text{ref}}\) | `q_ref = ref[:6]`, `qd_ref = ref[6:12]` (37–42행): `reference`에서 앞 6개가 목표 각도, 다음 6개가 목표 각속도 |
| \(q, \dot{q}\) | `q = state.q[:self.nq]`, `qd = state.qd[:self.nq]` (43–44행) |
| \(e_{\text{int}} \mathrel{+}= e \cdot \Delta t\) | `self._integral += err * self.dt` (46행) |
| 적분 클리핑 | `self._integral = np.clip(..., -integral_clip, integral_clip)` (47행) |
| \(\tau = K_p e + K_d \dot{e} + K_i e_{\text{int}}\) | `tau = self.Kp * err + self.Kd * (qd_ref - qd) + self.Ki * self._integral` (48행) |

**튜닝**  
- `Kp`: 오차에 비례해 토크를 주므로 크면 반응이 빠르지만 너무 크면 진동.  
- `Ki`: 적분 항으로 정상 오차(steady-state error)를 줄임. 너무 크면 오버슈트·진동.  
- `Kd`: 속도 오차로 “제동”을 걸어 진동을 줄임.  
- `integral_clip`: 적분 상한/하한으로 적분 발산을 막습니다.

---

### 2. 임피던스 제어기

**아이디어**  
로봇의 **동역학**을 사용합니다. 관절 공간에서 운동 방정식은
\[
M(q) \ddot{q} + c(q,\dot{q}) = \tau
\]
입니다. 여기서 \(M(q)\)는 관성 행렬, \(c(q,\dot{q})\)는 코리올리스·중력 등입니다.  
“원하는 가속도” \(u\)를 정해 두고, \(\tau = M(q) u + c(q,\dot{q})\)로 토크를 주면, 실제로 \(\ddot{q} = u\)가 됩니다. 그 \(u\)를 **목표 임피던스**(마치 질량–댐퍼–스프링 시스템처럼)로 설계하는 것이 임피던스 제어입니다.

**수식 정리**

1. **로봇 동역학** (연속 시간):
   \[
   M(q) \ddot{q} + c(q,\dot{q}) = \tau
   \]
   따라서 \(\tau = M(q) u + c(q,\dot{q})\) 이면 \(\ddot{q} = u\) 가 됩니다.

2. **목표 동작** (원하는 가속도 \(u\)):  
   “가상의 질량–댐퍼–스프링”에서 나오는 2차 동역학으로, 위치·속도 오차를 줄이도록 잡습니다.
   \[
   M_d \, u = K_d (q_{\text{ref}} - q) + B_d (\dot{q}_{\text{ref}} - \dot{q})
   \]
   \(M_d\): 목표 관성, \(B_d\): 목표 댐핑, \(K_d\): 목표 스티프니스.  
   (필요하면 \(\ddot{q}_{\text{ref}}\) 항을 더할 수 있습니다.)

3. **원하는 가속도** \(u\):
   \[
   u = M_d^{-1} \Bigl( K_d (q_{\text{ref}} - q) + B_d (\dot{q}_{\text{ref}} - \dot{q}) \Bigr)
   \]
   코드에서는 \(\ddot{q}_{\text{ref}}=0\) 으로 두었습니다.

4. **최종 토크**:
   \[
   \tau = M(q) \, u + c(q,\dot{q})
   \]

**코드와의 대응** (`controllers/impedance_controller.py`)

| 수식 | 코드 |
|------|------|
| \(M(q)\) | `M = self._robot.get_M()` (45행). 그 전에 `mj_forward`로 현재 \(q,\dot{q}\) 반영 (44행) |
| \(c(q,\dot{q})\) | `c = self._robot.get_c()` (46행) |
| \(q_{\text{ref}}-q\), \(\dot{q}_{\text{ref}}-\dot{q}\) | `err_q = q_ref - q`, `err_qd = qd_ref - qd` (47–48행) |
| \(M_d^{-1}(K_d e + B_d \dot{e})\) | `u = qdd_ref + np.linalg.solve(self.Md, self.Kd @ err_q + self.Bd @ err_qd)` (51행). `qdd_ref`는 0 |
| \(\tau = M u + c\) | `tau = M @ u + c` (52행) |

**튜닝**  
- `Md`: 목표 “질량” 느낌. 보통 대각 행렬로 두고, 크면 반응이 무겁게, 작으면 가볍게 느껴짐.  
- `Bd`: 댐핑. 크면 진동이 줄고, 작으면 더 흔들릴 수 있음.  
- `Kd`: 스티프니스. 크면 목표 위치로 더 세게 당김.

---

### 3. LQR 제어기

**아이디어**  
로봇은 비선형이지만, **한 점(평형점)** 근처에서 1차 근사하면 **선형** 시스템으로 쓸 수 있습니다. 그 선형 모델에 대해 “상태 오차와 입력에 대한 2차 비용”을 최소화하는 제어가 **LQR**이고, 해가 **선형 상태 피드백** \(u = -K (x - x_{\text{ref}})\) 형태로 나옵니다.  
여기서는 평형점 \(q_{\text{eq}}, \dot{q}=0\) 주변에서 선형화하고, 그 점을 기준으로 LQR 게인 \(K\)를 한 번 구한 뒤, 매 스텝 \(x = [q,\dot{q}]\), \(x_{\text{ref}} = [q_{\text{ref}}, \dot{q}_{\text{ref}}]\) 로 두고 같은 \(K\)를 씁니다.

**수식 정리**

1. **상태 정의**:
   \[
   x = \begin{bmatrix} q \\ \dot{q} \end{bmatrix}, \quad
   x_{\text{ref}} = \begin{bmatrix} q_{\text{ref}} \\ \dot{q}_{\text{ref}} \end{bmatrix}
   \]

2. **비선형 동역학** (관절 공간):
   \[
   \ddot{q} = M(q)^{-1} \bigl( \tau - c(q,\dot{q}) \bigr)
   \]
   즉 \(\dot{x} = [\dot{q},\ \ddot{q}]^{\top}\).

3. **평형점 근처 선형화**  
   평형점 \((q_{\text{eq}}, \dot{q}=0)\)에서 \(\tau = c(q_{\text{eq}},0)\)일 때 \(\ddot{q}=0\).  
   \(q\)만 미세하게 움직인다고 하면 \(c(q,0)\)를 \(q\)로 미분한 행렬을 쓰고, \(\dot{q}\)에 대한 항은 평형점에서는 0으로 두어:
   \[
   \delta\ddot{q} \approx -M^{-1} \frac{\partial c}{\partial q}\Big|_{\text{eq}} \delta q + M^{-1} \delta\tau
   \]
   따라서 선형 모델은
   \[
   \dot{x} = A x + B u, \quad
   A = \begin{bmatrix} 0 & I \\ -M^{-1} \frac{\partial c}{\partial q} & 0 \end{bmatrix}, \quad
   B = \begin{bmatrix} 0 \\ M^{-1} \end{bmatrix}, \quad u = \tau
   \]
   (상수항은 \(x_{\text{ref}}\) 쪽으로 넘기면 되고, 코드에서는 오차 \(x - x_{\text{ref}}\)에 대해 \(u = -K(x-x_{\text{ref}})\)를 쓰므로 위 형태만 있으면 됨.)

4. **LQR**  
   비용
   \[
   J = \int \bigl( (x-x_{\text{ref}})^{\top} Q (x-x_{\text{ref}}) + u^{\top} R u \bigr) dt
   \]
   를 최소화하는 연속시간 LQR의 해는 **CARE**(Continuous-time Algebraic Riccati Equation) 해 \(P\)로:
   \[
   A^{\top} P + P A - P B R^{-1} B^{\top} P + Q = 0, \qquad
   K = R^{-1} B^{\top} P
   \]
   그리고 최적 제어는
   \[
   u = -K (x - x_{\text{ref}})
   \]

**코드와의 대응** (`controllers/lqr_controller.py`)

| 수식 | 코드 |
|------|------|
| 평형점 \(q_{\text{eq}}\) 설정 | `self.q_eq` (25행), 기본값 0 |
| \(M(q_{\text{eq}})\), \(c(q_{\text{eq}},0)\) | `mj_forward` 후 `get_M()`, `get_c()` (38–40행) |
| \(\frac{\partial c}{\partial q}\) 수치 미분 | `dcdq[:, j] = (cp - cm) / (2*eps)` (43–55행): \(q_j\)만 ±eps 움직여서 \(c\) 차이로 편미분 |
| \(A_{21} = -M^{-1} \frac{\partial c}{\partial q}\) | `A21 = -Minv @ dcdq` (61행) |
| \(A, B\) 블록 행렬 | `A = np.block([...])`, `B = np.block([...])` (62–63행) |
| CARE 해 \(P\), 게인 \(K\) | `P = solve_continuous_are(A, B, Q, R)`, `K = np.linalg.solve(R, B.T @ P)` (65–66행) |
| \(x\), \(x_{\text{ref}}\) | `x = np.concatenate([state.q[:6], state.qd[:6]])`, `x_ref = np.concatenate([q_ref, qd_ref])` (78–79행) |
| \(u = -K(x - x_{\text{ref}})\) | `u = -self.K @ (x - x_ref)` (80행) |

**튜닝**  
- `Q`: 상태(위치·속도) 오차에 대한 가중치. 크면 오차를 더 줄이려 하고, 제어 입력이 커질 수 있음.  
- `R`: 입력(토크)에 대한 가중치. 크면 입력을 아끼고, 작으면 더 공격적으로 제어.  
- 평형점이 바뀌면 이론상 선형화 지점을 바꾸고 \(K\)를 다시 구하는 것이 맞고, 여기서는 한 번 구한 \(K\)를 궤적 추종에 그대로 씁니다.

---

### 4. MPC 제어기

**아이디어**  
매 제어 주기마다 “앞으로 N 스텝” 동안의 **토크 시퀀스**를 한 번에 정합니다. 그때 사용하는 모델은 **비선형 동역학** \(M(q)\ddot{q}+c(q,\dot{q})=\tau\)를 그대로 쓰고, 시간을 작은 구간 \(\Delta t\)로 나눠서 **이산화**(예: 오일러)한 뒤,  
- 비용: 각 스텝에서 \((x - x_{\text{ref}})^{\top} Q (x - x_{\text{ref}}) + \tau^{\top} R \tau\) 를 다 더한 값  
- 제약: 매 스텝 \(x_{k+1} = f(x_k, \tau_k)\) (동역학 한 스텝)  
를 만족하도록 \(\tau_0, \tau_1, \ldots, \tau_{N-1}\)를 최적화합니다.  
실제로는 **첫 토크 \(\tau_0\)만** 로봇에 넣고, 다음 주기에서 다시 현재 상태를 기준으로 N 스텝 최적화를 반복합니다(Receding Horizon).

**수식 정리**

1. **연속 동역학** (관절 공간):
   \[
   M(q) \ddot{q} + c(q,\dot{q}) = \tau \quad\Rightarrow\quad \ddot{q} = M(q)^{-1}\bigl( \tau - c(q,\dot{q}) \bigr)
   \]

2. **한 스텝 이산화** (오일러, 스텝 \(\Delta t\)):
   \[
   q_{k+1} = q_k + \Delta t \, \dot{q}_k, \qquad
   \dot{q}_{k+1} = \dot{q}_k + \Delta t \, \ddot{q}_k, \qquad
   \ddot{q}_k = M(q_k)^{-1}\bigl( \tau_k - c(q_k,\dot{q}_k) \bigr)
   \]

3. **비용** (horizon \(N\)):
   \[
   J = \sum_{k=0}^{N-1} \Bigl( (x_k - x_{\text{ref}})^{\top} Q (x_k - x_{\text{ref}}) + \tau_k^{\top} R \tau_k \Bigr)
   \]
   \(x_k = [q_k,\ \dot{q}_k]^{\top}\).

4. **최적화**  
   변수: \(\tau_0, \ldots, \tau_{N-1}\).  
   초기 상태 \(x_0\)에서 위 이산 동역학으로 \(x_1, \ldots, x_N\)을 구하면서 \(J\)를 계산하고, \(J\)를 최소화합니다.  
   해에서 **\(\tau_0^{\ast}\)만** 사용:
   \[
   \tau_{\text{apply}} = \tau_0^{\ast}
   \]

**코드와의 대응** (`controllers/mpc_controller.py`)

| 수식 | 코드 |
|------|------|
| \(\ddot{q} = M^{-1}(\tau - c)\) | `qdd = np.linalg.solve(M, tau - c)` (39행) |
| \(q_{k+1} = q_k + \Delta t \dot{q}_k\) | `q_next = q + self.dt * qd` (40행) |
| \(\dot{q}_{k+1} = \dot{q}_k + \Delta t \ddot{q}_k\) | `qd_next = qd + self.dt * qdd` (41행) |
| 한 스텝 전체 | `_dynamics_step(q, qd, tau)` (33–42행): `set_state` → `mj_forward` → M, c → 위 두 식 |
| \((x_k - x_{\text{ref}})^{\top} Q (x_k - x_{\text{ref}})\) | `(np.concatenate([q, qd]) - x_ref) @ self.Q @ (...)` (53행) |
| \(\tau_k^{\top} R \tau_k\) | `tau @ self.R @ tau` (54행) |
| \(J = \sum_k (\ldots)\) | `_rollout_cost` 안의 `for k in range(N):` 루프 (51–55행)에서 cost 누적 |
| \(\tau_0^{\ast}\) 적용 | `u_opt = result.x.reshape(N, nq); return u_opt[0]` (81–82행): 최적화 결과의 첫 토크만 반환 |

**튜닝**  
- `horizon`: N. 크면 더 먼 미래를 보지만 계산량이 늘어남.  
- `Q`, `R`: LQR과 비슷하게 상태 오차 vs 입력 크기 트레이드오프.  
- `bounds`: 토크 상한/하한(코드에서는 ±50).  
이렇게 하면 **수식 전개 → 코드 위치**가 일대일로 대응해, 초보자도 “이 줄이 이 수식이다”라고 따라가기 쉬워집니다.

## 로봇 모델

- `robot/assets/arm_6dof.xml`: MuJoCo MJCF 6관절 매니퓰레이터. 다른 6-DOF 모델로 교체 가능.

## 새 제어기 추가

1. `controllers/new_controller.py`에 `ControllerBase`를 상속한 클래스 구현 (`compute_control(state, reference, t) -> tau`).
2. `controllers/__init__.py`의 `CONTROLLERS`에 `"new": ("controllers.new_controller", "NewController")` 추가.
3. `run.py`의 `make_controller`에서 필요 시 `robot`/`dt` 인자 전달 규칙에 맞춰 생성.
