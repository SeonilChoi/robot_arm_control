# 6-DOF Robot Arm Control

6자유도 로봇 팔을 MuJoCo 가상 환경에서 시뮬레이션하고, 여러 제어기를 선택해 실험·학습할 수 있는 파이썬 프로젝트입니다.

---

## 목차

1. [프로젝트 구조](#구조)
2. [설치 및 실행](#설치)
3. [제어기 요약](#제어기-요약)
4. [제어기 상세 설명](#제어기-상세-설명-수식과-코드-매칭) (수식 전개 및 코드 매칭)
5. [로봇 모델](#로봇-모델)
6. [새 제어기 추가](#새-제어기-추가)

---

## 구조

| 디렉터리/파일 | 역할 |
|---------------|------|
| **robot/** | 로봇 정의: MuJoCo 모델 로드, 상태 $(q, \dot{q})$, 동역학 헬퍼 $M(q)$, $c(q,\dot{q})$ |
| **robot/assets/arm_6dof.xml** | MuJoCo 6관절 팔 모델 (MJCF) |
| **controllers/** | 제어기 (파일별 독립): PID, 임피던스, LQR, MPC |
| **sim/** | MuJoCo 시뮬레이션: `step(tau)`, `get_state()`, `reset()` |
| **run.py** | 실험 진입점: `--controller`로 제어기 선택 후 시뮬레이션 실행 |

---

## 설치

```bash
pip install -r requirements.txt
```

필요 패키지: `mujoco`, `numpy`, `scipy`, `matplotlib`

---

## 실행

```bash
# PID 제어 (기본). MuJoCo 뷰어가 기본으로 켜짐.
python3 run.py --controller pid

# 다른 제어기
python3 run.py --controller impedance
python3 run.py --controller lqr
python3 run.py --controller mpc --duration 2.0

# 뷰어 없이 실행 (서버/배치용)
python3 run.py --controller pid --no-render
```

### 실행 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--controller` | 제어기: `pid`, `impedance`, `lqr`, `mpc` | `pid` |
| `--model` | MuJoCo 모델 파일명 (robot/assets 내) | `arm_6dof.xml` |
| `--duration` | 시뮬레이션 시간(초) | `5.0` |
| `--render` | MuJoCo 뷰어 켜기 | 켜짐 |
| `--no-render` | MuJoCo 뷰어 끄기 | - |

실행이 끝나면 `run_result.png`에 관절 궤적과 추종 오차가 저장됩니다.

---

## 제어기 요약

| 제어기 | 한 줄 설명 |
|--------|------------|
| **PID** | 관절 오차에 비례(P)·적분(I)·미분(D) 항을 곱해 토크 계산. 동역학 미사용. |
| **Impedance** | 동역학 $M(q)\ddot{q}+c=\tau$를 이용해 “원하는 가속도” $u$를 만들고, $u$를 질량-댐퍼-스프링 형태로 설계. |
| **LQR** | 평형점에서 선형화한 뒤, 2차 비용을 최소화하는 선형 피드백 $u=-K(x-x_{\text{ref}})$ 적용. |
| **MPC** | 앞으로 N 스텝 토크를 비선형 동역학으로 예측·최적화한 뒤, 첫 토크만 적용 (Receding Horizon). |

각 제어기는 서로 독립이며, `controllers/` 아래 해당 파일만 수정·추가하면 됩니다.

---

## 제어기 상세 설명 (수식과 코드 매칭)

아래에서는 **공통 기호**를 먼저 정의하고, 각 제어기별로 **수식을 단계별로 풀어 쓴 뒤**, **어떤 수식이 코드의 어디에 대응하는지** 표로 정리합니다.

> **수식 렌더링**: GitHub 등에서 수식이 그대로 보이지 않으면, 에디터/뷰어에서 Markdown 수학 확장을 켜거나, `$ ... $`(인라인), `$$ ... $$`(디스플레이)를 지원하는 도구로 열어보세요.

---

### 기호 정의 (공통)

| 기호 | 의미 | 차원 |
|------|------|------|
| $q$ | 관절 각도 (위치) | $(6 \times 1)$ |
| $\dot{q}$ | 관절 각속도 | $(6 \times 1)$ |
| $\ddot{q}$ | 관절 각가속도 | $(6 \times 1)$ |
| $\tau$ | 관절 토크 (제어 입력) | $(6 \times 1)$ |
| $q_{\text{ref}}$, $\dot{q}_{\text{ref}}$ | 목표 관절 각도·각속도 | $(6 \times 1)$ |
| $M(q)$ | 관성 행렬 | $(6 \times 6)$ |
| $c(q, \dot{q})$ | 코리올리스·원심력·중력 벡터 | $(6 \times 1)$ |
| $\Delta t$ | 시뮬레이션/제어 시간 간격 | 스칼라 |

로봇 동역학은 관절 공간에서 다음으로 주어집니다.

$$
M(q) \ddot{q} + c(q, \dot{q}) = \tau
$$

즉, 가해준 토크 $\tau$와 현재 $q$, $\dot{q}$에 의해 가속도 $\ddot{q}$가 결정됩니다.

---

### 1. PID 제어기

#### 1.1 아이디어

목표 관절 각도 $q_{\text{ref}}$와 현재 각도 $q$의 **오차**를 줄이기 위해,  
오차에 **비례(P)**, 오차의 **적분(I)**, 오차의 **미분(D)** 에 각각 게인을 곱해 토크를 만듭니다.  
로봇의 질량·관성은 사용하지 않고, “오차가 크면 더 세게”, “적분이 쌓이면 보정”, “속도가 다르면 감쇠”만 합니다.

#### 1.2 수식 전개

**① 위치 오차**

목표와 현재의 차이를 정의합니다.

$$
e = q_{\text{ref}} - q
$$

**② 오차의 적분 (적분항)**

과거 오차가 쌓인 값을 매 제어 스텝마다 갱신합니다.

$$
e_{\text{int}}(t + \Delta t) = e_{\text{int}}(t) + e \cdot \Delta t
$$

적분이 무한히 커지면 제어가 불안정해질 수 있으므로, **상한/하한으로 클리핑**합니다.

**③ 속도 오차 (미분항)**

목표 각속도와 현재 각속도의 차이입니다.

$$
\dot{e} = \dot{q}_{\text{ref}} - \dot{q}
$$

**④ PID 토크**

비례·적분·미분에 게인 $K_p$, $K_i$, $K_d$를 곱해 더합니다.

$$
\tau = K_p \, e + K_i \, e_{\text{int}} + K_d \, \dot{e}
$$

#### 1.3 코드와의 대응

파일: `controllers/pid_controller.py`

| 수식 | 대응 코드 (요지) |
|------|------------------|
| $e = q_{\text{ref}} - q$ | `err = q_ref - q` |
| $q_{\text{ref}}$, $\dot{q}_{\text{ref}}$ | `q_ref = ref[:6]`, `qd_ref = ref[6:12]` (reference 앞 6개=목표 각도, 다음 6개=목표 각속도) |
| $q$, $\dot{q}$ | `q = state.q[:self.nq]`, `qd = state.qd[:self.nq]` |
| $e_{\text{int}} \mathrel{+}= e \cdot \Delta t$ | `self._integral += err * self.dt` |
| 적분 클리핑 | `self._integral = np.clip(..., -integral_clip, integral_clip)` |
| $\tau = K_p e + K_d \dot{e} + K_i e_{\text{int}}$ | `tau = self.Kp * err + self.Kd * (qd_ref - qd) + self.Ki * self._integral` |

#### 1.4 튜닝 요약

- **Kp**: 오차에 비례해 토크를 냄. 크면 반응 빠르지만 과하면 진동.
- **Ki**: 정상 오차를 줄이지만, 너무 크면 오버슈트·진동.
- **Kd**: 속도 오차로 “제동”을 걸어 진동을 줄임.
- **integral_clip**: 적분 상·하한으로 적분 발산 방지.

---

### 2. 임피던스 제어기

#### 2.1 아이디어

로봇의 **동역학**을 이용합니다.  
“원하는 가속도” $u$를 정한 뒤, $\tau = M(q) u + c(q,\dot{q})$ 로 토크를 주면 실제로 $\ddot{q} = u$ 가 됩니다.  
이 $u$를 **목표 임피던스**(질량–댐퍼–스프링 계처럼)로 설계하는 것이 임피던스 제어입니다.

#### 2.2 수식 전개

**① 로봇 동역학**

$$
M(q) \ddot{q} + c(q, \dot{q}) = \tau
$$

따라서

$$
\tau = M(q) u + c(q, \dot{q}) \quad \Rightarrow \quad \ddot{q} = u
$$

**② 목표 동작 (가상의 2차 시스템)**

위치·속도 오차를 줄이도록 “가상의 질량–댐퍼–스프링”에서 나오는 가속도를 씁니다.

$$
M_d \, u = K_d (q_{\text{ref}} - q) + B_d (\dot{q}_{\text{ref}} - \dot{q})
$$

- $M_d$: 목표 관성 (대각 행렬로 둠)
- $B_d$: 목표 댐핑
- $K_d$: 목표 스티프니스

**③ 원하는 가속도 $u$**

$$
u = M_d^{-1} \Bigl( K_d (q_{\text{ref}} - q) + B_d (\dot{q}_{\text{ref}} - \dot{q}) \Bigr)
$$

(필요하면 $\ddot{q}_{\text{ref}}$ 항을 더할 수 있으며, 코드에서는 $\ddot{q}_{\text{ref}} = 0$ 으로 두었습니다.)

**④ 최종 토크**

$$
\tau = M(q) \, u + c(q, \dot{q})
$$

#### 2.3 코드와의 대응

파일: `controllers/impedance_controller.py`

| 수식 | 대응 코드 (요지) |
|------|------------------|
| $M(q)$ | `M = self._robot.get_M()` (그 전에 `mj_forward`로 현재 $q$, $\dot{q}$ 반영) |
| $c(q, \dot{q})$ | `c = self._robot.get_c()` |
| $q_{\text{ref}}-q$, $\dot{q}_{\text{ref}}-\dot{q}$ | `err_q = q_ref - q`, `err_qd = qd_ref - qd` |
| $u = M_d^{-1}(K_d e + B_d \dot{e})$ | `u = qdd_ref + np.linalg.solve(self.Md, self.Kd @ err_q + self.Bd @ err_qd)` (qdd_ref=0) |
| $\tau = M u + c$ | `tau = M @ u + c` |

#### 2.4 튜닝 요약

- **Md**: 목표 “질량” 느낌. 크면 반응이 무겁고, 작으면 가벼워짐.
- **Bd**: 댐핑. 크면 진동 감소, 작으면 더 흔들릴 수 있음.
- **Kd**: 스티프니스. 크면 목표 위치로 더 세게 당김.

---

### 3. LQR 제어기

#### 3.1 아이디어

로봇은 비선형이지만, **한 점(평형점)** 근처에서 1차 근사하면 **선형** 시스템으로 쓸 수 있습니다.  
그 선형 모델에 대해 “상태 오차와 입력에 대한 2차 비용”을 최소화하는 제어가 **LQR**이고, 해는 **선형 상태 피드백** $u = -K (x - x_{\text{ref}})$ 형태로 나옵니다.  
여기서는 평형점 $(q_{\text{eq}}, \dot{q}=0)$ 주변에서 선형화한 뒤 LQR 게인 $K$를 한 번 구하고, 매 스텝 $x = [q,\, \dot{q}]^{\top}$, $x_{\text{ref}} = [q_{\text{ref}},\, \dot{q}_{\text{ref}}]^{\top}$ 로 두고 같은 $K$를 사용합니다.

#### 3.2 수식 전개

**① 상태 정의**

$$
x = \begin{bmatrix} q \\ \dot{q} \end{bmatrix}, \qquad
x_{\text{ref}} = \begin{bmatrix} q_{\text{ref}} \\ \dot{q}_{\text{ref}} \end{bmatrix}
$$

**② 비선형 동역학**

$$
\ddot{q} = M(q)^{-1} \bigl( \tau - c(q, \dot{q}) \bigr), \qquad
\dot{x} = \begin{bmatrix} \dot{q} \\ \ddot{q} \end{bmatrix}
$$

**③ 평형점 근처 선형화**

평형점 $(q_{\text{eq}}, \dot{q}=0)$에서 $\tau = c(q_{\text{eq}}, 0)$ 이면 $\ddot{q}=0$ 입니다.  
$q$만 미세하게 움직인다고 하면 $c(q,0)$를 $q$에 대해 미분한 행렬 $\frac{\partial c}{\partial q}\big|_{\text{eq}}$ 를 쓰고, $\dot{q}$ 항은 평형점에서 0으로 둡니다.

$$
\delta\ddot{q} \approx -M^{-1} \frac{\partial c}{\partial q}\Big|_{\text{eq}} \delta q + M^{-1} \delta\tau
$$

선형 모델은 다음과 같습니다.

$$
\dot{x} = A x + B u, \qquad u = \tau
$$

$$
A = \begin{bmatrix} 0 & I \\ -M^{-1} \frac{\partial c}{\partial q}\big|_{\text{eq}} & 0 \end{bmatrix}, \qquad
B = \begin{bmatrix} 0 \\ M^{-1} \end{bmatrix}
$$

**④ LQR 비용과 최적 제어**

비용:

$$
J = \int \Bigl( (x - x_{\text{ref}})^{\top} Q (x - x_{\text{ref}}) + u^{\top} R u \Bigr) \, dt
$$

연속시간 LQR의 해는 **CARE**(Continuous-time Algebraic Riccati Equation)의 해 $P$로 주어지고:

$$
A^{\top} P + P A - P B R^{-1} B^{\top} P + Q = 0
$$

$$
K = R^{-1} B^{\top} P
$$

최적 제어는:

$$
u = -K (x - x_{\text{ref}})
$$

#### 3.3 코드와의 대응

파일: `controllers/lqr_controller.py`

| 수식 | 대응 코드 (요지) |
|------|------------------|
| 평형점 $q_{\text{eq}}$ | `self.q_eq` (run.py에서 초기 목표로 설정 가능) |
| $M(q_{\text{eq}})$, $c(q_{\text{eq}}, 0)$ | `get_M()`, `get_c()` (mj_forward 후) |
| $\frac{\partial c}{\partial q}$ 수치 미분 | `dcdq[:, j] = (cp - cm) / (2*eps)` ($q_j$만 ±eps 움직여 $c$ 차이로 편미분) |
| $A_{21} = -M^{-1} \frac{\partial c}{\partial q}$ | `A21 = -Minv @ dcdq` |
| $A$, $B$ 블록 행렬 | `A = np.block([...])`, `B = np.block([...])` |
| CARE 해 $P$, 게인 $K$ | `P = solve_continuous_are(A, B, Q, R)`, `K = np.linalg.solve(R, B.T @ P)` |
| $x$, $x_{\text{ref}}$ | `x = np.concatenate([state.q[:6], state.qd[:6]])`, `x_ref = np.concatenate([q_ref, qd_ref])` |
| $u = -K(x - x_{\text{ref}})$ | `u = -self.K @ (x - x_ref)` |

#### 3.4 튜닝 요약

- **Q**: 상태(위치·속도) 오차 가중치. 크면 오차를 더 줄이려 하고 입력이 커질 수 있음.
- **R**: 입력(토크) 가중치. 크면 입력을 아끼고, 작으면 더 공격적으로 제어.
- 평형점을 바꾸면 선형화 지점이 바뀌므로 이론상 $K$를 다시 구하는 것이 맞습니다. run.py에서는 초기 목표를 평형점으로 넘겨 그 근처에서 선형화합니다.

---

### 4. MPC 제어기

#### 4.1 아이디어

매 제어 주기마다 **앞으로 N 스텝** 동안의 토크 시퀀스 $\tau_0, \ldots, \tau_{N-1}$ 를 한 번에 정합니다.  
사용하는 모델은 **비선형 동역학** $M(q)\ddot{q}+c(q,\dot{q})=\tau$ 를 그대로 쓰고, 시간을 $\Delta t$ 로 나눠 **이산화**(오일러)한 뒤,

- **비용**: 각 스텝에서 $(x_k - x_{\text{ref}})^{\top} Q (x_k - x_{\text{ref}}) + \tau_k^{\top} R \tau_k$ 를 모두 더한 값
- **제약**: 매 스텝 $x_{k+1} = f(x_k, \tau_k)$ (동역학 한 스텝)

를 만족하도록 $\tau_0, \ldots, \tau_{N-1}$ 를 최적화합니다.  
실제로는 **첫 토크 $\tau_0^*$ 만** 로봇에 넣고, 다음 주기에서 다시 현재 상태를 기준으로 N 스텝 최적화를 반복합니다 (**Receding Horizon**).

#### 4.2 수식 전개

**① 연속 동역학**

$$
M(q) \ddot{q} + c(q, \dot{q}) = \tau \quad \Rightarrow \quad \ddot{q} = M(q)^{-1}\bigl( \tau - c(q, \dot{q}) \bigr)
$$

**② 한 스텝 이산화 (오일러, 간격 $\Delta t$)**
$$
\ddot{q}_k = M(q_k)^{-1}\bigl( \tau_k - c(q_k, \dot{q}_k) \bigr)
$$

$$
q_{k+1} = q_k + \Delta t \, \dot{q}_k
$$

$$
\dot{q}_{k+1} = \dot{q}_k + \Delta t \, \ddot{q}_k
$$

**③ 비용 (구간 길이 $N$)**
$$
x_k = \begin{bmatrix} q_k \\ \dot{q}_k \end{bmatrix}
$$

$$
J = \sum_{k=0}^{N-1} \Bigl( (x_k - x_{\text{ref}})^{\top} Q (x_k - x_{\text{ref}}) + \tau_k^{\top} R \tau_k \Bigr)
$$

**④ 최적화 및 적용**

- **결정 변수**: $\tau_0, \tau_1, \ldots, \tau_{N-1}$
- **초기 상태** $x_0$에서 위 이산 동역학으로 $x_1, \ldots, x_N$ 을 구하면서 $J$ 를 계산하고, $J$ 를 최소화합니다.
- 해에서 **첫 토크만** 적용:

$$
\tau_{\text{apply}} = \tau_0^*
$$

#### 4.3 코드와의 대응

파일: `controllers/mpc_controller.py`

| 수식 | 대응 코드 (요지) |
|------|------------------|
| $\ddot{q} = M^{-1}(\tau - c)$ | `qdd = np.linalg.solve(M, tau - c)` |
| $q_{k+1} = q_k + \Delta t \dot{q}_k$ | `q_next = q + self.dt * qd` |
| $\dot{q}_{k+1} = \dot{q}_k + \Delta t \ddot{q}_k$ | `qd_next = qd + self.dt * qdd` |
| 한 스텝 전체 | `_dynamics_step(q, qd, tau)`: set_state → mj_forward → M, c → 위 두 식 |
| $(x_k - x_{\text{ref}})^{\top} Q (x_k - x_{\text{ref}})$ | `(np.concatenate([q, qd]) - x_ref) @ self.Q @ (...)` |
| $\tau_k^{\top} R \tau_k$ | `tau @ self.R @ tau` |
| $J = \sum_k (\ldots)$ | `_rollout_cost` 내부의 `for k in range(N):` 루프에서 cost 누적 |
| $\tau_0^*$ 적용 | `u_opt = result.x.reshape(N, nq); return u_opt[0]` (최적화 결과의 첫 토크만 반환) |

#### 4.4 튜닝 요약

- **horizon** $N$: 크면 더 먼 미래를 보지만 계산량이 늘어남.
- **Q, R**: 상태 오차 vs 입력 크기 트레이드오프 (LQR과 유사).
- **bounds**: 토크 상·하한 (코드에서는 ±150, 모델 액추에이터 한도와 맞춤).

---

## 로봇 모델

- **robot/assets/arm_6dof.xml**: MuJoCo MJCF 형식의 6관절 매니퓰레이터입니다.  
  다른 6-DOF 모델(URDF 등)을 MJCF로 변환해 교체할 수 있습니다.

---

## 새 제어기 추가

1. **controllers/new_controller.py** 에 `ControllerBase` 를 상속한 클래스를 만들고, `compute_control(state, reference, t) -> tau` 를 구현합니다.
2. **controllers/__init__.py** 의 `CONTROLLERS` 에 `"new": ("controllers.new_controller", "NewController")` 를 추가합니다.
3. **run.py** 의 `make_controller` 에서, 필요하면 `robot` / `dt` / `q0_ref` 등 인자 전달 규칙에 맞춰 새 제어기를 생성하도록 분기합니다.

---

이 문서의 수식은 **인라인은 $ ... $**, **디스플레이(한 줄)는 $$ ... $$** 로 작성했습니다.  
뷰어가 수학 렌더링을 지원하면 위와 같이 수식으로 보이고, 지원하지 않으면 LaTeX 소스가 그대로 보일 수 있습니다.
