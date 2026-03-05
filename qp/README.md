# QP 제어 (목표 각도 추종)

목표 관절각 \(\mathbf{q}_d\)를 추종하기 위해, 매 스텝에서 **원하는 가속도** \(\mathbf{a}\)를 2차 비용(정규화 포함)의 QP로 정해 \(\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\mathbf{a} + \mathbf{C} + \mathbf{G}\)로 토크를 계산한다. 여기서는 제약 없는 QP만 사용한다.

## 1. 수식 정리

### 목표 가속도 (PD형)
\[
\mathbf{b} = \ddot{\mathbf{q}}_d + K_p(\mathbf{q}_d - \mathbf{q}) + K_d(\dot{\mathbf{q}}_d - \dot{\mathbf{q}})
\]

### QP (정규화 포함)
\[
\min_{\mathbf{a}} \ \frac{1}{2} \|\mathbf{b} - \mathbf{a}\|^2 + \frac{\lambda}{2} \|\mathbf{a}\|^2
\]
전개하면 \(\frac{1}{2}\mathbf{a}^T(I + \lambda I)\mathbf{a} + (-\mathbf{b})^T\mathbf{a}\) + 상수.  
즉 \(H = (1+\lambda)I\), \(\mathbf{c} = -\mathbf{b}\).

### 해 및 토크
\[
\mathbf{a}^* = H^{-1}(-\mathbf{c}) = (1+\lambda)^{-1}\mathbf{b}
\]
\[
\boldsymbol{\tau} = \mathbf{M}(\mathbf{q})\mathbf{a}^* + \mathbf{C}(\mathbf{q},\dot{\mathbf{q}}) + \mathbf{G}(\mathbf{q})
\]

## 2. 수식–코드 매칭

| 수식 | 코드 |
|------|------|
| \(\mathbf{b} = \ddot{\mathbf{q}}_d + K_p\mathbf{e} + K_d\dot{\mathbf{e}}\) | `run.py`: `b = QDD_D + Kp @ e + Kd @ edot` |
| \(H = (1+\lambda)I\), \(\mathbf{c} = -\mathbf{b}\) | `run.py`: `H_qp = (1.0 + lam) * np.eye(3)`, `c_qp = -b` |
| \(\mathbf{a}^* = H^{-1}(-\mathbf{c})\) (QP 해) | `run.py`: `a_star = np.linalg.solve(H_qp, -c_qp)` |
| \(\boldsymbol{\tau} = \mathbf{M}\mathbf{a}^* + \mathbf{C} + \mathbf{G}\) | `run.py`: `tau = M(q) @ a_star + C_vec(q, qd) + G(q)` |

## 3. 실행 방법

```bash
cd qp
python run.py
```

## 4. 결과

`qp_result.png`: 위—관절각 \(q_1,q_2,q_3\)와 목표(점선). 아래—각도 오차 \(\mathbf{e}\).
