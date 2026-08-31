# Sensor Physics & Ring Schedule Justification (Day 0–1 Memo)

**From:** Srinivas (R1 — Representation & Prior Art)  
**To:** Aakash (D1 — Grid Engine) & Team  
**Date:** 2026-08-29  
**Subject:** Physical Justification of the 5/10/20/40 cm Schedule, Ring-Sweep Filling, and Hard Geometric Limits

---

## 1. Executive Summary

This memo provides the physical and mathematical justification for our variable-resolution ring schedule. It reframes our coarsening from a mere "compression trick" into a **physically derived necessity dictated by LiDAR beam divergence**.

### Key Takeaways for the Team:
1. **Azimuthal spacing grows linearly ($s_{\text{az}} \propto r$), but radial ground spacing grows quadratically ($s_{\text{rad}} \propto r^2$).**
2. **A uniform 5 cm grid at 50 m is 99.87% empty in a single frame.** Uniform high resolution at range is an empty array with a confident label.
3. **Far rings (Rings 2 & 3) are populated by vehicle ego-motion ("Ring-Sweep Filling"), not instantaneous returns.**
4. **Potholes ($30\text{ cm}$) are physically invisible beyond $\sim 8.3\text{ m}$ in a single scan.**
5. **The blind cone under the vehicle has a radius of $3.74\text{ m}$ (11.0% of Ring 0) and must always remain `UNKNOWN`, never `FREE`.**

---

## 2. Mathematical Derivations

### 2.1 Azimuthal Spacing (Linear with Range)
Along the arc of a single laser ring at range $r$:
$$s_{\text{az}}(r) = r \cdot \Delta\theta$$
For KITTI's Velodyne HDL-64E ($\Delta\theta = 0.2^\circ = 3.49 \times 10^{-3}\text{ rad}$):
- At $10\text{ m}$: $s_{\text{az}} = 3.5\text{ cm}$
- At $25\text{ m}$: $s_{\text{az}} = 8.7\text{ cm}$
- At $50\text{ m}$: $s_{\text{az}} = 17.5\text{ cm}$
- At $100\text{ m}$: $s_{\text{az}} = 34.9\text{ cm}$

Our schedule ($5 / 10 / 20 / 40\text{ cm}$) tracks the sensor's own sample spacing to within a factor of $1.15\times$ at every ring boundary (Nyquist-optimal matching).

---

### 2.2 Radial Ground Spacing (Quadratic with Range)
A laser beam emitted at sensor height $h_s$ with depression angle $\phi$ strikes flat ground at $r = h_s / \tan\phi \approx h_s / \phi$. Differentiating with respect to $\phi$:
$$\frac{dr}{d\phi} = -\frac{h_s}{\phi^2} = -\frac{r^2}{h_s}$$
The radial ground spacing between adjacent vertical beams separated by $\Delta\phi$ is:
$$\mathbf{s_{\text{rad}}(r) = \frac{r^2}{h_s} \cdot \Delta\phi}$$

For KITTI ($h_s = 1.73\text{ m}$, $\Delta\phi = \frac{26.9^\circ}{63} = 7.45 \times 10^{-3}\text{ rad}$):

| Range $r$ | Azimuthal $s_{\text{az}}$ | **Radial Ground Spacing $s_{\text{rad}}$** |
|---|---|---|
| $10\text{ m}$ | $3.5\text{ cm}$ | **$0.43\text{ m}$** |
| $25\text{ m}$ | $8.7\text{ cm}$ | **$2.69\text{ m}$** |
| $50\text{ m}$ | $17.5\text{ cm}$ | **$10.77\text{ m}$** |
| $80\text{ m}$ | $27.9\text{ cm}$ | **$27.6\text{ m}$** |
| $100\text{ m}$ | $34.9\text{ cm}$ | **$43.1\text{ m}$** |

> **Critical Observation:** At $50\text{ m}$, consecutive laser rings strike the asphalt **$10.8\text{ meters}$ apart**.

---

### 2.3 Single-Frame Cell Fill Rate ($P_{\text{fill}}$)
The probability that a ground cell of size $c$ at range $r$ receives any return in a single sweep is:
$$P_{\text{fill}}(r, c) \approx \min\left(1, \frac{c}{s_{\text{rad}}(r)}\right) \cdot \min\left(1, \frac{c}{s_{\text{az}}(r)}\right)$$

| Ring | Range ($r$) | Cell Size ($c$) | $c / s_{\text{rad}}$ | $c / s_{\text{az}}$ | **$P_{\text{fill}}$** |
|---|---|---|---|---|---|
| **Ring 0** | $10\text{ m}$ | $5\text{ cm}$ | $0.116$ | $1.0$ | **$11.6\%$** |
| **Ring 1** | $25\text{ m}$ | $10\text{ cm}$ | $0.037$ | $1.0$ | **$3.7\%$** |
| **Ring 2** | $50\text{ m}$ | $20\text{ cm}$ | $0.019$ | $1.0$ | **$1.9\%$** |
| **Ring 3** | $100\text{ m}$ | $40\text{ cm}$ | $0.009$ | $1.0$ | **$0.9\%$** |
| *Uniform Baseline* | $50\text{ m}$ | $5\text{ cm}$ | $0.005$ | $0.29$ | **$0.13\%$ ($99.87\%$ empty)** |

Coarsening to $20\text{ cm}$ at Ring 2 improves single-frame cell evidence by **$15\times$** over the uniform $5\text{ cm}$ baseline.

---

### 2.4 The "Ring-Sweep Filling" Principle
Because single-frame fill rate in Rings 2–3 is $< 2\%$, the far-field elevation map is **not constructed instantaneously**; it is integrated across time as the vehicle drives forward. The laser rings sweep across the road surface like a push-broom scanner.

**Consequence for Evaluation:** Single-frame far-ring metrics are meaningless. All evaluation in Rings 2 and 3 must report accuracy as a function of **frames-since-first-observation**.

---

## 3. Derived Hard Physical Limits

### 3.1 Negative Obstacle (Pothole) Range Limit
A negative obstacle of diameter $W$ is only sampled if $W > s_{\text{rad}}(r)$. Thus, the maximum detection range is:
$$\mathbf{r_{\max}(W) = \sqrt{\frac{W \cdot h_s}{\Delta\phi}}}$$

| Defect Diameter ($W$) | Single-Scan Maximum Detection Range |
|---|---|
| **$30\text{ cm}$ (Typical Pothole)** | **$8.3\text{ m}$** |
| **$50\text{ cm}$ (Manhole / Large Trench)** | **$10.8\text{ m}$** |
| **$1.0\text{ m}$ (Road Washout)** | **$15.2\text{ m}$** |

**Rule for D1/D2:** Never claim potholes can be detected at $25\text{ m}$ or $50\text{ m}$ in a single scan. Cells beyond $8.3\text{ m}$ must be marked `UNKNOWN`, never `FREE`.

---

### 3.2 Sensor Blind Cone Geometry
The lowest laser beam has a maximum depression of $\phi_{\min} = 24.8^\circ$. The blind disc directly beneath the vehicle has radius:
$$r_{\text{blind}} = \frac{h_s}{\tan\phi_{\min}} = \frac{1.73}{\tan(24.8^\circ)} = \mathbf{3.74\text{ m}}$$
- **Blind area:** $\pi \cdot (3.74)^2 = 43.9\text{ m}^2$ (which is **$11.0\%$** of Ring 0's $20\text{ m} \times 20\text{ m}$ area).
- **Rule:** These cells must have `FLAG_BLIND` asserted and remain `OCC_UNKNOWN`.
