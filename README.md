# 🚀 Ballistic Missile Trajectory Simulator

A high-fidelity physics simulator and interactive dashboard built in Python. This tool models the flight paths of ballistic objects using 4th-order Runge-Kutta (RK4) numerical integration and calculates resulting ground-impact blast zones using empirical overpressure formulas.

## ✨ Features

* **Advanced Trajectory Physics:** Uses RK4 integration to calculate flight paths with a configurable time-step, accounting for variable gravity and aerodynamic drag up to 1,200+ km altitudes.
* **Atmospheric Modeling:** Implements an ISA exponential atmosphere model (8,500 m scale height) to dynamically calculate air density and drag forces throughout the flight envelope.
* **Interactive Dashboard:** Built with Streamlit and Plotly, allowing simultaneous, real-time comparison of up to 4 different missile configurations.
* **Blast Overpressure Engine:** Uses Hopkinson-Cranz scaling and the Brode (1955) empirical formula to map 5 NATO TM 5-1300 damage thresholds (7–350 kPa) into geo-accurate, top-down blast zones up to a 10,000 m radius.

## 🧮 The Physics Engine

### Kinematics & Drag
The simulation goes beyond standard projectile motion by updating environmental variables at every time-step:
* **Variable Gravity:** Recalculated dynamically based on distance from the Earth's center using $g = G \cdot M / r^2$.
* **Aerodynamic Drag:** Calculated using $F = \frac{1}{2}\rho v^2 C_d A$, where $\rho$ (air density) decays exponentially with altitude.

### Blast Modeling
Impact effects are calculated using:
* **Scaled Distance ($Z$):** $Z = R / W^{1/3}$ 
* **Overpressure:** Converted to peak kPa to determine lethal radii, structural collapse zones, and minor damage thresholds based on standard military guidelines.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sparsh-hub/ballistic-simulator.git](https://github.com/sparsh-hub/ballistic-simulator.git)
   cd ballistic-simulator```

2. **Install Dependencies:**
   ```bash
   ```pip install numpy pandas streamlit plotly```

3. **Run The App:**
   ```bash
   ```streamlit run ICBM.py```
