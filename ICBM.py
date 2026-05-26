import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List

GRAVITATIONAL_CONSTANT = 6.67430e-11
EARTH_MASS             = 5.972e24
EARTH_RADIUS           = 6_371_000.0   
CROSS_SECTIONAL_AREA   = 1.0           

@dataclass
class MissileConfig:
    name:             str
    mass:             float   
    initial_velocity: float   
    launch_angle:     float   
    drag_coefficient: float
    warhead_yield:    float   
    color:            str

def air_density(altitude: float) -> float:
    """Exponential atmosphere (ISA approximation)."""
    return 1.225 * np.exp(-max(altitude, 0) / 8500.0)


def _eom(state: np.ndarray, t: float, mass: float, cd: float) -> np.ndarray:
    """
    Equations of motion including variable gravity and aerodynamic drag.
    state = [x, y, vx, vy]
    returns d/dt [x, y, vx, vy]
    """
    x, y, vx, vy = state
    altitude = max(y, 0.0)
    speed    = np.hypot(vx, vy)
    r        = EARTH_RADIUS + altitude

    g   = GRAVITATIONAL_CONSTANT * EARTH_MASS / r**2
    rho = air_density(altitude)

    if speed > 1e-6:
        drag_acc = 0.5 * rho * speed**2 * cd * CROSS_SECTIONAL_AREA / mass
        ax = -drag_acc * (vx / speed)
        ay = -g - drag_acc * (vy / speed)
    else:
        ax, ay = 0.0, -g

    return np.array([vx, vy, ax, ay])


def rk4_step(state: np.ndarray, t: float, dt: float,
             mass: float, cd: float) -> np.ndarray:
    """Single 4th-order Runge-Kutta step."""
    k1 = _eom(state,              t,          mass, cd)
    k2 = _eom(state + 0.5*dt*k1, t + 0.5*dt, mass, cd)
    k3 = _eom(state + 0.5*dt*k2, t + 0.5*dt, mass, cd)
    k4 = _eom(state +     dt*k3, t +     dt, mass, cd)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate_trajectory(cfg: MissileConfig, dt: float = 10.0) -> dict:
    """Integrate full trajectory with RK4; stop when y < 0."""
    angle_rad = np.radians(cfg.launch_angle)
    state = np.array([
        0.0, 0.0,
        cfg.initial_velocity * np.cos(angle_rad),
        cfg.initial_velocity * np.sin(angle_rad),
    ])

    states, times = [state.copy()], [0.0]
    t = 0.0
    MAX_T = 3600 * 4 

    while t < MAX_T:
        state = rk4_step(state, t, dt, cfg.mass, cfg.drag_coefficient)
        t += dt
        states.append(state.copy())
        times.append(t)
        if state[1] < 0:
            break

    arr     = np.array(states)
    speeds  = np.hypot(arr[:, 2], arr[:, 3])
    apogee  = int(np.argmax(arr[:, 1]))

    return {
        "x":           arr[:, 0],
        "y":           arr[:, 1],
        "vx":          arr[:, 2],
        "vy":          arr[:, 3],
        "speed":       speeds,
        "altitude":    np.maximum(arr[:, 1], 0),
        "time":        np.array(times),
        "apogee_idx":  apogee,
        "max_altitude": arr[:, 1].max(),
        "range":       arr[-1, 0],
        "flight_time": times[-1],
        "config":      cfg,
    }


#  Blast Overpressure 

def hopkinson_z(R: float, W: float) -> float:
    """Scaled distance Z = R / W^(1/3)  [m / kg^(1/3)]."""
    return R / (W ** (1.0 / 3.0))


def brode_overpressure_kpa(Z: float) -> float:
    """
    Brode (1955) empirical peak overpressure from scaled distance Z.
    Returns kPa; clamped to 0.
    """
    if Z <= 0:
        return 1e6
    if Z < 0.5:
        P_bar = 6.7 / Z**3 + 1.0
    else:
        P_bar = (0.975 / Z) + (1.455 / Z**2) + (5.85 / Z**3) - 0.019
    return max(P_bar * 100.0, 0.0)   


def blast_profile(yield_kg: float, max_r: float = 3000, n: int = 600):
    """
    Returns (radii, overpressures_kpa, damage_radii_dict).
    Damage thresholds from NATO/TM 5-1300.
    """
    radii     = np.linspace(1, max_r, n)
    pressures = np.array([brode_overpressure_kpa(hopkinson_z(r, yield_kg)) for r in radii])

    thresholds = {
        "Eardrum rupture (34 kPa)":      34,
        "Window/glass breakage (7 kPa)":  7,
        "Lung damage (100 kPa)":         100,
        "Structural collapse (170 kPa)": 170,
        "Lethal overpressure (350 kPa)": 350,
    }
    damage_radii = {}
    for label, thr in thresholds.items():
        above = np.where(pressures >= thr)[0]
        damage_radii[label] = float(radii[above[-1]]) if len(above) else 0.0

    return radii, pressures, damage_radii, thresholds



# STREAMLIT 

PALETTE = ["#FF4B4B", "#4B9CFF", "#4BFF9A", "#FFD700"]
DEFAULTS = [
    dict(name="Alpha",  velocity=8000,  angle=45, mass=15000, yield_kg=500,  drag=0.005),
    dict(name="Beta",   velocity=10000, angle=55, mass=20000, yield_kg=1000, drag=0.005),
    dict(name="Gamma",  velocity=12000, angle=40, mass=25000, yield_kg=2000, drag=0.004),
    dict(name="Delta",  velocity=9000,  angle=50, mass=18000, yield_kg=750,  drag=0.006),
]

PLOTLY_LAYOUT = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(color="#e0e0e0"),
    xaxis         = dict(showgrid=True, gridcolor="rgba(180,180,180,0.15)"),
    yaxis         = dict(showgrid=True, gridcolor="rgba(180,180,180,0.15)"),
)


def hex_to_rgba(h: str, alpha: float = 0.15) -> str:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def circle_xy(radius: float, n: int = 120):
    t = np.linspace(0, 2 * np.pi, n)
    return radius * np.cos(t), radius * np.sin(t)


def main():
    st.set_page_config(
        page_title = "Ballistic Missile Simulator",
        layout     = "wide",
        page_icon  = "",
    )

    st.title("Ballistic Missile Trajectory Simulator")
    st.caption(
        "RK4 integration · Multi-missile comparison · "
        "Hopkinson-Cranz blast · Monte Carlo CEP · Streamlit dashboard"
    )
    st.sidebar.header("Configuration")
    n_missiles = st.sidebar.slider("Missiles to compare", 1, 4, 2)
    dt = float(st.sidebar.slider("RK4 time-step  Δt (s)", 2, 60, 10))

    configs: List[MissileConfig] = []
    for i in range(n_missiles):
        d = DEFAULTS[i]
        with st.sidebar.expander(f"Missile {d['name']}", expanded=(i == 0)):
            name  = st.text_input("Name",                      d["name"],     key=f"n{i}")
            vel   = st.slider("Initial velocity (m/s)",  3000, 15000, d["velocity"], key=f"v{i}")
            ang   = st.slider("Launch angle (°)",           10,    80, d["angle"],   key=f"a{i}")
            mass  = st.slider("Mass (kg)",               1000, 50000, d["mass"],    key=f"m{i}", step=500)
            cd    = st.slider("Drag coefficient",         0.001, 0.05, d["drag"],   key=f"cd{i}", format="%.3f")
            yld   = st.slider("Warhead yield (kg TNT)",    100, 10000, d["yield_kg"], key=f"y{i}")
            configs.append(MissileConfig(name, mass, vel, ang, cd, yld, PALETTE[i]))

    with st.spinner("Running RK4 simulations …"):
        trajs = [simulate_trajectory(c, dt=dt) for c in configs]

    tab1, tab2 = st.tabs([
        "Trajectories",
        "Blast Overpressure",
    ])

    with tab1:
        st.subheader("Multi-Missile Trajectory Comparison  (RK4)")

        fig = go.Figure()
        for tr in trajs:
            cfg = tr["config"]
            xk  = tr["x"] / 1000
            yk  = tr["y"] / 1000
            ap  = tr["apogee_idx"]

            fig.add_trace(go.Scatter(
                x=xk[:ap+1], y=yk[:ap+1],
                mode="lines", name=f"{cfg.name}  ↑",
                line=dict(color=cfg.color, width=2.5),
            ))
            fig.add_trace(go.Scatter(
                x=xk[ap:], y=yk[ap:],
                mode="lines", name=f"{cfg.name}  ↓",
                line=dict(color=cfg.color, width=2.5, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=[xk[ap]], y=[yk[ap]],
                mode="markers+text",
                marker=dict(color=cfg.color, size=11, symbol="star"),
                text=[f"  {yk[ap]:.0f} km"],
                textfont=dict(color=cfg.color),
                textposition="middle right",
                showlegend=False,
            ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Ballistic Trajectories",
            xaxis_title="Horizontal Range (km)",
            yaxis_title="Altitude (km)",
            height=520,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, width="stretch")

        
        c1, c2 = st.columns(2)
        with c1:
            fs = go.Figure()
            for tr in trajs:
                fs.add_trace(go.Scatter(
                    x=tr["time"]/60, y=tr["speed"],
                    name=tr["config"].name,
                    line=dict(color=tr["config"].color),
                ))
            fs.update_layout(**PLOTLY_LAYOUT,
                title="Speed vs Time",
                xaxis_title="Time (min)", yaxis_title="Speed (m/s)", height=340)
            st.plotly_chart(fs, width="stretch")

        with c2:
            fa = go.Figure()
            for tr in trajs:
                fa.add_trace(go.Scatter(
                    x=tr["time"]/60, y=tr["altitude"]/1000,
                    name=tr["config"].name,
                    line=dict(color=tr["config"].color),
                ))
            fa.update_layout(**PLOTLY_LAYOUT,
                title="Altitude vs Time",
                xaxis_title="Time (min)", yaxis_title="Altitude (km)", height=340)
            st.plotly_chart(fa, width="stretch")

        
        st.subheader("Comparison Table")
        rows = []
        for tr in trajs:
            rows.append({
                "Missile":              tr["config"].name,
                "Launch V (m/s)":       tr["config"].initial_velocity,
                "Launch Angle (°)":     tr["config"].launch_angle,
                "Max Altitude (km)":    round(tr["max_altitude"] / 1000, 1),
                "Range (km)":           round(tr["range"] / 1000, 1),
                "Flight Time (min)":    round(tr["flight_time"] / 60, 1),
                "Max Speed (m/s)":      round(float(tr["speed"].max()), 0),
                "Warhead (kg TNT)":     tr["config"].warhead_yield,
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with tab2:
        st.subheader("Blast Overpressure  —  Hopkinson-Cranz Scaling")
        st.markdown(
            r"""
            **Scaled distance:** $Z = R \;/\; W^{1/3}$  
            """
        )

        sel_b = st.selectbox("Missile for blast analysis",
                             [c.name for c in configs], key="sel_blast")
        cfg_b  = configs[[c.name for c in configs].index(sel_b)]
        max_r  = st.slider("Analysis radius (m)", 500, 10_000, 3000, key="max_r")

        radii, pressures, dmg_radii, thresholds = blast_profile(cfg_b.warhead_yield, max_r)

        # ------------- OVERPRESSURE CURVE ----------------------------
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=radii, y=pressures,
            mode="lines", name="Peak Overpressure",
            line=dict(color="#FF4B4B", width=3),
            fill="tozeroy", fillcolor="rgba(255,75,75,0.10)",
        ))

        thr_colors = ["#FFA500", "#FF6600", "#CC0000", "#880000", "#440000"]
        for (label, val), col in zip(thresholds.items(), thr_colors):
            fig_b.add_hline(
                y=val, line_dash="dot", line_color=col,
                annotation_text=label,
                annotation_position="bottom right",
                annotation_font_color=col,
            )

        fig_b.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Blast Profile — {cfg_b.warhead_yield} kg TNT",
            xaxis_title="Standoff Distance (m)",
            yaxis_title="Peak Overpressure (kPa)",
            yaxis_type="log",
            height=480,
        )
        st.plotly_chart(fig_b, width="stretch")

        cols = st.columns(len(dmg_radii))
        for col, (label, r) in zip(cols, dmg_radii.items()):
            col.metric(label.split("(")[0].strip(), f"{r:.0f} m" if r > 0 else "—")

        st.subheader("Blast-Effect Zone Map (top-down view)")
        zone_colors = [
            "rgba(68,0,0,0.35)",
            "rgba(136,0,0,0.28)",
            "rgba(204,0,0,0.22)",
            "rgba(255,102,0,0.18)",
            "rgba(255,165,0,0.14)",
        ]
        fig_map = go.Figure()
        for (label, r), zc in zip(
            sorted(dmg_radii.items(), key=lambda x: -x[1]), zone_colors
        ):
            if r > 0:
                cx, cy = circle_xy(r)
                fig_map.add_trace(go.Scatter(
                    x=cx, y=cy,
                    fill="toself",
                    name=f"{label.split('(')[0].strip()}  ({r:.0f} m)",
                    line=dict(color=zc, width=1.5),
                    fillcolor=zc,
                ))
        fig_map.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(color="white", size=12, symbol="x-thin", line_width=2),
            name="Impact point",
        ))
        fig_map.update_layout(
            **PLOTLY_LAYOUT,
            title="Blast Zones",
            xaxis_title="m", yaxis_title="m",
            yaxis_scaleanchor="x",
            height=520,
        )
        st.plotly_chart(fig_map, width="stretch")


if __name__ == "__main__":
    main()