import json
import copy
from pathlib import Path
import csv
import numpy as np
import ast
import matplotlib.pyplot as plt
from dataclasses import dataclass
from dataclasses import asdict
from typing import Optional
from bisect import bisect_left


@dataclass
class Inputs:
    dp: float      # particle diameter [m]
    rho_s: float    # particle density [kg/m3]
    rho_g: float    # gas density [kg/m3]
    mu: float      # gas viscosity [Pa*s]
    eps_mf: float  # void fraction at mf [-]
    phi_s: float  # sphericity [-]
    g: float = 9.81 # gravity [m/s2]
    L_mf: Optional[float] = None
    gas_name: Optional[str] = None #"air", "nitrogen", "helium", "co2"
    T: Optional[float] = None # gas temperature [K]
    P: float = 101325.0# absolute pressure [Pa]
    column_D: Optional[float] = None# column inner diameter [m], for L from mass
    bed_mass: Optional[float] = None# total solids mass in bed [kg], for L from mass
    u_sp: Optional[float]  = None  # total solids mass in bed [kg], for L from mass

def get_si_unit(key: str) -> str:
    """
    Returns the SI unit for a given parameter key used in the JSON input files.
    """
    units = {
        "dp": "[m]",                # particle diameter
        "rho_s": "[kg/m^3]",        # solid density
        "rho_g": "[kg/m^3]",        # gas density
        "mu": "[Pa s]",             # dynamic viscosity
        "eps_mf": "[-]",            # dimensionless (void fraction)
        "phi_s": "[-]",             # dimensionless (sphericity)
        "g": "[m/s^2]",             # gravitational acceleration
        "gas_name": "[-]",          # string, no unit
        "T": "[K]",                 # temperature
        "P": "[Pa]",                # pressure
        "L_mf": "[m]",              # bed height at minimum fluidization
        "bed_mass": "[kg]",         # mass of granular bed
        "column_D": "[m]",          # column diameter
        "u_sp": "[m/s]",            # superficial velocity
    }
    return units.get(key, "")


def get_gas_properties_from_input(data: dict, coolprop: str) -> tuple[float, float]:
    """
    Resolves rho_g [kg/m3], mu [Pa*s] if not defined in the input file based on T,P.
    Resolves T [K], P[Pa] if not  defined in the input file.
    Uses Coolprop if wished, else it calls the function gas_properties which is based on empiric correlations(Sutherland's law) and the ideal gas law.
    """
    # extract data
    gas = data.get("gas_name", "air")
    T = data.get("T")
    P = data.get("P")
    if T is None:
        T = 298.15
    if P is None:
        P = 101325.0
    gas_l = gas.strip().lower()

    # map to CoolProp identifiers
    cp_map = {
        "air": "Air",
        "nitrogen": "Nitrogen", "n2": "Nitrogen",
        "oxygen": "Oxygen", "o2": "Oxygen",
        "helium": "Helium", "he": "Helium",
        "argon": "Argon", "ar": "Argon",
        "co2": "CO2",
    }
    if gas_l not in cp_map:
        raise ValueError(f"Unsupported gas '{gas}'. Supported: {list(cp_map.keys())}")
    cp_name = cp_map[gas_l]
    use_cp = (coolprop == "c")

    if use_cp:
        try:
            from CoolProp.CoolProp import PropsSI
        except ImportError:
            print("CoolProp not installed, falling back to ideal-gas correlation.")
            return gas_properties(gas, T, P)
        rho = PropsSI("D", "T", T, "P", P, cp_name)
        mu  = PropsSI("V", "T", T, "P", P, cp_name)
        return float(rho), float(mu)
    else:
        rho, mu = gas_properties(gas, T, P)
        return rho, mu
def gas_properties(gas: str, T: float, P: float = 101325.0) -> tuple[float, float]:
    """Resolves rho_g [kg/m3] and mu [Pa*s] based on empiric correlations(Sutherland's law) and the ideal gas law"""
    # molar masses [kg/mol]
    M = {
        "air": 0.02897,
        "nitrogen": 0.0280134, "n2": 0.0280134,
        "helium": 0.0040026, "he": 0.0040026,
        "co2": 0.04401,
        "argon": 0.03995, "ar": 0.03995,
        "oxygen": 0.0319988, "o2": 0.0319988,

    }
    R = 8.314462618  # J/mol/K
    rho = P * M[gas] / (R * T)  # ideal gas

    #Sutherlands law data:, mu0 at T0, S in K
    #from NIST WebBook: ideal-gas viscosity correlation
    if gas in ("air",):
        mu0, T0, S = 1.716e-5, 273.15, 111.0
    elif gas in ("nitrogen", "n2"):
        mu0, T0, S = 1.663e-5, 300.0, 111.0
    elif gas in ("helium", "he"):
        mu0, T0, S = 1.864e-5, 300.0, 79.4
    elif gas in ("co2",):
        mu0, T0, S = 1.370e-5, 300.0, 240.0
    elif gas in ("oxygen", "o2"):
        mu0, T0, S = 2.07e-5, 300.0, 127.0
    else:
        mu0, T0, S = 1.8e-5, 300.0, 120.0  # fallback

    mu = mu0 * ((T / T0) ** 1.5) * (T0 + S) / (T + S)
    return rho, mu
def calc_ab_ar(p: Inputs):
    """
    Resolves some blocks of the equation Eq. 3.19
    This combines the Ergun equation Eq. 3.6 and the equation Eq. 3.17. The latter equates drag force to the weight of the particles.
    """
    B = 150.0 * (1.0 - p.eps_mf) / (p.eps_mf**3 * p.phi_s**2)
    A = 1.75 / (p.eps_mf**3 * p.phi_s)
    Ar = (p.dp**3) * p.rho_g * (p.rho_s - p.rho_g) * p.g / (p.mu**2)
    return B, A, Ar
def pressure_grad_ergun(p: Inputs, u: float) -> float:
    """
    Ergun equation: Resolves gradient, thearby i mean delta_p_b/L_m for a fixed bed. Ergun equation Eq. 3.6
    """
    #delta_p_fr / L_m [Pa/m] from Ergun at superficial velocity u.
    term1 = 150.0 * (1.0 - p.eps_mf) ** 2 / (p.eps_mf ** 3) * (p.mu * u) / ((p.phi_s * p.dp) ** 2)
    term2 = 1.75 * (1.0 - p.eps_mf) / (p.eps_mf ** 3) * (p.rho_g * u * u) / (p.phi_s * p.dp)
    return term1 + term2
def solve_ergun(p: Inputs):
    """
    Ergun equation: Resolves delta_p_b for a fixed bed. Ergun equation Eq. 3.6
    takes gradient, multiplies by input bed height
    puts results into a dictionary
    """
    grad = pressure_grad_ergun(p, p.u_sp)
    delta_p_b = grad * p.L_mf
    return {
        "grad_ergun": grad,
        "delta_p_b": delta_p_b,
        "u_sp": p.u_sp
    }
def weight_drag_eq(p: Inputs, coolprop):
    """
    Weight drag= weight of particles equation: Resolves gradient delta_p_b/L_m . Equation Eq. 3.17
    """
    p.rho_g, p.mu, p.L_mf = get_rho_mu_L(p, coolprop)
    grad = (1.0 - p.eps_mf) * (p.rho_s - p.rho_g) * p.g
    delta_p_b = grad * p.L_mf
    return delta_p_b
def solve_umf(p: Inputs):
    """
    Resolves the minimum fluidization speed with using the "Mitternachtsformel"(/quadratic formula to find Re_mf) according to the equation Eq. 3.19
    This combines the Ergun equation Eq. 3.6 and the equation Eq. 3.17. The latter equates drag force to the weight of the particles.
    Puts the results into a dictionary.
    """

    B, A, Ar = calc_ab_ar(p)
    Re_mf = (-B + np.sqrt(B**2 + 4.0*A*Ar)) / (2.0*A)
    u_mf = Re_mf * p.mu / (p.rho_g * p.dp)
    # Pressure gradients at u_mf
    grad_e = pressure_grad_ergun(p, u_mf)
    delta_p_b = grad_e * p.L_mf
    delta_p_d = delta_p_b * 0.3
    return {
        "Re_mf": Re_mf,
        "u_mf": u_mf,
        "grad_ergun": grad_e,  # Pa/m
        "delta_p_b": delta_p_b,  # Pa
        "delta_p_d": delta_p_d  # Pa
    }
def load_params(name) -> dict:
    """
    Returns/loads parameter input file(json file)
    """

    #Debugging
    if (name ==""):
        name = "test"

    script_dir = Path(__file__).parent
    p = script_dir / "configs" / name
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    with p.open(encoding="utf-8") as f:
        return json.load(f)
def save_results_ergun(fname, rows_list, multiple_plots_of, values):
    """
    Takes in the
    1. Name of the resulting files
    2. Results("rows_list"): Each plot is a List of Dictionaries, the contents of each dictionary are always for each dictionary one x,y datapoint
                    rows_list is a list of such lists, therefore a list of plots, therefore the structure is as follows:
                    list of plots, where each plot is-> list of dictionaries, where each entry->one x,y pair
                    list(list(dictionary("x_value":x, "y_value": y))) of course the keys have different names
    3. secondary sweep variable = "multiple_plots_of". this is the variable of which when it is not "No" the outer list is a sweep of.
        This result in multiple printed tables
    4. "values" is the values of the secondary sweep variable that causes multiple tables.

    So this function prints the results into a csv file(next to each other) and a txt file(below each other)
    """
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    csvname = results_dir / (fname + ".csv")
    with open(csvname, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = []
        if (multiple_plots_of != "No"):
            for val in values:
                header += [
                    f"u[m/s]_{multiple_plots_of}={val}",
                    f"grad_ergun[Pa/m]_{multiple_plots_of}={val}",
                    f"delta_p[Pa]_{multiple_plots_of}={val}",
                    ""  # spacer column
                ]
        else:
            header = [
                f"u[m/s]",
                f"grad_ergun[Pa/m]",
                f"delta_p[Pa]",
            ]

        writer.writerow(header)
        # Write rows side by side
        for i in range(len(rows_list[0])):
            row_data = []
            for rows in rows_list:
                r = rows[i]
                row_data += [
                    f"{r['u_sp']:.6g}",
                    f"{r['grad_ergun']:.6g}",
                    f"{r['delta_p_b']:.6g}",
                    ""
                ]
            writer.writerow(row_data)
    print(f"Ergun Δp–u results written to {csvname}")

    txtname = results_dir / (fname + ".txt")
    header = ["u[m/s]", "grad_ergun[Pa/m]", "delta_p[Pa]"]
    with open(txtname, "w") as f:
        f.write("Ergun delta_p–u results\n\n")
        if (multiple_plots_of != "No"):
            for val, rows in zip(values, rows_list):
                f.write(f"--- {multiple_plots_of} = {val} ---\n")
                f.write("\t".join(header) + "\n")
                for r in rows:
                    fields = [
                        f"{r['u_sp']:.6g}",
                        f"{r['grad_ergun']:.6g}",
                        f"{r['delta_p_b']:.6g}",
                    ]
                    f.write("\t".join(fields) + "\n")
                f.write("\n\n")
        else:
            for rows in rows_list:
                f.write("\t".join(header) + "\n")
                for r in rows:
                    fields = [
                        f"{r['u_sp']:.6g}",
                        f"{r['grad_ergun']:.6g}",
                        f"{r['delta_p_b']:.6g}",
                    ]
                    f.write("\t".join(fields) + "\n")

    print(f"Ergun Δp–u results written to {txtname}")
def save_results_combined_eq(fname: str, plots_combined_eq, sweep_param, multiple_plots_of, values):
    """
    Takes in the
    1.Name of the resulting files
    2. Results("plots_combined_eq"): Each plot is a List of Dictionaries, the contents of each dictionary are always for each dictionary one x,y datapoint
                    rows_list is a list of such lists, therefore a list of plots, therefore the structure is as follows:
                    list of plots, where each plot is-> list of dictionaries, where each entry->one x,y pair
                    list(list(dictionary("x_value":x, "y_value": y))) of course the keys have different names
    3. secondary sweep variable = "multiple_plots_of". this is the variable of which when it is not "No" the outer list is a sweep of.
        This result in multiple printed tables
    4. "values" is the values of the secondary sweep variable that causes multiple tables.

    So this function prints the results into a csv file(next to each other) and a txt file(below each other)
    last entry of each table(each primary sweep) is at minumum fluidization conditions(not in order, really just appended to the end)
    """
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    csvname = results_dir / (fname + ".csv")
    with open(csvname, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        #  header
        header = []
        if (multiple_plots_of == "No"):
            header = [
                f"{sweep_param}",
                f"Re_mf",
                f"u_mf",
                f"grad_ergun[Pa/m]",
                f"delta_p_b[Pa]",
                f"delta_p_d[Pa]"
            ]
        else:
            for val in values:
                header += [
                    f"{sweep_param}_{multiple_plots_of}={val}",
                    f"Re_mf_{multiple_plots_of}={val}",
                    f"u_mf_{multiple_plots_of}={val}",
                    f"grad_ergun[Pa/m]_{multiple_plots_of}={val}",
                    f"delta_p_b[Pa]_{multiple_plots_of}={val}",
                    f"delta_p_d[Pa]_{multiple_plots_of}={val}",
                    ""  # spacer column
                ]
        writer.writerow(header)
        # data rows
        for i in range(len(plots_combined_eq[0])):
            row_data = []
            for rows in plots_combined_eq:
                r = rows[i]
                row_data += [
                    f"{r[sweep_param]:.6g}",
                    f"{r['Re_mf']:.6g}",
                    f"{r['u_mf']:.6g}",
                    f"{r['grad_ergun']:.6g}",
                    f"{r['delta_p_b']:.6g}",
                    f"{r['delta_p_d']:.6g}",
                    ""
                ]
            writer.writerow(row_data)
    print(f"\nSweep results written to {csvname}")

    txtname = results_dir / (fname + ".txt")
    with open(txtname, "w") as f:
        f.write(f"Sweep results for {sweep_param}\n\n")
        header = [
            sweep_param, "Re_mf", "u_mf",
            "grad_ergun[Pa/m]", "delta_p_b[Pa]", "delta_p_d[Pa]"
        ]
        if (multiple_plots_of != "No"):
            for val, rows in zip(values, plots_combined_eq):
                f.write(f"--- {multiple_plots_of} = {val} ---\n")
                f.write("\t".join(header) + "\n")

                for r in rows:
                    fields = [
                        f"{r[sweep_param]:.6g}",
                        f"{r['Re_mf']:.6g}",
                        f"{r['u_mf']:.6g}",
                        f"{r['grad_ergun']:.6g}",
                        f"{r['delta_p_b']:.6g}",
                        f"{r['delta_p_d']:.6g}"
                    ]
                    f.write("\t".join(fields) + "\n")
        else:
            for rows in plots_combined_eq:
                f.write("\t".join(header) + "\n")
                for r in rows:
                    fields = [
                        f"{r[sweep_param]:.6g}",
                        f"{r['Re_mf']:.6g}",
                        f"{r['u_mf']:.6g}",
                        f"{r['grad_ergun']:.6g}",
                        f"{r['delta_p_b']:.6g}",
                        f"{r['delta_p_d']:.6g}"
                    ]
                    f.write("\t".join(fields) + "\n")
def plot_sweep_combined_eq(rows_list, sweep_key: str, plotting: str, fname: str, multiple_plots_of, values):
    """
    rows_list = results
    Results("rows_list"): Each plot is a List of Dictionaries, the contents of each dictionary are always for each dictionary one x,y datapoint
                    rows_list is a list of such lists, therefore a list of plots, therefore the structure is as follows:
                    list of plots, where each plot is-> list of dictionaries, where each entry->one x,y pair
                    list(list(dictionary("x_value":x, "y_value": y))) of course the keys have different names
    sweep_key = the primary sweep variable that we want as x axis
    plotting = our y axis
    fname = output file name
    multiple_plots_of = secondary sweep variable that causes multiple plots in the same graphic if it is not "no"
    values = vlaues of the secondary sweep

    plots the results
    """
    if plotting == "u_mf":
        ylabel  = "u_mf [m/s]"
    elif plotting == "Re_mf":
        ylabel  = "Re_mf [-]"
    elif plotting == "pressure_grad":
        ylabel  = "Pressure gradient [Pa/m]"
    elif plotting == "grad_weight":
        ylabel  = "Weight balance gradient [Pa/m]"
    elif plotting == "grad_ergun":
        ylabel  = "Ergun gradient [Pa/m]"
    elif plotting == "delta_p_b":
        ylabel  = "Δp_d [Pa]"
    elif plotting == "delta_p_d":
        ylabel  = "Δp_d [Pa]"
    else:
        ylabel  = plotting

    xlabel = f"{sweep_key} {get_si_unit(sweep_key)}"
    plt.figure(figsize=(6, 4))
    for i, rows in enumerate(rows_list):
        x = [r[sweep_key] for r in rows]
        y = [r[plotting] for r in rows]
        if (multiple_plots_of != "No"):
            unit =  get_si_unit(sweep_key)
            plt.plot(x, y, marker="o", label=f"Dataset {multiple_plots_of} = {values[i]} {unit}")
        else:
            plt.plot(x, y, marker="o")
    if (multiple_plots_of != "No"):
        plt.legend()
    title = f"Sweep of {xlabel} vs {ylabel}"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    outpath = results_dir / f"{fname}_{plotting}.png"
    plt.savefig(outpath, dpi=300)   # high-resolution PNG
    print(f"Plot saved to {outpath}")
def plot_sweep_ergun(rows_list, values, multiple_plots_of: str, fname: str):
    """
    rows_list = results(already explained the structure of that variable in )
    Results("rows_list"): Each plot is a List of Dictionaries, the contents of each dictionary are always for each dictionary one x,y datapoint
                rows_list is a list of such lists, therefore a list of plots, therefore the structure is as follows:
                list of plots, where each plot is-> list of dictionaries, where each entry->one x,y pair
                list(list(dictionary("x_value":x, "y_value": y))) of course the keys have different names
    sweep_key = doesn't exist because always u
    plotting = doesn't exist because always delta_p_b
    fname = output file name
    multiple_plots_of = secondary sweep variable that causes multiple plots in the same graphic if it is not "no"
    values = vlaues of the secondary sweep

    1. plots the results
    2. if the last entry of a plot(if one plot does, then all do) contains the key-value pair "__marker": True in its last dictionary,
        then this dictionary represents the x,y values at minimum fluidization conditions as found by function solve_umf.
        This causes this last entry to be resorted into the right position so that the plot contains the minimum fluidization at the right position.
        This is the edge in the plot if the values after minimum fluidizing conditions are flattened according to equation Eq. 3.17
    """
    xlabel, ylabel, title = "u_sp [m/s]", "Δp_b [Pa]", "Ergun pressure drop vs superficial velocity"
    plt.figure(figsize=(6, 4))
    plt.grid(True)
    ax = plt.gca()
    for i, rows in enumerate(rows_list):

        marker = rows[-1] if (rows and rows[-1].get("__marker")) else None
        base = rows[:-1] if marker is not None else rows

        if marker is not None:
            u_vals = [r["u_sp"] for r in base]  # already monotone
            j = bisect_left(u_vals, float(marker["u_sp"]))  # insertion index
            pts = base[:j] + [marker] + base[j:]
        else:
            pts = base

        x = [r["u_sp"] for r in pts]
        y = [r["delta_p_b"] for r in pts]

        if multiple_plots_of != "No":
            unit = get_si_unit(multiple_plots_of)
            ax.plot(x, y, marker="o", label=f"Dataset {multiple_plots_of} = {values[i]} {unit}")
        else:
            ax.plot(x, y, marker="o")

        if marker is not None:
            ax.scatter(marker["u_sp"], marker["delta_p_b"], s=60, zorder=5, color="k")

    if (multiple_plots_of != "No"):
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()

    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)
    outpath = results_dir / f"{fname}.png"
    plt.savefig(outpath, dpi=300)   # high-resolution PNG
    print(f"Plot saved to {outpath}")
def run_combined_eq(fname: str, sweep_keys, data: dict, coolprop):
    """
    top level controlling function calling all other functions if we are not interested in u_sp as primary sweep variable
    calls functions that return results, save results and plot results
    """

    outname = fname.rsplit(".", 1)[0] + "_combined_eq"

    #choosing plotting y value
    available = ["u_mf", "Re_mf", "grad_ergun", "delta_p_b", "delta_p_d"]
    print("\nAvailable quantities to plot:")
    for i, key in enumerate(available, 1):
        print(f" {i}. {key}")
    plotting_choice = input("Select quantity to plot (type key): ")
    if plotting_choice not in available:
        print(f"Invalid choice. Defaulting to 'u_mf'.")
        plotting_choice = "u_mf"
    #no secondary sweep variable = 1 plot
    if (len(sweep_keys) == 1):
        sweep_key = sweep_keys[0]

        rows = run_sweep(data, sweep_key, coolprop, 0)
        save_results_combined_eq(outname, [rows], sweep_key, "No", [])
        plot_sweep_combined_eq([rows], sweep_key, plotting_choice, outname, "No", [])
    else:
        # there is a secondary sweep variable = multiple plots
        print("\nMultiple sweeps detected:")
        print("Do you want to have 1. " + sweep_keys[0] +" or 2. "+ sweep_keys[1] +" on the x axis? (The other sweep variable will result in multiple plots on the same diagram.)")
        sweep_choice = int(input("Type 1/2: "))
        #choosing which is primary and secondary
        if (sweep_choice == 1):
            sweep_key = sweep_keys[0]
            multiple_of = sweep_keys[1]
        else:
            sweep_key = sweep_keys[1]
            multiple_of = sweep_keys[0]
        v = data[multiple_of]
        #arranging the multiple plots by replacing secondary variable range with specific values
        if isinstance(v, dict) and {"min", "max", "step"} <= v.keys():
            # Case 1: dictionary specifying min/max/step
            values = np.arange(v["min"], v["max"] + v["step"], v["step"])
        elif isinstance(v, (list, tuple, np.ndarray)):
            # Case 2: explicit list or array of numeric values
            values = np.array(v, dtype=float)
        else:
            raise TypeError(f"Unsupported sweep specification for '{multiple_of}': {type(v)}")
        rows_list = []
        for val in values:
            d = copy.deepcopy(data)
            d[multiple_of] = val
            plot_i = run_sweep(d, sweep_key, coolprop, 0)
            rows_list.append(plot_i)
        plot_sweep_combined_eq(rows_list, sweep_key, plotting_choice, outname, multiple_of, values)
        save_results_combined_eq(outname, rows_list, sweep_key, multiple_of, values)

    print("Hint: The results in the each output list of each output file are ordered with the lowest sweep parameter value at the top")
def get_Velocity_from_volume_flow(p, flowType):
    """
    converts volume flow to velocity based on volumeflow units and Tube diameter
    """
    factor = 1
    if (flowType == 1):
        factor = 1e-3
    elif (flowType == 2 or flowType == 4):
        factor = 1e-6
    elif (flowType == 5):
        factor = 1e-9
    p.u_sp = factor * p.u_sp / (15 * np.pi * p.column_D**2)
    return p.u_sp
def flatten_y_after_threshold(plot_i, y_target):
    """
    flattens all y-values of the plot after minimum fluidization conditions according to function solve_umf and equation Eq. 3.19 to be constant.
    So the plot ramains the same until u_mf then it stays constant to create a picter like eg. figure 3.4
    """
    y_values = np.array([r["delta_p_b"] for r in plot_i])

    #find the first index where y >= y_target
    condition = y_values >= y_target
    if np.any(condition):
        idx = np.argmax(condition)
        y_values[idx:] = y_target
        # Write modified values back into the list of dicts
        for i, r in enumerate(plot_i):
            r["delta_p_b"] = y_values[i]

    return plot_i
def run_sweep(data: dict, sweep_key: str, coolprop, flowType):
    """
    data = input file loaded into a dictionary
    -> gets converted to a struct
    sweep_key = primary sweep variable
    -> input sweep parameters get converted to a list of sweep values, so that then the y values can be calculated for each point in values
    -> according to this in the for loop for each x-datapoint in values the data dictionary is cast into a struct with the right integer x-value in the earlier position of the sweep conditions
    -> the functions calculating the results for each x-value then use the updated struct as input.
    coolprop = if coolprop or ideal gas law should be used
    flowType = if volume flow and if so, what type

    creates the x-y pairs for each plot where each datapoint(x,y pair) is one dictionary
    """
    v = data[sweep_key]
    if isinstance(v, dict) and {"min", "max", "step"} <= v.keys():
        # case 1: dictionary specifying min/max/step
        values = np.arange(v["min"], v["max"] + v["step"], v["step"])
    else: #case 2: explicit list or array of numeric values
        values = np.array(v, dtype=float)
    rows = []
    for val in values:
        d = copy.deepcopy(data)
        d[sweep_key] = val
        p = Inputs(**d)
        p.rho_g, p.mu, p.L_mf = get_rho_mu_L(p, coolprop)
        if (sweep_key == "u_sp"):
            if (flowType):
                p.u_sp = get_Velocity_from_volume_flow(p, flowType)
            res = solve_ergun(p)
        else:
            res = solve_umf(p)
        rows.append({sweep_key: val, **res})
    return rows
def get_rho_mu_L(p, coolprop):
    """
    resolves rho_g and mu if they are  not given in the input file
    """
    if p.rho_g is None or p.mu is None:
        rho_g_calc, mu_calc = get_gas_properties_from_input(asdict(p), coolprop)

        if p.rho_g is None:
            p.rho_g = rho_g_calc
        if p.mu is None:
            p.mu = mu_calc
    if p.L_mf is None:
        p.L_mf = p.bed_mass/((np.pi/4) * p.column_D**2 * (1.0 - p.eps_mf) * (p.rho_s))
    return p.rho_g, p.mu, p.L_mf
def run_ergun_curve(fname: str, sweep_key: str, data: dict, coolprop):
    """
    top level controlling function calling all other functions if we are interested in u_sp as primary sweep variable
    calls functions that return results, save results and plot results
    """
    #there is a secondary sweep variable= multiple plots
    volume_flow = str(input("Is the superficial velocity value of the Input file a volume flow?")).strip().lower()
    #asking if we have a volume flow

    if (volume_flow == ""):#debugging helper, to not always have to type in 1
        flowType = int(1)
    else:
        if (volume_flow != "no" and volume_flow != "n" and volume_flow != "0"):
            print("Type '1' for L/min")
            print("Type '2' for mL/min")
            print("Type '3' for m^3/min")
            print("Type '4' for cm^3/min")
            print("Type '5' for mm^3/min")
            flowType = int(input("Type an integer! "))
        else:
            flowType = int(0)
    #means that we have only one plot and no secondary sweep
    if (sweep_key == "u_sp"):
        rows = run_sweep(data, "u_sp", coolprop, flowType)
        p0 = Inputs(**copy.deepcopy(data))  # from current inputs
        umf_marker = make_umf_marker(copy.deepcopy(p0), coolprop, "No_Sweep")

        #plot that is only ergun
        plot_just_ergun = rows + [umf_marker]
        # plot that is flattened after u_mf
        weight_drag_pressure = weight_drag_eq(p0, coolprop)  # pressure of wheight=drag eq at certain sweep value
        flattened = flatten_y_after_threshold(copy.deepcopy(rows), weight_drag_pressure)
        plot_drag_weight_ergun = flattened + [umf_marker]


        outname = fname.rsplit(".", 1)[0] + "_ergun_just_ergun"
        plot_sweep_ergun([plot_just_ergun], [], "No", outname)
        save_results_ergun(outname, [plot_just_ergun], "No", [])

        outname = fname.rsplit(".", 1)[0] + "_ergun_drag_weight"
        plot_sweep_ergun([plot_drag_weight_ergun], [], "No", outname)
        save_results_ergun(outname, [plot_drag_weight_ergun], "No", [])
    else:
    #means that we have multiple plots since there is a secondary sweep variable (primary must be u_sp in this function)
        v = data[sweep_key]
        if isinstance(v, dict) and {"min", "max", "step"} <= v.keys():
            # Case 1: dictionary specifying min/max/step
            values = np.arange(v["min"], v["max"] + v["step"], v["step"])
        else: # Case 2: explicit list or array of numeric values
            values = np.array(v, dtype=float)

        plots_just_ergun = []
        plots_drag_weight_ergun = []
        # arranging the multiple plots by replacing secondary variable range with specific values
        for val in values:
            d = copy.deepcopy(data)
            d[sweep_key] = val
            p = Inputs(**d)
            plot_i = run_sweep(d, "u_sp", coolprop, flowType)   #plot of ergun with certain sweep value of e.g. diameter
            umf_marker = make_umf_marker(copy.deepcopy(p), coolprop, sweep_key)
            plots_just_ergun.append(copy.deepcopy(plot_i) + [umf_marker])
            #flattened curve
            weight_drag_pressure = weight_drag_eq(p, coolprop)  # pressure of wheight=drag eq at certain sweep value
            flattened = flatten_y_after_threshold(copy.deepcopy(plot_i), weight_drag_pressure)
            plots_drag_weight_ergun.append(flattened + [umf_marker])     #plot of ergun and pressures replaced with pressure of wheight=drag eq.

        outname = fname.rsplit(".", 1)[0] + "_ergun_just_ergun"
        plot_sweep_ergun(plots_just_ergun, values, sweep_key, outname)
        save_results_ergun(outname, plots_just_ergun, sweep_key, values)

        outname = fname.rsplit(".", 1)[0] + "_ergun_drag_weight"
        plot_sweep_ergun(plots_drag_weight_ergun, values, sweep_key, outname)
        save_results_ergun(outname, plots_drag_weight_ergun, sweep_key, values)
    print("Hint: Results u_sp@umf are always added to the last row in each output list of each output file. The rest of the rows are ordered with lowest u_sp at the top")
def make_umf_marker(p: Inputs, coolprop: str, sweep_key: str) -> dict:
    """
    returns entry for ergun equation exactly at minimum fluidization conditions using function solve_umf.
    whats returned here is then used to be appended including the marker key-value pair to the end of the entries of each plot
    the marker is here so that the plotting function can then sort this entry according tothe x values into the right place
    """
    p.rho_g, p.mu, p.L_mf = get_rho_mu_L(p, coolprop)
    res = solve_umf(p)  # uses existing correlations, already fills L, rho, mu"grad_ergun": grad_e

    if (sweep_key !="No_Sweep"):
        val = getattr(p, sweep_key)
        print(f"{sweep_key} = {val:.6g}  ->  u_sp@umf = {res['u_mf']:.6g} m/s,  Δp_b = {res['delta_p_b']:.6g} Pa")
    else:
        print(f"u_sp@umf = {res['u_mf']:.6g} m/s,  Δp_b = {res['delta_p_b']:.6g} Pa")
    return {"u_sp": res["u_mf"],
            "delta_p_b": res["delta_p_b"],
            "grad_ergun": res["grad_ergun"],
            "__marker": True
            }

def main():
    fname = input("Enter parameter file name: ")
    data = load_params(fname)
    p = Inputs(**data)

    coolprop = input("Use CoolProp or ideal-gas correlation? (c/i): ").strip().lower()

    # detect sweep parameter/-s
    sweep_keys = [
                     k for k, v in data.items()
                     if ((isinstance(v, dict) and {"min", "max", "step"} <= v.keys())  # range sweep
                or (isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v))  # explicit list of values
                or (k == "u_sp" and v is not None)
        )] or None

    if (sweep_keys == None):  # single run of finding u_mf
        p.rho_g, p.mu, p.L_mf = get_rho_mu_L(p, coolprop)
        res = solve_umf(p)
        print(f"Single run")
        for k in ["Re_mf", "u_mf", "grad_ergun", "delta_p_b", "delta_p_d"]:
            print(f"{k}: {res[k]:.6g}")
    elif (len(sweep_keys) >= 3): # too many sweeps
        print(f"Too many Sweeps")
        quit()
    elif ("u_sp" in sweep_keys):  #ergun superficial velocity sweep
        if (len(sweep_keys) == 1): #one ergun plot
            print(f"Detected sweep on superficial velocity through fixed bed")
            run_ergun_curve(fname, "u_sp", data, coolprop)
        else:                                                           #multiple ergun plots, since there is a sweep variable u_sp and also another
            sweep_key = [x for x in sweep_keys if x != "u_sp"][0]       #automatically assigns sweep_key the value of the secondary sweep value, since in the ergun plots u_sp is always primary
            run_ergun_curve(fname, sweep_key, data, coolprop)

    else: # combined eq: either single plot sweep or multiple plot sweeps (primary/secondary decided later on)
        print("\n--- Ergun Δp vs u curve mode ---")
        run_combined_eq(fname, sweep_keys, data, coolprop)

if __name__ == "__main__":
    main()
