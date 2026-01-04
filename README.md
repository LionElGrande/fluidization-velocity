# Minimum Fluidization Velocity Framework

Python based computational framework for predicting minimum fluidization velocity and pressure drop in gas solid fluidized beds using Ergun type correlations.

The code supports single point calculations as well as one dimensional and two dimensional parameter sweeps and exports numerical results and plots.

---

## Features and Outputs

The user may define up to two independent sweep variables. Each sweep variable can be specified either as a discrete list of values or as a numerical range defined by minimum value, maximum value, and step size.

If neither sweep variable corresponds to the superficial gas velocity usp and no value for usp is provided, the program automatically computes the following quantities at minimum fluidization:

- Reynolds number Re  
- Superficial velocity u_sp  
- Pressure drop across the bed Δp_b  
- Pressure gradient across the bed Δp_b / L_mf  
- Pressure drop across the filter Δp_fr  

These quantities are calculated using the Ergun equation together with the definitions of the Reynolds and Archimedes numbers and the quadratic solution for u_sp.

If no sweep variables are defined, the results are printed to the console.

If one sweep variable is defined, a user selected output quantity from the list above is plotted against it.

If two sweep variables are defined, one variable is used as the x axis while the second produces multiple curves in the same diagram.

All plotted data is additionally saved to both CSV and TXT files.

---

If one of the sweep variables corresponds to the superficial velocity usp, the program computes and plots the pressure drop Δp_b as a function of usp.

The point of minimum fluidization is identified as the intersection between the Ergun equation and the constant pressure drop regime and is visually marked in the plot. Depending on the specified velocity range, only one regime may be visible.

A second sweep variable can again be used to generate multiple curves in a single diagram. The corresponding umf value is appended to the output data for each curve and all results are saved to CSV and TXT files.

---

## Handling of Physical Quantities

The framework provides flexible handling of gas flow and bed properties.

The superficial velocity u_sp can either be provided directly or computed from a volumetric flow rate V_dot and the column diameter D using the column cross sectional area.

Gas density rho_g and viscosity mu may be specified directly. If they are not provided, they are estimated from temperature and pressure using either the CoolProp library or ideal gas behavior combined with Sutherland’s law.

If no gas type or temperature is specified, atmospheric conditions are assumed with T = 298.15 K and p = 101325 Pa.

If the bed height Lmf is not defined, it is automatically computed from the total bed mass, particle density, void fraction, and column diameter.

---

## Inputs

All inputs are provided through a single JSON file, which is interpreted directly as a Python dictionary.

Required and optional input parameters are:

- dp: particle diameter [m]  
- rho_s: particle density [kg/m3]  
- rho_g: gas density [kg/m3], optional  
- mu: gas viscosity [Pa s], optional  
- eps_mf: void fraction at minimum fluidization  
- phi_s: particle sphericity  
- g: gravitational acceleration [m/s2]  
- gas_name: gas type  
- T: gas temperature [K], optional, default 298.15 K  
- P: gas pressure [Pa], optional, default 101325 Pa  
- L_mf: bed height at minimum fluidization [m], optional  
- bed_mass: total mass of particles [kg]  
- column_D: column internal diameter [m]  
- u_sp: superficial velocity or velocity range [m/s]  

---

## Usage

1. Define all physical parameters and sweep variables in the JSON input file  
2. Run the main Python script  
3. Inspect generated plots and exported CSV and TXT files  

---

## License

This project is released under the MIT License.
