# Quick Start

To install the package, please refer to [Installation](installation.md).

In the following, we provide a quick example on how to add execute a simple example with pysmspp.
The problem that follows optimizes the capacity expansion a single thermal generator.

First, we need to import the required packages:

```python
import pypsa
import pandas as pd
import pypsa2smspp
```

Second, we create a simple PyPSA network with 2 buses, 1 line, 2 loads and 1 generator.

```python
n = pypsa.Network()
n.set_snapshots(pd.date_range("2024-01-01T00:00", "2024-01-01T23:00", freq="h"))

# Add carriers
n.add("Carrier", "AC")

# Add buses
n_buses = 2
for b in range(n_buses):
    n.add("Bus", f"bus{b}", carrier="AC")

# Add lines in a radial topology using bidirectional links
n_lines = n_buses - 1
for l in range(n_lines):
    n.add(
        "Link",
        f"line{l}",
        bus0=f"bus{l}",
        bus1=f"bus{l+1}",
        length=1,
        capital_cost=1000,
        p_min_pu=-1,
        p_nom_extendable=True,
    )

# Add a load to each bus
n_loads = n_buses
for l in range(n_loads):
    n.add("Load", f"load{l}", bus=f"bus{l}", p_set=pd.Series(100, index=n.snapshots))

# Add a generator to the first bus
n.add(
    "Generator",
    "gen0",
    bus="bus0",
    p_nom_extendable=True,
    capital_cost=1000,
    marginal_cost=1,
)
```

Third, we create a transformation object to convert the PyPSA network into an SMS++ model. The transformation object has several configuration options, but we will use the default settings for this example.

```python
tran = pypsa2smspp.Transformation()
```

Finally, the network is automatically converted into a SMS++ object, optimized, and the reconstructed back from the results of SMS++ using the following code. The variable `n_smspp` contains the optimized PyPSA network repopulated of the solution from SMS++:

```python
n_smspp = tran.run(n, verbose=False)
n_smspp
```
