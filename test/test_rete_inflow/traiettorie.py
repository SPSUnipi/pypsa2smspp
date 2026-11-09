"""Copia SDDPBlock_inflow.nc4 sostituendo la matrice Scenarios con le 27
traiettorie (scenario scelto in ciascuno degli stadi 1-3; stadio 0 comune)."""
import itertools, numpy as np, netCDF4 as nc

def copy_group(src, dst, replace=None, path=""):
    for a in src.ncattrs():
        dst.setncattr(a, src.getncattr(a))
    for name, dim in src.dimensions.items():
        size = None if dim.isunlimited() else len(dim)
        if replace and (path, name) in replace["dims"]:
            size = replace["dims"][(path, name)]
        dst.createDimension(name, size)
    for name, v in src.variables.items():
        fv = v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
        w = dst.createVariable(name, v.datatype, v.dimensions, fill_value=fv)
        for a in v.ncattrs():
            if a != "_FillValue":
                w.setncattr(a, v.getncattr(a))
        if replace and (path, name) in replace["vars"]:
            w[:] = replace["vars"][(path, name)]
        elif v.dimensions:
            w[:] = v[:]
        else:
            w[...] = v[...]
    for name, g in src.groups.items():
        copy_group(g, dst.createGroup(name), replace, path + "/" + name)

def make(src_file, dst_file, combos):
    s = nc.Dataset(src_file)
    g = s.groups["Block_0"]
    S = np.asarray(g.groups["ScenarioGenerator"]["Scenarios"][:])
    sub = len(g.dimensions["SubScenarioSize"])
    rows = []
    for c in combos:
        rows.append(np.concatenate([S[c[t], t * sub:(t + 1) * sub] for t in range(len(c))]))
    rows = np.array(rows)
    base = "/Block_0/ScenarioGenerator"
    rep = {"dims": {(base, "NumberScenarios"): len(rows)},
           "vars": {(base, "Scenarios"): rows}}
    d = nc.Dataset(dst_file, "w", format="NETCDF4")
    copy_group(s, d, rep)
    d.close(); s.close()

if __name__ == "__main__":
    combos = [(0,) + c for c in itertools.product(range(3), repeat=3)]
    make("smspp/SDDPBlock_inflow.nc4", "smspp/SDDPBlock_traj.nc4", combos)
    print(len(combos), "traiettorie")
