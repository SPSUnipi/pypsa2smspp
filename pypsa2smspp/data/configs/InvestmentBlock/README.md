# The configuration of an InvestmentBlock

What `Transformation` hands to SMS++ when it writes an InvestmentBlock and
`configfile` is left at "auto". It is the configuration of the BundleSolver
2.0, i.e. the one whose master is a MasterProblemBlock solved by a
:MILPSolver, and it differs from the template of pySMSpp in the three things
that decide whether these instances are solved at all:

- `BSPar.txt` names the master through `strMPBSolverCfg`, and the inner Block
  through `strInnerBSC`;
- `BSCfg1.txt`, the inner Block, is on GUROBI and sets
  `intHomogeneousDirection 1`: a design that can starve the inner Block is
  answered with a feasibility cut, which is read off the unbounded dual
  direction, and that takes a Solver that returns one in the form - A' y.
  `BSCfg2.txt` is the same one on HiGHS, which does not, hence an instance
  whose inner Block the design can make infeasible ends with no answer;
- `MPBCfg.txt` leaves the presolve of the master at its default: with it off
  the master of a design over several extendable lines fails outright.
