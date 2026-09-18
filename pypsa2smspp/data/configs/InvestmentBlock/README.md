# The configuration of an InvestmentBlock

What `Transformation` hands to SMS++ when it writes an InvestmentBlock and
`configfile` is left at "auto". `BSPar.txt` is the one that is used, and it
differs from the template of pySMSpp in two things, both of which decide
whether these instances are solved at all:

- the master of the bundle is the OSI one, `intMPName 15`, and not QPPenalty:
  a feasibility cut reaches the master as a constraint, which QPPenalty
  refuses outright, and on an instance where the design of an asset sits at 0
  QPPenalty also stops on a point four times the optimum;
- `BSCfg.txt`, the inner Block, says what a feasibility cut takes:
  `intHomogeneousDirection 1` and a Solver that returns the unbounded dual
  direction, i.e. CPLEX or GUROBI. It is left on HiGHS, which is what every
  build has, so an instance whose inner Block the design can starve, a
  sector-coupled network with an extendable generator for one, ends with no
  answer until that line is uncommented.

A configuration is read as a whole: a parameter the SMS++ at hand does not
know makes its whole ComputeConfig fail to load, and it is dropped in silence,
taking with it the ones that would have been read. This is why the default
here is the one the released SMS++ reads, the BundleSolver being 1.0 there.

`test/instance_generator.py` passes the 2.0 one explicitly, being run against
a build of the 2.0.

`BSPar_2.0.txt` is the same configuration for the BundleSolver 2.0, whose
master is a MasterProblemBlock solved by a :MILPSolver (`strMPBSolverCfg` ->
`MPBCfg.txt`, `strInnerBSC` -> `BSCfg1.txt` on GUROBI). It solves every
instance of the set, and the presolve of its master is left at its default:
with it off the master of a design over several extendable lines ends in
"Bundle::FormD: unrecoverable MP failure".
