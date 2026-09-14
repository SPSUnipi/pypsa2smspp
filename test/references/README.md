# references

The independent references the stochastic conversions are checked against, and
the generators of the instances they are checked on. They are scripts, not
tests: each answers one question with a number, and the number is what a claim
about the converter rests on.

The instances themselves are written to and read from `test/output/mssb_tree`,
which is generated and stays out of the repository; these scripts create it if
it is not there.

| script | what it answers |
|---|---|
| `gen_tree_instance.py` | builds the two-level tree, climate outside and demand inside |
| `gen_two_stage_design.py` | the same tree with a second decision stage, every expandable technology split in a root part and a later one carrying a premium |
| `gen_resilient_tree.py` | the resilient tree instances |
| `emit_two_stage.py` | converts the two-decision tree to SMS++ |
| `emit_design_cost_outside.py` | converts it with the design cost stated outside the units |
| `bounds_two_stage.py` | the two values the optimum has to sit between: everything decided at the root, and everything decided once the climate is known |
| `extensive_two_stage.py` | the exact optimum, the tree written out as one flat program with the sharing stated by hand |
| `crosscheck_tssb.py`, `run_*.py`, `solve_flat.py` | run one form and report what it gives |

## What they have said so far

On `pypsa_2stage_c3_d3_b1_t1`, the two-decision tree:

```
everything at the root, what a two-stage problem can say   2275376.14
the exact optimum, extensive_two_stage.py                  2222993.75
SMS++ on the file emit_two_stage.py writes                 2.22299e+06  (6 digits)
everything once the climate is known, wait-and-see         2202994.58
```

The optimum sits strictly between the two bounds, which says the second
decision stage is there and is worth 52382, and it agrees with what SMS++
makes of the emitted file, which says the file states the tree we mean.

The instance these numbers come from is the one written after the fix
described below: an earlier one had no `p_max_pu` at all, and on it the same
three scripts gave 229591.79, 221972.59 and 219432.86.

## The one bug these references have caught so far

`gen_two_stage_design.py` took the list of the solar Generator **after**
`set_scenarios()`. On a stochastic network the index is a MultiIndex, so the
element is a tuple and `"solar" in g` stops being a substring test and becomes
an equality one: the list came out empty and the availability profile was never
written. The climate then acted on the demand alone, and `renewable_maxpower`,
which the conversion is asked to make stochastic, had nothing to vary.

It is worth knowing how it surfaced, because the reference did its job and was
not believed: `extensive_two_stage.py`, written with the profile, disagreed with
the instance by a factor of ten. The first reading was that the reference was
wrong, and the profile was taken out of it to make the two agree. Only the
second reading looked at the instance.
