# Fan Spot Finder

A browser app that sketches the air flow in a room and searches for the best spot for a pedestal fan.
Open `app/index.html` in a browser. It needs no build step.

**Status: a rough model.** Use it to compare fan spots, not to promise numbers. Read
[docs/critical-review.md](docs/critical-review.md) before you trust a result.

## Model

- 3D staggered (MAC) grid, about 40 cells along the longest side of the room.
- Semi-Lagrangian advection for air speed, first-order upwind flux transport for temperature, SOR pressure projection.
- Turbulence: eddy viscosity. The default is the Chen and Xu formula with the factor cut to 0.01 (tuned to fan data).
- A fan is a disk of cells that pushes air at a fixed speed.
- Heat: heat sources on boxes and on the person, Boussinesq buoyancy, walls and furniture at a fixed temperature (3 W/m²K).
- Openings: zero pressure outside, with a sharp-edged loss (discharge coefficient 0.6).
- "Feels like": air temperature minus the ASHRAE 55 cooling effect, from the Gagge two-node SET model
  (port of pythermalcomfort). Fixed: 1.1 met, 0.5 clo, 50 % humidity.

## Checks

The tests load the solver directly from `app/index.html`, so they check the code that the app runs.

```
node --test tests/*.test.cjs        # mass, energy and comfort checks, about 2 to 3 min
node validation/annex20.cjs         # IEA Annex 20 room, 106 measured points, limit RMS 0.20 of U0
node validation/fan-decay.cjs       # fan jet decay against Buildings 2025, 15(3):308, limit RMS 0.10
```

| Check | Result (version 12) |
|---|---|
| Sealed room, 500 W, adiabatic walls | Mean rise equal to Q·t/(ρ·cp·V) within 0.01 K |
| Sealed room, walls at 20 °C | Walls take 500 W at steady state |
| Window, 3 °C outside | Heat out of the window equal to heat from the walls within 3 % |
| Default room | Air never colder than the coldest input |
| Cooling effect | Equal to pythermalcomfort within 0.02 K |
| Annex 20 | RMS 0.157 of U0 (23 cm grid) |
| Fan jet decay | RMS 0.055 (9.4 cm grid); the same data was used to tune the model |

Not validated: warm plumes and stratification, the optimizer ranking on a finer grid.

## Known open issue

On an iPhone, the default room once showed 3 to 15 °C after 9.6 min. It was not reproduced in Node or Chromium.
Version 12 removed the step that could cause it, but the cause is not known.

## References

- Nielsen, P. V. (1990). Specification of a two-dimensional test case. IEA Annex 20. Aalborg University.
- Buildings 2025, 15(3), 308 (MDPI): measured air speed in front of household fans.
- ASHRAE Standard 55-2020, cooling effect of elevated air speed.
- Tartarini and Schiavon (2020), pythermalcomfort, SoftwareX 12.
