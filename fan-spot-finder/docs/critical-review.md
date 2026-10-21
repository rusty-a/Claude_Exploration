# Fan Spot Finder: critical review (version 12)

Short verdict: the app is a good-looking toy that shows precise numbers from a coarse model. The air-flow core has some validation. The heat and comfort features have almost none. It is not trustworthy enough to support decisions beyond "point the fan at you".

Tags: [Certain] = I checked it. [Likely] = strong reason, not checked. [Guessing] = opinion.

## 1. The main question has a trivial answer

- For "Breeze on me", the optimizer mostly finds "put the fan close to you and point it at you". You do not need CFD to know that. [Certain]
- "Coolest air" and "Feels coolest" mostly agree with "Breeze on me". [Likely]
- So the brute-force search (about 150 to 300 runs, up to 90 s on a phone) costs a lot of compute to confirm common sense. [Certain]

## 2. Precise output from imprecise physics

- The grid has 12 to 14 cm cells in the app and 23 cm cells in the coarse optimizer pass. A 0.4 m fan is about 2 cells wide in that pass. [Certain]
- A fan is a disk of fixed velocity. It has no swirl, no real suction side and no blade wake. [Certain]
- The default turbulence factor (C = 0.01) was tuned to 4 fans from one paper (Buildings 2025). This is a fit to the data I tested against, not an independent validation. [Certain]
- The UI shows 0.1 °C and 0.01 m/s. The model error is much larger than that. [Likely]
- The "Feels like" formula matches pythermalcomfort to 0.01 K, but its input (air speed at you) can have an error of 30 % or more. Precision in the formula does not give accuracy in the result. [Likely]

## 3. Validation gaps

| Feature | Checked against | Status |
|---|---|---|
| Mass balance | In = out | Pass [Certain] |
| Air flow, isothermal | IEA Annex 20, 106 points | RMS 0.16 of supply speed [Certain] |
| Fan jet decay | Buildings 2025, 4 fans | RMS 0.06, but same data used to tune [Certain] |
| Energy balance | Hand calculation, 3 cases | Pass [Certain] |
| Warm plumes, stratification | Nothing | Not validated [Certain] |
| Window flow | Textbook formula | About 10 % high [Likely] |
| Feels like | pythermalcomfort | Formula pass; inputs not checked [Certain] |

- I did not rerun the Annex 20 and fan decay checks after the version 12 changes (opening loss, walls, heat). The earlier numbers may no longer be true. [Certain]
- I did not do a grid study. The optimizer ranking at 23 cm cells may change at finer cells. Only the top 6 get a second check. [Certain]

## 4. An open bug

- You saw 3 to 15 °C on an iPhone with default heat settings. I could not reproduce it in about 10 runs in node and Chromium. [Certain]
- In version 12 I removed the step that could cause it. That stops the symptom, but I do not know the cause. It can come back in another form. [Certain]
- I cannot test Safari or the iPhone app here. [Certain]

## 5. Heat and comfort model limits

- Walls and furniture stay at one fixed temperature. There is no sun, no radiation and no heat storage in walls. [Certain]
- The person is a heated column of air cells, not a body with a surface. [Certain]
- The heat goals compare spots after about 18 s from the same start, not at steady state. The room is still warming at that time. [Certain]
- "Feels like" uses fixed values: seated, light clothes, 50 % humidity, walls at air temperature. [Certain]

## 6. Engineering

- The whole app is one 70 KB HTML file. [Certain]
- The test scripts are in a temporary session folder. When this session ends, they are lost. There is no repository, no version history except artifact versions, and no automatic regression tests. [Certain]
- Each feature was added on top of the last one without rerunning the earlier checks. [Certain]

## 7. Process: my part

- I added each feature you asked for, and I did not push back enough on whether it added value. Heat, "Coolest air" and "Feels like" were each built in under an hour, then found to need fixes. [Certain]
- Two of my earlier claims were wrong and I had to correct them: Smagorinsky "best" from 2 points, and "30 min" of testing that was really 3 min. [Certain]

## 8. Options

1. **Keep it a fun toy.** Remove the heat goals, keep "Breeze on me", round displayed numbers, and put a clear "rough model" label at the top. Small effort. [Guessing]
2. **Make it a credible engineering sketch tool.** Move it to a repository with regression tests (energy, Annex 20, fan decay), do a grid study, compare a plume against published data, run heat goals to steady state, and find the iPhone bug. Large effort. [Guessing]
3. **Go back to the first idea (a surrogate model).** Train a fast model on solver runs to replace the brute-force search. Warning: a surrogate copies the errors of the solver, so option 2 must come first. [Likely]

My recommendation: option 1 now, unless you have a real use where the numbers matter. Then option 2 before option 3. [Guessing]
