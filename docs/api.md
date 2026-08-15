# API Reference

This page documents every public function in catpie. Each entry lists the
function's purpose, its arguments (with expected value ranges), its return
value and the errors it can raise.

A quick orientation:

- **Item model:** [`pi`](#catpie.irf.pi), [`ii`](#catpie.irf.ii), [`ji`](#catpie.irf.ji)
- **Estimation:** [`thetaEst`](#catpie.estimators.thetaEst), [`semTheta`](#catpie.estimators.semTheta), [`eapEst`](#catpie.eap.eapEst), [`eapSem`](#catpie.eap.eapSem), [`estimateTheta`](#catpie.estimateTheta)
- **Selection:** [`nextItem`](#catpie.selection.nextItem), [`selectNextItem`](#catpie.selectNextItem)
- **Simulation:** [`genPattern`](#catpie.simulation.genPattern), [`simulateRespondents`](#catpie.simulation.simulateRespondents), [`checkStopRule`](#catpie.simulation.checkStopRule), [`randomCAT`](#catpie.simulation.randomCAT)
- **Numerical helpers:** [`dnorm`](#catpie.math.dnorm), [`linspace`](#catpie.math.linspace), [`integrateCatR`](#catpie.math.integrateCatR), [`qnorm`](#catpie.math.qnorm), [`uniroot`](#catpie.math.uniroot), [`optimizeScalar`](#catpie.math.optimizeScalar)

For the background and the meaning of the value ranges, see
[Concepts](concepts.md). For a worked example, see [Quickstart](quickstart.md).

## Item response functions

The building blocks: probability of a correct answer, item information, and
the weighted-likelihood helper. See [Concepts, sections 3–4](concepts.md).

::: catpie.irf.pi

::: catpie.irf.ii

::: catpie.irf.ji

## Ability estimation

Estimate a person's ability (theta) and its standard error from their answers.
See [Concepts, section 5](concepts.md).

::: catpie.eap.eapEst

::: catpie.eap.eapSem

::: catpie.estimators.thetaEst

::: catpie.estimators.semTheta

::: catpie.estimateTheta

## Item selection

Pick the next, most informative item. See [Concepts, section 6](concepts.md).

::: catpie.selection.nextItem

::: catpie.selectNextItem

## Simulation

Generate fake response patterns, check stopping rules, and run whole adaptive
sessions offline. See [Concepts, section 7](concepts.md).

::: catpie.simulation.genPattern

::: catpie.simulation.simulateRespondents

::: catpie.simulation.checkStopRule

::: catpie.simulation.randomCAT

## Result types

The named tuples that catpie returns, with their fields.

::: catpie.irf.PiResult

::: catpie.irf.IiResult

::: catpie.irf.JiResult

::: catpie.selection.NextItemResult

::: catpie.simulation.StopRuleResult

::: catpie.simulation.RandomCatResult

::: catpie.ThetaResult

::: catpie.math.OptimizeResult

## Numerical helpers

Small functions that mirror R's base functions. You rarely need them directly,
but they are part of the public API for completeness and parity with catR.

::: catpie.math.dnorm

::: catpie.math.linspace

::: catpie.math.integrateCatR

::: catpie.math.qnorm

::: catpie.math.uniroot

::: catpie.math.optimizeScalar
