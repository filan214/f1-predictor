"""Live inference — forecast an upcoming race from the trained model.

Turns the trained 2010-2024 model into a real pre-race forecaster: given an
upcoming race's circuit, date and starting grid, it assembles a leakage-free
feature row per driver (carrying each driver's most recent form/ELO/standings
forward) and predicts the finishing order, with win / podium / points
probabilities.

This package only *reads* the trained artefacts and the feature matrix; it never
modifies the modelling or feature-engineering code.
"""
