from xopt import Xopt
import pytest


class TestLegacy:
    legacy_yaml = """
xopt:
  output_path: .
  verbose: true
  algorithm: cnsga
  
generator:
  name: upper_confidence_bound
  options:  
    n_steps: 2
    n_initial_samples: 50

simulation:
  name: impact_with_distgen
  evaluate: xopt.tests.test_functions.TNK.evaluate_TNK
  options:
    extra_option: null
    
vocs:
  name: LCLS cu_inj Impact-T and Disgten full optimization v9
  description: data set for 20 pc for lcls_cu_inj, 20k particles
  simulation: impact_with_distgen
  variables:

    # Distgen
    distgen:r_dist:sigma_xy:value: [0.1, 0.3]
    distgen:t_dist:length:value: [3.0, 12.0]
  linked_variables: null
  
  constants: null
    
  objectives:
    end_norm_emit_xy: MINIMIZE   
"""

    def test_legacy(self):
        X = Xopt(self.legacy_yaml)
