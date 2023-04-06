from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Zad 4.1
bayesNet = BayesianModel([('Battery', 'Radio'), ('Battery', 'Ignition'), ('Ignition', 'Starts'), ('Gas', 'Starts'), ('Starts', 'Moves')])

cpd_bat = TabularCPD('Battery', 2, [[0.3], [0.7]])
cpd_gas = TabularCPD('Gas', 2, [[0.5], [0.5]])
cpd_radio = TabularCPD('Radio', 2, [[1.0, 0.1], [0.0, 0.9]], ['Battery'], [2])
cpd_ignition = TabularCPD('Ignition', 2, [[1.0, 0.03], [0.0, 0.97]], ['Battery'], [2])
cpd_starts = TabularCPD('Starts', 2, [[1.00, 1.00, 1.00, 0.05], [0.00, 0.00, 0.00, 0.95]], ['Ignition', 'Gas'], [2, 2])
cpd_moves = TabularCPD('Moves', 2, [[1.0, 0.2], [0.0, 0.8]], ['Starts'], [2])

bayesNet.add_cpds(cpd_bat, cpd_gas, cpd_radio, cpd_ignition, cpd_starts, cpd_moves)

print('Check model :', bayesNet.check_model())
bayes_infer = VariableElimination(bayesNet)

# Zad 4.2
q = bayes_infer.query(['Starts'], evidence={'Radio': 1, 'Gas': 1})
print('P(Starts | Gas, Radio) =\n', q)

# Zad 4.3
q = bayes_infer.query(['Battery'], evidence={'Moves': 1})
print('P(Battery | Moves) =\n', q)

# Zad 4.4
bayesNet_ = BayesianModel([('Battery', 'Radio'), ('Battery', 'Ignition'), ('Ignition', 'Starts'), ('Gas', 'Starts'),
                           ('Starts', 'Moves'), ('NotIcyWeather', 'Starts'), ('Battery', 'StarterMotor'), ('StarterMotor', 'Starts')])

cpd_bat_ = TabularCPD('Battery', 2, [[0.3], [0.7]])
cpd_gas_ = TabularCPD('Gas', 2, [[0.5], [0.5]])
cpd_radio_ = TabularCPD('Radio', 2, [[1.0, 0.1], [0.0, 0.9]], ['Battery'], [2])
cpd_ignition_ = TabularCPD('Ignition', 2, [[1.0, 0.03], [0.0, 0.97]], ['Battery'], [2])
cpd_moves_ = TabularCPD('Moves', 2, [[1.0, 0.2], [0.0, 0.8]], ['Starts'], [2])
cpd_NotIcyWeather_ = TabularCPD('NotIcyWeather', 2, [[0.1], [0.9]])
cpd_StarterMotor_ = TabularCPD('StarterMotor', 2,  [[1.0, 0.05], [0.0, 0.95]], ['Battery'], [2])

# Zad 4.5
cpd_starts_ = TabularCPD('Starts', 2, [[1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 0.15], [0.00, 0.00, 0.00, 0.00, 0.00, 0.00,0.00, 0.00, 0.00,0.00, 0.00, 0.00,0.00, 0.00, 0.00, 0.85]], ['Ignition', 'Gas', 'StarterMotor', 'NotIcyWeather'], [2, 2, 2, 2]) 
bayesNet_.add_cpds(cpd_bat_, cpd_gas_, cpd_radio_, cpd_ignition_, cpd_starts_, cpd_moves_, cpd_NotIcyWeather_, cpd_StarterMotor_)
bayes_infer_ = VariableElimination(bayesNet_)

# Zad 4.6
q = bayes_infer_.query(['Radio'], evidence={'Starts': 0})
print('P(Radio | ~Start) =\n', q)
