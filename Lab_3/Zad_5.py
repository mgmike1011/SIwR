from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Model markowa I rzędzu, rekurencyjny system do oszacowania Rt potrzeba tylko Rt-1 i ut
# Potrzebujemy macierzy tranzycji i modelu sensora
# Oszacowanie jaki jest stan robota w danej chwili mając obserwacje
# P(Rt+1|u1:t+1) = P(Rt+1 | U1:t,U1:t+1) = alpha*P(Ut+1|Rt+1,U1:t)P(Rt+1|U1:t)=
# = alpha*P(Ut+1|Rt+1)P(Rt+1|U1:t) = alpha*P(Ut+1|Rt+1)*Suma(
# alpha*P(Ut+1|Rt+1) - model sensora
# P(Rt+1|Rt) - model ruchu
# P(Rt|U1:t) - to co w poprzednim stanie
# Jesteśmy w każdej chwili czasowej estymować

bayesNet = BayesianModel([('Rain_t_1', 'Rain_t'), ('Rain_t', 'Umbrela_t')])
cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[0.4], [0.6]])
cpd_rain_t = TabularCPD('Rain_t', 2, [[0.7, 0.3], [0.3, 0.7]], ['Rain_t_1'], [2])
cpd_umbrela_t = TabularCPD('Umbrela_t', 2, [[0.8, 0.1], [0.2, 0.9]], ['Rain_t'], [2])
bayesNet.add_cpds(cpd_rain_t_1, cpd_rain_t, cpd_umbrela_t)
print('Check model :', bayesNet.check_model())
bayes_infer = VariableElimination(bayesNet)

# Zad 5.1
kolejne_dni = []
for i in range(5):
    if i == 2:
        q = bayes_infer.query(['Rain_t'], evidence={'Umbrela_t': 0})
        kolejne_dni.append(q.values[1])
    else:
        q = bayes_infer.query(['Rain_t'], evidence={'Umbrela_t': 1})
        kolejne_dni.append(q.values[1])
    cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
    bayesNet.add_cpds(cpd_rain_t_1)
print('P(R1 | U1) =\n', q)

# Zad 5.2
print(kolejne_dni)

# Zad 5.3
q = bayes_infer.query(['Rain_t'])
print('Dla dnia 6 =\n', q)
cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
bayesNet.add_cpds(cpd_rain_t_1)
q = bayes_infer.query(['Rain_t'])
cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
bayesNet.add_cpds(cpd_rain_t_1)
q = bayes_infer.query(['Rain_t'])
cpd_rain_t_1 = TabularCPD('Rain_t_1', 2, [[q.values[0]], [q.values[1]]])
bayesNet.add_cpds(cpd_rain_t_1)
q = bayes_infer.query(['Rain_t'])
print('Dla dnia 9 =\n', q)