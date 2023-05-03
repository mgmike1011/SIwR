"""
code template
"""
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import FactorGraph
import numpy as np

G = FactorGraph()
G.add_nodes_from(['Rain_t_1', 'Rain_t', 'Umbrela_t'])
cpd_rain_t_1 = DiscreteFactor(['Rain_t_1'], [2], [0.4, 0.6])
cpd_rain_t = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], [0.7, 0.3, 0.3, 0.7])
cpd_umbrela_t = DiscreteFactor(['Umbrela_t', 'Rain_t' ], [2, 2], [0.8, 0.1, 0.2, 0.9])
G.add_factors(cpd_rain_t_1)
G.add_factors(cpd_rain_t)
G.add_factors(cpd_umbrela_t)
G.add_edge('Rain_t_1', cpd_rain_t_1)
G.add_edge('Rain_t_1', cpd_rain_t)
G.add_edge('Rain_t', cpd_rain_t)
G.add_edge('Rain_t', cpd_umbrela_t)
G.add_edge('Umbrela_t', cpd_umbrela_t)

bp = BeliefPropagation(G)

'''
Zad 1
'''
print("################## ZAD 1 #################")
kolejne_dni = []
q = bp.query(['Rain_t_1'])
print('Dla dnia 0 =\n', q)
kolejne_dni.append(q.values[1])

for i in range(5):
    if i == 2:
        G = FactorGraph()
        G.add_nodes_from(['Rain_t_1', 'Rain_t', 'Umbrella_t'])
        phi1 = DiscreteFactor(['Rain_t_1'], [2], values=np.array([q.values[0], q.values[1]]))
        phi2 = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], values=np.array([0.7, 0.3, 0.3, 0.7]))
        phi3 = DiscreteFactor(['Umbrella_t', 'Rain_t'], [2, 2], values=np.array([0.8, 0.1, 0.2, 0.9]))

        G.add_factors(phi1, phi2, phi3)

        # G.add_edges_from([('Rain_t_1', phi1), ('Rain_t', phi2), ('Rain_t_1', phi2), ('Umbrella_t', phi3), ('Rain_t', phi3)])
        G.add_edges_from([('Rain_t_1', phi1), ('Rain_t_1', phi2), ('Rain_t', phi2), ('Rain_t', phi3), ('Umbrella_t', phi3)])
        bp = BeliefPropagation(G)
        q = bp.query(['Rain_t'], evidence={'Umbrella_t': 0})

        kolejne_dni.append(q.values[1])
    else:
        G = FactorGraph()
        G.add_nodes_from(['Rain_t_1', 'Rain_t', 'Umbrella_t'])
        phi1 = DiscreteFactor(['Rain_t_1'], [2], values=np.array([q.values[0], q.values[1]]))
        phi2 = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], values=np.array([0.7, 0.3, 0.3, 0.7]))
        phi3 = DiscreteFactor(['Umbrella_t', 'Rain_t'], [2, 2], values=np.array([0.8, 0.1, 0.2, 0.9]))

        G.add_factors(phi1, phi2, phi3)

        # G.add_edges_from([('Rain_t_1', phi1), ('Rain_t', phi2), ('Rain_t_1', phi2), ('Umbrella_t', phi3), ('Rain_t', phi3)])
        G.add_edges_from([('Rain_t_1', phi1), ('Rain_t_1', phi2), ('Rain_t', phi2), ('Rain_t', phi3), ('Umbrella_t', phi3)])
        bp = BeliefPropagation(G)
        q = bp.query(['Rain_t'], evidence={'Umbrella_t': 1})

        kolejne_dni.append(q.values[1])

print(kolejne_dni)

'''
Zad 2
'''
print("################## ZAD 2 #################")
G = FactorGraph()
G.add_nodes_from(['Rain_t_1', 'Rain_t', 'Umbrela_t'])
cpd_rain_t_1 = DiscreteFactor(['Rain_t_1'], [2], [0.25, 0.75])
cpd_rain_t = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], [0.7, 0.3, 0.3, 0.7])
cpd_umbrela_t = DiscreteFactor(['Umbrela_t', 'Rain_t' ], [2, 2], [0.8, 0.1, 0.2, 0.9])
G.add_factors(cpd_rain_t_1)
G.add_factors(cpd_rain_t)
G.add_factors(cpd_umbrela_t)
G.add_edge('Rain_t_1', cpd_rain_t_1)
G.add_edge('Rain_t_1', cpd_rain_t)
G.add_edge('Rain_t', cpd_rain_t)
G.add_edge('Rain_t', cpd_umbrela_t)
G.add_edge('Umbrela_t', cpd_umbrela_t)

bp = BeliefPropagation(G)

kolejne_dni = []
q = bp.query(['Rain_t_1'])
print('Dla dnia 0 =\n', q)
kolejne_dni.append(q.values[1])

for i in range(4, 0, -1):
    if i == 2:
        G = FactorGraph()
        G.add_nodes_from(['Rain_t_1', 'Rain_t', 'Umbrella_t'])
        phi1 = DiscreteFactor(['Rain_t_1'], [2], values=np.array([q.values[0], q.values[1]]))
        phi2 = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], values=np.array([0.7, 0.3, 0.3, 0.7]))
        phi3 = DiscreteFactor(['Umbrella_t', 'Rain_t'], [2, 2], values=np.array([0.8, 0.1, 0.2, 0.9]))

        G.add_factors(phi1, phi2, phi3)

        G.add_edges_from([('Rain_t_1', phi1), ('Rain_t_1', phi2), ('Rain_t', phi2), ('Rain_t', phi3), ('Umbrella_t', phi3)])
        bp = BeliefPropagation(G)
        q = bp.query(['Rain_t'], evidence={'Umbrella_t': 0})

        kolejne_dni.append(q.values[1])
    else:
        G = FactorGraph()
        G.add_nodes_from(['Rain_t_1', 'Rain_t', 'Umbrella_t'])
        phi1 = DiscreteFactor(['Rain_t_1'], [2], values=np.array([q.values[0], q.values[1]]))
        phi2 = DiscreteFactor(['Rain_t', 'Rain_t_1'], [2, 2], values=np.array([0.7, 0.3, 0.3, 0.7]))
        phi3 = DiscreteFactor(['Umbrella_t', 'Rain_t'], [2, 2], values=np.array([0.8, 0.1, 0.2, 0.9]))

        G.add_factors(phi1, phi2, phi3)

        G.add_edges_from([('Rain_t_1', phi1), ('Rain_t_1', phi2), ('Rain_t', phi2), ('Rain_t', phi3), ('Umbrella_t', phi3)])
        bp = BeliefPropagation(G)
        q = bp.query(['Rain_t'], evidence={'Umbrella_t': 1})

        kolejne_dni.append(q.values[1])

print(kolejne_dni)