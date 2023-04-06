"""code template"""

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def main():
    # Create the model with edges specified as tuples (parent, child)
    dentist_model = BayesianModel([('Cavity', 'Toothache'),
                                   ('Cavity', 'Catch')])
    # Create tabular CPDs, values has to be 2-D array
    cpd_cav = TabularCPD('Cavity', 2, [[0.2], [0.8]])
    cpd_too = TabularCPD('Toothache', 2, [[0.6, 0.1],
                                          [0.4, 0.9]],
                         evidence=['Cavity'], evidence_card=[2])
    cpd_cat = TabularCPD('Catch', 2, [[0.9, 0.2],
                                      [0.1, 0.8]],
                         evidence=['Cavity'], evidence_card=[2])
    # Add CPDs to model
    dentist_model.add_cpds(cpd_cav, cpd_too, cpd_cat)

    print('Check model :', dentist_model.check_model())

    print('Independencies:\n', dentist_model.get_independencies())

    # Initialize inference algorithm
    dentist_infer = VariableElimination(dentist_model)

    # Some exemple queries
    q = dentist_infer.query(['Toothache'])
    print('P(Toothache) =\n', q)

    q = dentist_infer.query(['Cavity'])
    print('P(Cavity) =\n', q)

    q = dentist_infer.query(['Toothache'], evidence={'Cavity': 0})
    print('P(Toothache | cavity) =\n', q)

    q = dentist_infer.query(['Toothache'], evidence={'Cavity': 1})
    print('P(Toothache | ~cavity) =\n', q)

    # Zad 3.1
    q = dentist_infer.query(['Cavity'], evidence={'Toothache': 0, 'Catch': 1})
    print('P(Cavity|toothache,~cavity) =\n', q)

    # Zad 3.2
    q2 = dentist_infer.query(['Toothache', 'Catch', 'Cavity'])
    q4 = dentist_infer.query(['Toothache', 'Catch'])
    # 2 pierwsze wiersze z odpowiedzi to wiersze, gdzie Cavity(0), czyli prawda.
    # 2 ostatnie wiersze z odpowiedzi to wiersze, gdzie wiersze w "print" to Cavity (1), czyli Cavity falsz
    print('P(Cavity | Toothache, Catch)  =\n', q2 / q4)

    # Zad 3.3

if __name__ == '__main__':
    main()