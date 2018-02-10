from experiments import comparisons as cmp
from jdot.jdot_nn import JDOT_NN
from jdot.jdot_svm import JDOT_SVM
from loorls.loorsl_model import LooRLS


def main():
    cmp.compare_classifiers([JDOT_SVM()], ('mars', 'supernova'))
    pass


if __name__=="__main__":
    main()