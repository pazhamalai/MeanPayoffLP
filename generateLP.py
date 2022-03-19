# Given a model this file generates an LP according to arXiv:1502.00611

# mdp_model_global = None
import gurobipy as gp
from gurobipy import GRB


class LPGenerator:
    def __init__(self, model, name=None):
        self.gurobi_model = None
        self.stormpy_model = model
        self.stormpy_model_name = name

        self.ya_var = None
        self.xa_var = None
        self.ys_var = None

    def generate(self):
        self.create_gurobi_model()
        self.init_variables()
        self.add_constraints()
        # self.add_objective_function()
        # self.solve_lp()

    def create_gurobi_model(self):
        self.gurobi_model = gp.Model("mdp_lp")

    def init_variables(self):
        self.init_ya_variable()
        self.init_xa_variable()
        self.init_ys_variable()

    def init_ya_variable(self):
        indices = []
        names = []
        for s in self.stormpy_model.states:
            for a in s.actions:
                indices.append((s.id, a.id))
                names.append("ya(" + str(s.id) + ", " + str(a.id) + ")")

        self.ya_var = self.gurobi_model.addVars(indices, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=names)

    def init_xa_variable(self):
        indices = []
        names = []
        for s in self.stormpy_model.states:
            for a in s.actions:
                indices.append((s.id, a.id))
                names.append("xa(" + str(s.id) + ", " + str(a.id) + ")")

        self.xa_var = self.gurobi_model.addVars(indices, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=names)

    def init_ys_variable(self):
        indices = []
        names = []

        for s in self.stormpy_model.states:
            indices.append(s.id)
            names.append("xs(" + str(s.id) + ")")

        self.ys_var = self.gurobi_model.addVars(indices, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=names)

    def add_constraints(self):
        self.add_constraint_1()
        self.add_constraint_2()
        self.add_constraint_3()
        self.add_constraint_4()

    def add_constraint_1(self):
        for s in self.stormpy_model.states:
            self.add_constraint_1_for_state(s)

    def add_constraint_1_for_state(self, state):
        lhs_expr = self.get_constraint_1_lhs_for_state(state)
        rhs_expr = self.get_constraint_1_rhs_for_state(state)
        self.gurobi_model.addLConstr(lhs_expr, GRB.EQUAL, rhs_expr, name="C1(" + str(state.id) + ")")
        pass

    def get_constraint_1_lhs_for_state(self, state):
        linear_expr = gp.LinExpr()

        if state.id in self.stormpy_model.initial_states:
            linear_expr.addConstant(1)

        self.for_each_incoming_transition(state, lambda source_state, action, prob, target_id: linear_expr.add(
            self.ya_var[source_state.id, action.id], prob))

        return linear_expr

    def get_constraint_1_rhs_for_state(self, state):
        linear_expr = gp.LinExpr()
        for a in state.actions:
            linear_expr.add(self.ya_var[state.id, a.id], 1)

        linear_expr.add(self.ys_var[state.id], 1)
        return linear_expr

    def add_constraint_2(self):
        pass

    def add_constraint_3(self):
        pass

    def add_constraint_4(self):
        pass

    def for_each_incoming_transition(self, state, consumer):
        for s in self.stormpy_model.states:
            for a in s.actions:
                for t in a.transitions:
                    if t.column == state:
                        consumer.consume(s, a, t.value(), t.column)
