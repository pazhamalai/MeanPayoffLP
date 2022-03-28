# Given a model this file generates an LP according to arXiv:1502.00611

import gurobipy as gp
import stormpy
from gurobipy import GRB


class LPGenerator:
    def __init__(self, model, name=None):
        self.gurobi_model = None
        self.decompositions = None
        self.stormpy_model = model
        self.stormpy_model_name = name

        self.ya_var = None
        self.xa_var = None
        self.ys_var = None

    def generate(self):
        self.create_gurobi_model()
        print("Model Created")
        self.init_variables()
        print("Variables Initialized")
        self.add_constraints()
        print("Added Constraints")
        self.add_objective_function()
        print("Created LP")
        # self.write_lp()
        self.solve_lp()

    def create_gurobi_model(self):
        self.gurobi_model = gp.Model("mdp_lp")
        self.decompositions = stormpy.get_maximal_end_components(self.stormpy_model)

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
            names.append("ys(" + str(s.id) + ")")

        self.ys_var = self.gurobi_model.addVars(indices, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=names)

    def add_constraints(self):
        self.add_constraint_1_and_4()
        print("Constraint 1 and 4 added")
        self.add_constraint_2()
        print("Constraint 2 added")
        self.add_constraint_3()
        print("Constraint 3 added")

    def add_constraint_1_and_4(self):
        lhs_constraint_1_exprs = [gp.LinExpr() for _ in range(self.stormpy_model.nr_states)]
        rhs_constraint_1_exprs = [gp.LinExpr() for _ in range(self.stormpy_model.nr_states)]
        lhs_constraint_4_exprs = [gp.LinExpr() for _ in range(self.stormpy_model.nr_states)]
        rhs_constraint_4_exprs = [gp.LinExpr() for _ in range(self.stormpy_model.nr_states)]

        for state in self.stormpy_model.states:
            # Add 1 if initial state
            if state.id in self.stormpy_model.initial_states:
                lhs_constraint_1_exprs[state.id].addConstant(1)

            # Add ys
            rhs_constraint_1_exprs[state.id].add(self.ys_var[state.id])

            for action in state.actions:
                rhs_constraint_1_exprs[state.id].add(self.ya_var[state.id, action.id], 1)
                rhs_constraint_4_exprs[state.id].add(self.xa_var[state.id, action.id], 1)
                for transition in action.transitions:
                    prob = transition.value()
                    next_state = transition.column

                    lhs_constraint_1_exprs[next_state].add(self.ya_var[state.id, action.id], prob)
                    lhs_constraint_4_exprs[next_state].add(self.xa_var[state.id, action.id], prob)

        for i in range(self.stormpy_model.nr_states):
            self.gurobi_model.addLConstr(lhs_constraint_1_exprs[i], GRB.EQUAL, rhs_constraint_1_exprs[i],
                                         name="C1(" + str(i) + ")")

        for i in range(self.stormpy_model.nr_states):
            self.gurobi_model.addLConstr(lhs_constraint_4_exprs[i], GRB.EQUAL, rhs_constraint_4_exprs[i],
                                         name="C4(" + str(i) + ")")

    def add_constraint_2(self):
        linear_expr = gp.LinExpr()
        for mec in self.decompositions:
            for state_id, choices in mec:
                linear_expr.add(self.ys_var[state_id], 1)

        self.gurobi_model.addLConstr(linear_expr, GRB.EQUAL, 1.0, name="C2")

    def add_constraint_3(self):
        counter = 0
        for mec in self.decompositions:
            self.add_constraint_3_for_mec(mec, counter + 1)
            counter = counter + 1

    def add_constraint_3_for_mec(self, mec, mec_index):
        lhs_constraint = self.get_lhs_constraint_3_for_mec(mec)
        rhs_constraint = self.get_rhs_constraint_3_for_mec(mec)
        self.gurobi_model.addLConstr(lhs_constraint, GRB.EQUAL, rhs_constraint, name="C3(" + str(mec_index) + ")")

    def get_lhs_constraint_3_for_mec(self, mec):
        linear_constraint = gp.LinExpr()
        for state_id, choices in mec:
            linear_constraint.add(self.ys_var[state_id], 1)

        return linear_constraint

    def get_rhs_constraint_3_for_mec(self, mec):
        linear_constraint = gp.LinExpr()
        matrix = self.stormpy_model.transition_matrix
        for state_id, choices in mec:
            offset = matrix.get_row_group_start(state_id)
            for choice_id in choices:
                state = self.stormpy_model.states[state_id]
                action = state.actions[choice_id - offset]
                linear_constraint.add(self.xa_var[state_id, action.id], 1)

        return linear_constraint

    def add_objective_function(self):
        reward_model_name = list(self.stormpy_model.reward_models.keys())[0]
        reward_model = self.stormpy_model.reward_models[reward_model_name]
        has_state_rewards = reward_model.has_state_rewards
        has_state_action_rewards = reward_model.has_state_action_rewards

        if has_state_rewards:
            state_rewards = reward_model.state_rewards
        else:
            state_rewards = None

        if has_state_action_rewards:
            state_action_rewards = reward_model.state_action_rewards
        else:
            state_action_rewards = None

        linear_expr = gp.LinExpr()
        counter = 0
        xa_vars = []
        coeffs = []
        print("Created objective variables")
        for state in self.stormpy_model.states:
            if has_state_rewards:
                state_reward = state_rewards[state.id]
            else:
                state_reward = 0

            for action in state.actions:
                if has_state_action_rewards:
                    state_action_reward = state_action_rewards[counter]
                else:
                    state_action_reward = 0
                total_reward = state_reward + state_action_reward
                xa_vars.append(self.xa_var[state.id, action.id])
                coeffs.append(total_reward)
                counter = counter + 1

        print("Computed values")

        linear_expr.addTerms(coeffs, xa_vars)
        self.gurobi_model.setObjective(linear_expr, GRB.MAXIMIZE)

    def solve_lp(self):
        self.gurobi_model.optimize()
        if self.gurobi_model.status == GRB.OPTIMAL:
            print("Optimal Solution Found")
            print(self.gurobi_model.ObjVal)
        else:
            print("No Solution Found")

    def write_lp(self):
        gp.Model.write(self.gurobi_model, "model.lp")

    def for_each_incoming_transition(self, state, consumer):
        for s in self.stormpy_model.states:
            for a in s.actions:
                for t in a.transitions:
                    if t.column == state.id:
                        consumer(s, a, t.value(), t.column)
