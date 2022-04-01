# Given a model this file generates an LP according to arXiv:1502.00611
import time

import gurobipy as gp
import stormpy
from gurobipy import GRB


class LPSolver:
    def __init__(self, model, name=None):
        self.gurobi_model = None
        self.decompositions = None
        self.stormpy_model = model
        self.stormpy_model_name = name
        self.time_taken_to_populate_lp = None
        self.time_taken_to_solve_lp = None

        # Initialize variables
        self.ya_var = None
        self.xa_var = None
        self.ys_var = None

        # Initialize rewards
        self.reward_model_name = list(self.stormpy_model.reward_models.keys())[0]
        self.reward_model = self.stormpy_model.reward_models[self.reward_model_name]
        self.has_state_rewards = self.reward_model.has_state_rewards
        self.has_state_action_rewards = self.reward_model.has_state_action_rewards

        self.state_rewards = self.reward_model.state_rewards if self.has_state_rewards else None
        self.state_action_rewards = self.reward_model.state_action_rewards if self.has_state_action_rewards else None

    def get_state_reward(self, state_id):
        return self.state_rewards[state_id] if self.has_state_rewards else 0

    def get_state_action_reward(self, index):
        return self.state_action_rewards[index] if self.has_state_action_rewards else 0

    def solve_mean_payoff(self):
        start_time = time.time()
        self.create_gurobi_model()
        self.init_variables()
        self.populate_lp()
        self.time_taken_to_populate_lp = time.time() - start_time
        start_time = time.time()
        result = self.solve_lp()
        self.time_taken_to_solve_lp = time.time() - start_time
        return result

    def create_gurobi_model(self):
        self.gurobi_model = gp.Model("mdp_lp")
        self.decompositions = stormpy.get_maximal_end_components(self.stormpy_model)

        # For larger models gurobi takes more time, because it also performs Crossover.
        # So we always use GRB.METHOD_BARRIER and disable Crossover.
        self.gurobi_model.setParam("Method", GRB.METHOD_BARRIER)
        self.gurobi_model.setParam("Crossover", 0)

    def init_variables(self):
        ya_indices, ya_names = [], []
        xa_indices, xa_names = [], []
        ys_indices, ys_names = [], []

        for s in self.stormpy_model.states:
            # ys variables
            ys_indices.append(s.id)
            ys_names.append("ys(" + str(s.id) + ")")

            for a in s.actions:
                # ya variables
                ya_indices.append((s.id, a.id))
                ya_names.append("ya(" + str(s.id) + ", " + str(a.id) + ")")

                # xa variables
                xa_indices.append((s.id, a.id))
                xa_names.append("xa(" + str(s.id) + ", " + str(a.id) + ")")

        self.ya_var = self.gurobi_model.addVars(ya_indices, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                                name=ya_names)
        self.xa_var = self.gurobi_model.addVars(xa_indices, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=xa_names)
        self.ys_var = self.gurobi_model.addVars(ys_indices, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=ys_names)

    def populate_lp(self):
        self.add_constraint_1_4_and_objective_function()
        self.add_constraint_2_and_3()

    # Visits each transition once and populates constraints 1, 4 and objective functions
    # At last adds the constraints and objective function to gurobi model
    def add_constraint_1_4_and_objective_function(self):
        lhs_constraint_1_exprs = [gp.LinExpr() for _ in range(self.stormpy_model.nr_states)]
        rhs_constraint_1_exprs = [gp.LinExpr() for _ in range(self.stormpy_model.nr_states)]
        lhs_constraint_4_exprs = [gp.LinExpr() for _ in range(self.stormpy_model.nr_states)]
        rhs_constraint_4_exprs = [gp.LinExpr() for _ in range(self.stormpy_model.nr_states)]

        # variables for objective functions
        obj_linear_expr = gp.LinExpr()
        actions_counter = 0
        obj_vars = []
        obj_coeffs = []

        for state in self.stormpy_model.states:
            # Add 1 if initial state for constraint 1
            if state.id in self.stormpy_model.initial_states:
                lhs_constraint_1_exprs[state.id].addConstant(1)

            # Add ys for constraint 1
            rhs_constraint_1_exprs[state.id].add(self.ys_var[state.id])

            state_reward = self.get_state_reward(state.id)

            for action in state.actions:
                rhs_constraint_1_exprs[state.id].add(self.ya_var[state.id, action.id], 1)
                rhs_constraint_4_exprs[state.id].add(self.xa_var[state.id, action.id], 1)

                # Update objective function variables
                state_action_reward = self.get_state_action_reward(actions_counter)
                total_reward = state_reward + state_action_reward
                obj_vars.append(self.xa_var[state.id, action.id])
                obj_coeffs.append(total_reward)
                actions_counter = actions_counter + 1

                for transition in action.transitions:
                    prob = transition.value()
                    next_state = transition.column

                    lhs_constraint_1_exprs[next_state].add(self.ya_var[state.id, action.id], prob)
                    lhs_constraint_4_exprs[next_state].add(self.xa_var[state.id, action.id], prob)

        # Set Constraint 1
        for i in range(self.stormpy_model.nr_states):
            self.gurobi_model.addLConstr(lhs_constraint_1_exprs[i], GRB.EQUAL, rhs_constraint_1_exprs[i],
                                         name="C1(" + str(i) + ")")

        # Set Constraint 4
        for i in range(self.stormpy_model.nr_states):
            self.gurobi_model.addLConstr(lhs_constraint_4_exprs[i], GRB.EQUAL, rhs_constraint_4_exprs[i],
                                         name="C4(" + str(i) + ")")

        # Set objective function
        obj_linear_expr.addTerms(obj_coeffs, obj_vars)
        self.gurobi_model.setObjective(obj_linear_expr, GRB.MAXIMIZE)

    def add_constraint_2_and_3(self):
        constraint_2_expr = gp.LinExpr()
        mec_index = -1
        matrix = self.stormpy_model.transition_matrix
        for mec in self.decompositions:
            mec_index = mec_index + 1

            constraint_3_lhs_expr = gp.LinExpr()
            constraint_3_rhs_expr = gp.LinExpr()
            for state_id, choices in mec:
                constraint_2_expr.add(self.ys_var[state_id], 1)
                constraint_3_lhs_expr.add(self.ys_var[state_id], 1)

                offset = matrix.get_row_group_start(state_id)
                for choice_id in choices:
                    state = self.stormpy_model.states[state_id]
                    action = state.actions[choice_id - offset]
                    constraint_3_rhs_expr.add(self.xa_var[state_id, action.id], 1)

            self.gurobi_model.addLConstr(constraint_3_lhs_expr, GRB.EQUAL, constraint_3_rhs_expr,
                                         name="C3(" + str(mec_index) + ")")

        self.gurobi_model.addLConstr(constraint_2_expr, GRB.EQUAL, 1, name="C2")

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

    def solve_lp(self):
        self.gurobi_model.optimize()
        if self.gurobi_model.status == GRB.OPTIMAL:
            print("Optimal Solution Found")
            print(round(self.gurobi_model.ObjVal, 4))
            return round(self.gurobi_model.ObjVal, 4)
        else:
            print("No Solution Found")
            return -1

    def write_lp(self):
        gp.Model.write(self.gurobi_model, "model.lp")

    def for_each_incoming_transition(self, state, consumer):
        for s in self.stormpy_model.states:
            for a in s.actions:
                for t in a.transitions:
                    if t.column == state.id:
                        consumer(s, a, t.value(), t.column)

    def get_time_taken_to_populate_lp(self):
        return self.time_taken_to_populate_lp

    def get_time_taken_to_solve_lp(self):
        return self.time_taken_to_solve_lp
