from docplex.mp.model import *
from docplex.mp.utils import *
from docplex.util.status import JobSolveStatus
from docplex.mp.conflict_refiner import ConflictRefiner, VarUbConstraintWrapper, VarLbConstraintWrapper
from docplex.mp.relaxer import Relaxer
import time
import sys
import operator

import pandas as pd
import numpy as np
import math

import codecs
import sys

# Handle output of unicode strings
if sys.version_info[0] < 3:
    sys.stdout = codecs.getwriter('utf8')(sys.stdout)


from pandas.api.types import is_string_dtype


def helper_check_data_type(df, column, df_label, check_type):
    if not column in df.columns:
        print('Column "%s" does not exist in table "%s"' % (column, df_label))
        return False
    non_nan_values = df[column][~df[column].isnull()]
    if check_type == 'INTEGER':
        k = non_nan_values.dtype.kind
        if k != 'i':
            if k == 'f':
                non_integer_values = non_nan_values.values[np.where([not x.is_integer() for x in non_nan_values])]
                if len(non_integer_values) > 0:
                    print('Column "%s" of table "%s" contains non-integer value(s) which violates expected type: %s' % (column, df_label, non_integer_values))
                    return False
            else:
                print('Column "%s" of table "%s" is non-numeric which violates expected type: %s' % (column, df_label, non_nan_values.values))
                return False
    elif check_type == 'FLOAT' or check_type == 'NUMBER':
        non_float_values = non_nan_values.values[np.where([not isinstance(x, (int, float)) for x in non_nan_values])]
        k = non_nan_values.dtype.kind
        if not k in ['i', 'f']:
            print('Column "%s" of table "%s" contains non-float value(s) which violates expected type: %s' % (column, df_label, non_float_values))
            return False
    elif check_type == 'BOOLEAN':
        non_bool_values = non_nan_values.values[np.where([not isinstance(x, bool) for x in non_nan_values])]
        if len(non_bool_values) > 0:
            print('Column "%s" of table "%s" contains non-boolean value(s) which violates expected type: %s' % (column, df_label, non_bool_values))
            return False
    elif check_type == 'Date' or check_type == 'DateTime':
        try:
            pd.to_datetime(non_nan_values)
        except ValueError as e:
            print('Column "%s" of table "%s" cannot be converted to a DateTime : %s' % (column, df_label, str(e)))
            return False
    elif check_type == 'Time':
        try:
            pd.to_timedelta(non_nan_values)
        except ValueError as e:
            try:
                # Try appending ':00' in case seconds are not represented in time
                pd.to_timedelta(non_nan_values + ':00')
            except ValueError as e:
                print('Column "%s" of table "%s" cannot be converted to a Time : %s' % (column, df_label, str(e)))
                return False
    elif check_type == 'STRING':
        if not is_string_dtype(non_nan_values):
            print('Column "%s" of table "%s" is not of type "String"' % (column, df_label))
            return False
    else:
        raise Exception('Invalid check_type: %s' % check_type)
    return True


def helper_check_foreignKey_values(source_df, source_column, source_df_label, target_df, target_column, target_df_label):
    non_nan_values = source_df[source_column][~source_df[source_column].isnull()]
    invalid_FK_values = non_nan_values[~non_nan_values.isin(target_df[target_column])].values
    if len(invalid_FK_values) > 0:
        print('FK Column "%s" of table "%s" contains values that do not exist in PK column "%s" of target table "%s": %s' % (source_column, source_df_label, target_column, target_df_label, invalid_FK_values))
        return False
    return True


def helper_check_unique_primaryKey_values(df, key_cols, df_label):
    df_grp = df.groupby(key_cols).size()
    invalid_pk_values = df_grp[df_grp > 1].reset_index()[key_cols].values
    if len(invalid_pk_values) > 0:
        print('Non-unique values for PK of table "%s": %s' % (df_label, invalid_pk_values))
        return False
    return True


# Label constraint
def helper_add_labeled_cplex_constraint(mdl, expr, label, context=None, columns=None):
    global expr_counter
    if isinstance(expr, np.bool_):
        expr = expr.item()
    if isinstance(expr, bool):
        pass  # Adding a trivial constraint: if infeasible, docplex will raise an exception it is added to the model
    else:
        expr.name = '_L_EXPR_' + str(len(expr_to_info) + 1)
        if columns:
            ctxt = ", ".join(str(getattr(context, col)) for col in columns)
        else:
            if context:
                ctxt = context.Index if isinstance(context.Index, str) is not None else ", ".join(context.Index)
            else:
                ctxt = None
        expr_to_info[expr.name] = (label, ctxt)
    mdl.add(expr)

def helper_get_column_name_for_property(property):
    return helper_property_id_to_column_names_map.get(property, 'unknown')


def helper_get_index_names_for_type(dataframe, type):
    if not is_pandas_dataframe(dataframe):
        return None
    return [name for name in dataframe.index.names if name in helper_concept_id_to_index_names_map.get(type, [])]


helper_concept_id_to_index_names_map = {
    'customerDemand': ['id_of_CustomerDemand'],
    'cItem': ['id_of_Plants'],
    'plants': ['id_of_Plants']}
helper_property_id_to_column_names_map = {
    'customerDemand.Product': 'Product',
    'cItem.maxValueAllocation': 'Capacity',
    'plants.Cost': 'Cost',
    'plants.Capacity': 'Capacity',
    'plants.Plants': 'Plants',
    'customerDemand.Demand': 'Demand',
    'plants.Product': 'Product'}


# Data model definition for each table
# Data collection: list_of_CustomerDemand ['Product', 'Demand']
# Data collection: list_of_Plants ['Capacity', 'Product', 'Plants', 'Cost']

# Create a pandas Dataframe for each data table
list_of_CustomerDemand = inputs[u'customerDemand']
list_of_CustomerDemand = list_of_CustomerDemand[[u'Product', u'Demand']].copy()
list_of_CustomerDemand.rename(columns={u'Product': 'Product', u'Demand': 'Demand'}, inplace=True)
list_of_Plants = inputs[u'plants']
list_of_Plants = list_of_Plants[[u'Capacity', u'Product', u'Plants', u'Cost']].copy()
list_of_Plants.rename(columns={u'Capacity': 'Capacity', u'Product': 'Product', u'Plants': 'Plants', u'Cost': 'Cost'}, inplace=True)

# Perform input data checking against schema configured in Modelling Assistant along with unicity of PK values
data_check_result = True
# --- Handling data checking for table: customerDemand
data_check_result &= helper_check_data_type(list_of_CustomerDemand, 'Demand', 'customerDemand', 'NUMBER')
data_check_result &= helper_check_unique_primaryKey_values(list_of_CustomerDemand, ['Product'], 'customerDemand')
# --- Handling data checking for table: plants
data_check_result &= helper_check_data_type(list_of_Plants, 'Capacity', 'plants', 'NUMBER')
data_check_result &= helper_check_foreignKey_values(list_of_Plants, 'Product', 'plants', list_of_CustomerDemand, 'Product', 'customerDemand')
data_check_result &= helper_check_data_type(list_of_Plants, 'Plants', 'plants', 'NUMBER')
data_check_result &= helper_check_data_type(list_of_Plants, 'Cost', 'plants', 'NUMBER')
data_check_result &= helper_check_unique_primaryKey_values(list_of_Plants, ['Plants'], 'plants')
if not data_check_result:
    # Stop execution here
    raise Exception('Data checking detected errors')

# Set index when a primary key is defined
list_of_CustomerDemand.set_index('Product', inplace=True)
list_of_CustomerDemand.sort_index(inplace=True)
list_of_CustomerDemand.index.name = 'id_of_CustomerDemand'
list_of_Plants.set_index('Plants', inplace=True)
list_of_Plants.sort_index(inplace=True)
list_of_Plants.index.name = 'id_of_Plants'






def build_model():
    mdl = Model()

    # Definition of model variables
    list_of_Plants['allocationVar'] = mdl.integer_var_list(len(list_of_Plants))
    list_of_Plants['selectionVar'] = mdl.binary_var_list(len(list_of_Plants))


    # Definition of model
    # Objective cMaximizeGoalSelect-
    # Combine weighted criteria: 
    # 	cMaximizeGoalSelect cMaximizeGoalSelect 1.2{
    # 	numericExpr = decisionPath(cAllocation[plants]),
    # 	scaleFactorExpr = 1,
    # 	(static) goalFilter = null} with weight 10.0
    # 	cMinimizeGoalSelect cMinimizeGoalSelect 1.2{
    # 	numericExpr = cAllocation[plants] / plants / Cost,
    # 	scaleFactorExpr = 1,
    # 	(static) goalFilter = null} with weight 5.0
    agg_Plants_allocationVar_SG1 = mdl.sum(list_of_Plants.allocationVar)
    list_of_Plants['conditioned_Cost'] = list_of_Plants.allocationVar * list_of_Plants.Cost
    agg_Plants_conditioned_Cost_SG2 = mdl.sum(list_of_Plants.conditioned_Cost)
    
    kpis_expression_list = [
        (-1, 512.0, agg_Plants_allocationVar_SG1, 1, 0, u'total plants allocations'),
        (1, 16.0, agg_Plants_conditioned_Cost_SG2, 1, 0, u'total Cost of plants over all allocations')]
    custom_code.update_goals_list(kpis_expression_list)
    
    for _, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list:
        mdl.add_kpi(kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset), publish_name=kpi_name)
    
    mdl.minimize(sum([kpi_sign * kpi_weight * ((kpi_expr * kpi_factor) - kpi_offset) for kpi_sign, kpi_weight, kpi_expr, kpi_factor, kpi_offset, kpi_name in kpis_expression_list]))
    
    # [ST_1] Constraint : cLinkSelectionToAllocationConstraint_cIterativeRelationalConstraint
    # Synchronize selection with plants allocations
    # Label: CT_1_Synchronize_selection_with_plants_allocations
    join_Plants = list_of_Plants.reset_index().merge(list_of_Plants.reset_index(), left_on=['id_of_Plants'], right_on=['id_of_Plants'], suffixes=('', '_right')).set_index(['id_of_Plants'])
    groupbyLevels = ['id_of_Plants']
    groupby_Plants = join_Plants.allocationVar.groupby(level=groupbyLevels[0]).sum().to_frame(name='allocationVar')
    list_of_Plants['conditioned_Capacity'] = list_of_Plants.selectionVar * list_of_Plants.Capacity
    join_Plants_2 = groupby_Plants.join(list_of_Plants.conditioned_Capacity, how='inner')
    for row in join_Plants_2.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar <= row.conditioned_Capacity, u'Synchronize selection with plants allocations', row)
    
    # [ST_2] Constraint : cLinkSelectionToAllocationConstraint_cIterativeRelationalConstraint
    # Synchronize selection with plants allocations
    # Label: CT_2_Synchronize_selection_with_plants_allocations
    join_Plants = list_of_Plants.reset_index().merge(list_of_Plants.reset_index(), left_on=['id_of_Plants'], right_on=['id_of_Plants'], suffixes=('', '_right')).set_index(['id_of_Plants'])
    groupbyLevels = ['id_of_Plants']
    groupby_Plants = join_Plants.allocationVar.groupby(level=groupbyLevels[0]).sum().to_frame(name='allocationVar')
    list_of_Plants_minValueAllocationForAssignment = pd.Series([0] * len(list_of_Plants)).to_frame('minValueAllocationForAssignment').set_index(list_of_Plants.index)
    join_Plants_2 = list_of_Plants.join(list_of_Plants_minValueAllocationForAssignment.minValueAllocationForAssignment, how='inner')
    join_Plants_2['conditioned_minValueAllocationForAssignment'] = join_Plants_2.selectionVar * join_Plants_2.minValueAllocationForAssignment
    join_Plants_3 = groupby_Plants.join(join_Plants_2.conditioned_minValueAllocationForAssignment, how='inner')
    for row in join_Plants_3.itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar >= row.conditioned_minValueAllocationForAssignment, u'Synchronize selection with plants allocations', row)
    
    # [ST_3] Constraint : cIterativeRelationalConstraint_cIterativeRelationalConstraint
    # For each plants, allocation is less than or equal to Capacity
    # Label: CT_3_For_each_plants__allocation_is_less_than_or_equal_to_Capacity
    for row in list_of_Plants[list_of_Plants.Capacity.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar <= row.Capacity, u'For each plants, allocation is less than or equal to Capacity', row)
    
    # [ST_4] Constraint : cIterativeRelationalConstraint_cIterativeRelationalConstraint
    # For each customerDemand, total allocation of plants (such that plants Product is customerDemand) is less than or equal to Demand
    # Label: CT_4_For_each_customerDemand__total_allocation_of_plants__such_that_plants_Product_is_customerDemand__is_less_than_or_equal_to_Demand
    join_CustomerDemand = list_of_CustomerDemand[[]].reset_index().merge(list_of_Plants.reset_index(), left_on=['id_of_CustomerDemand'], right_on=['Product']).set_index(['id_of_CustomerDemand', 'id_of_Plants'])
    groupbyLevels = ['id_of_CustomerDemand']
    groupby_CustomerDemand = join_CustomerDemand.allocationVar.groupby(level=groupbyLevels).sum().to_frame()
    join_CustomerDemand_2 = groupby_CustomerDemand.join(list_of_CustomerDemand.Demand, how='inner')
    for row in join_CustomerDemand_2[join_CustomerDemand_2.Demand.notnull()].itertuples(index=True):
        helper_add_labeled_cplex_constraint(mdl, row.allocationVar <= row.Demand, u'For each customerDemand, total allocation of plants (such that plants Product is customerDemand) is less than or equal to Demand', row)


    return mdl


def solve_model(mdl):
    mdl.parameters.timelimit = 120
    # Call to custom code to update parameters value
    custom_code.update_solver_params(mdl.parameters)
    # Update parameters value according to environment variables definition
    cplex_param_env_prefix = 'ma.cplex.'
    cplex_params = [name.qualified_name for name in mdl.parameters.generate_params()]
    for param in cplex_params:
        env_param = cplex_param_env_prefix + param
        param_value = get_environment().get_parameter(env_param)
        if param_value:
            # Updating parameter value
            print("Updated value for parameter %s = %s" % (param, param_value))
            parameters = mdl.parameters
            for p in param.split('.')[1:]:
                parameters = parameters.__getattribute__(p)
            parameters.set(param_value)

    msol = mdl.solve(log_output=True)
    if not msol:
        print("!!! Solve of the model fails")
        if mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_SOLUTION or mdl.get_solve_status() == JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION:
            crefiner = ConflictRefiner()
            conflicts = crefiner.refine_conflict(model, log_output=True)
            export_conflicts(conflicts)
            
    print('Solve status: %s' % mdl.get_solve_status())
    if mdl.get_solve_status() == JobSolveStatus.UNKNOWN:
        print('UNKNOWN cause: %s' % mdl.get_solve_details().status)
    mdl.report()
    return msol


expr_to_info = {}


def export_conflicts(conflicts):
    # Display conflicts in console
    print('Conflict set:')
    list_of_conflicts = pd.DataFrame(columns=['constraint', 'context', 'detail'])
    for conflict, index in zip(conflicts, range(len(conflicts))):
        st = conflict.status
        ct = conflict.element
        label, context = expr_to_info.get(conflict.name, ('N/A', conflict.name))
        label_type = type(conflict.element)
        if isinstance(conflict.element, VarLbConstraintWrapper) \
                or isinstance(conflict.element, VarUbConstraintWrapper):
            label = 'Upper/lower bound conflict for variable: {}'.format(conflict.element._var)
            context = 'Decision variable definition'
            ct = conflict.element.get_constraint()

        # Print conflict information in console
        print("Conflict involving constraint: %s, \tfor: %s -> %s" % (label, context, ct))
        list_of_conflicts = list_of_conflicts.append({'constraint': label, 'context': str(context), 'detail': ct},
                                                     ignore_index=True)

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    #VGG#with output_lock:
    if outputs : #VGG#
        outputs['list_of_conflicts'] = list_of_conflicts


def export_solution(msol):
    start_time = time.time()
    mdl = msol.model
    list_of_Plants_solution = pd.DataFrame(index=list_of_Plants.index)
    list_of_Plants_solution['allocationVar'] = msol.get_values(list_of_Plants.allocationVar.values)
    list_of_Plants_solution = list_of_Plants_solution.round({'allocationVar': 2})
    list_of_Plants_solution['selectionVar'] = msol.get_values(list_of_Plants.selectionVar.values)
    plantsAllocation = pd.DataFrame(index=list_of_Plants.index)
    
    # Adding extra columns based on Solution Schema
    plantsAllocation['plants allocation decision'] = list_of_Plants_solution['allocationVar']
    plantsAllocation['plants selection decision'] = list_of_Plants_solution['selectionVar']
    plantsAllocation['plants Capacity'] = list_of_Plants['Capacity']
    plantsAllocation['plants Product'] = list_of_Plants['Product']
    plantsAllocation['plants Cost'] = list_of_Plants['Cost']
    list_of_Plants_minValueAllocationForAssignment = pd.Series([0] * len(list_of_Plants)).to_frame('minValueAllocationForAssignment').set_index(list_of_Plants.index)
    join_Plants = list_of_Plants.join(list_of_Plants_minValueAllocationForAssignment.minValueAllocationForAssignment, how='inner')
    plantsAllocation['plants minValueAllocationForAssignment'] = join_Plants['minValueAllocationForAssignment']
    

    # Update of the ``outputs`` dict must take the 'Lock' to make this action atomic,
    # in case the job is aborted
    global output_lock
    #VGG#with output_lock:
    if output_lock: #VGG#
        outputs['plantsAllocation'] = plantsAllocation[['plants allocation decision', 'plants selection decision', 'plants Capacity', 'plants Product', 'plants Cost', 'plants minValueAllocationForAssignment']].reset_index().rename(columns= {'id_of_Plants': 'plants'})
        custom_code.post_process_solution(msol, outputs)

    elapsed_time = time.time() - start_time
    print('solution export done in ' + str(elapsed_time) + ' secs')
    return


# Import custom code definition if module exists
try:
    from custom_code import CustomCode
    custom_code = CustomCode(globals())
except ImportError:
    # Create a dummy anonymous object for custom_code
    custom_code = type('', (object,), {'preprocess': (lambda *args: None),
                                       'update_goals_list': (lambda *args: None),
                                       'update_model': (lambda *args: None),
                                       'update_solver_params': (lambda *args: None),
                                       'post_process_solution': (lambda *args: None)})()

# Custom pre-process
custom_code.preprocess()

print('* building wado model')
start_time = time.time()
model = build_model()

# Model customization
custom_code.update_model(model)

elapsed_time = time.time() - start_time
print('model building done in ' + str(elapsed_time) + ' secs')

print('* running wado model')
start_time = time.time()
msol = solve_model(model)
elapsed_time = time.time() - start_time
print('model solve done in ' + str(elapsed_time) + ' secs')
if msol:
    export_solution(msol)
