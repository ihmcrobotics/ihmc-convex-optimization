package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DenseMatrix64F;

public interface ActiveSetQPSolverWithInactiveVariablesInterface extends ActiveSetQPSolver
{
   void setActiveVariables(DenseMatrix64F activeVariables);

   void setVariableActive(int variableIndex);

   void setVariableInactive(int variableIndex);

   void setAllVariablesActive();
}
