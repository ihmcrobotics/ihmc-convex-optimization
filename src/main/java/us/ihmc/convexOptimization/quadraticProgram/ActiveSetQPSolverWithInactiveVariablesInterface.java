package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;

/**
 * Extension of a {@link ActiveSetQPSolver} allowing to explicitly disable or enable components of
 * {@code x} when solving the problem.
 * <p>
 * Instead of resizing the problem's matrices, the user can simply indicate which component is to be
 * ignore during the next {@link #solve(DMatrixRMaj)}, cutting computation time to a similar
 * magnitude compared to the problem being resized in a compact form. This has the advantage from
 * the user's perspective to preserve a fixed indexing for each variable component without
 * increasing computation time.
 * </p>
 */
public interface ActiveSetQPSolverWithInactiveVariablesInterface extends ActiveSetQPSolver
{
   /**
    * Sets a N-by-1 vector filled with either 1s or 0s to indicate respectively the active and inactive
    * variables for the next {@link #solve(DMatrixRMaj)}.
    * 
    * @param activeVariables the N-by-1 vector used to identify the active and inactive variables. Not
    *                        modified.
    */
   void setActiveVariables(DMatrix activeVariables);

   /**
    * Marks the i<up>th</sup> component of {@code x} has an active variable, i.e. a solution for that
    * component is expected after the next call to {@link #solve(DMatrixRMaj)}.
    * 
    * @param variableIndex the index of the variable to be marked as active.
    */
   void setVariableActive(int variableIndex);

   /**
    * Marks the i<up>th</sup> component of {@code x} has an inactive variable, i.e. the variable can be
    * ignored for the next call of {@link #solve(DMatrixRMaj)}.
    * 
    * @param variableIndex the index of the variable to be marked as inactive.
    */
   void setVariableInactive(int variableIndex);

   /**
    * Marks all the components of {@code x} as active, i.e. the entire set variables is expected to be
    * solved for during the next call of {@link #solve(DMatrixRMaj)}.
    */
   void setAllVariablesActive();
}
