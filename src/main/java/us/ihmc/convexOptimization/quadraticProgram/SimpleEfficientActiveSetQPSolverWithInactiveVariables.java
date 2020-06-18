package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import us.ihmc.commons.MathTools;
import us.ihmc.matrixlib.MatrixTools;
import us.ihmc.matrixlib.NativeCommonOps;

public class SimpleEfficientActiveSetQPSolverWithInactiveVariables extends SimpleEfficientActiveSetQPSolver
      implements ActiveSetQPSolverWithInactiveVariablesInterface
{
   private final DMatrixRMaj originalQuadraticCostQMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalQuadraticCostQVector = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj originalLinearEqualityConstraintsAMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalLinearEqualityConstraintsBVector = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj originalLinearInequalityConstraintsCMatrixO = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalLinearInequalityConstraintsDVectorO = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj originalVariableLowerBounds = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalVariableUpperBounds = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj activeVariables = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj activeVariableSolution = new DMatrixRMaj(0, 0);

   private void setMatricesFromOriginal()
   {
      quadraticCostQMatrix.set(originalQuadraticCostQMatrix);
      quadraticCostQVector.set(originalQuadraticCostQVector);

      linearEqualityConstraintsAMatrix.set(originalLinearEqualityConstraintsAMatrix);
      linearEqualityConstraintsBVector.set(originalLinearEqualityConstraintsBVector);

      linearInequalityConstraintsCMatrixO.set(originalLinearInequalityConstraintsCMatrixO);
      linearInequalityConstraintsDVectorO.set(originalLinearInequalityConstraintsDVectorO);

      variableLowerBounds.set(originalVariableLowerBounds);
      variableUpperBounds.set(originalVariableUpperBounds);
   }

   private void removeInactiveVariables()
   {
      setMatricesFromOriginal();

      for (int variableIndex = activeVariables.getNumRows() - 1; variableIndex >= 0; variableIndex--)
      {
         if (activeVariables.get(variableIndex) == 1.0)
            continue;

         MatrixTools.removeRow(quadraticCostQMatrix, variableIndex);
         MatrixTools.removeColumn(quadraticCostQMatrix, variableIndex);

         MatrixTools.removeRow(quadraticCostQVector, variableIndex);

         if (linearEqualityConstraintsAMatrix.getNumElements() > 0)
            MatrixTools.removeColumn(linearEqualityConstraintsAMatrix, variableIndex);
         if (linearInequalityConstraintsCMatrixO.getNumElements() > 0)
            MatrixTools.removeColumn(linearInequalityConstraintsCMatrixO, variableIndex);

         if (variableLowerBounds.getNumElements() > 0)
            MatrixTools.removeRow(variableLowerBounds, variableIndex);
         if (variableUpperBounds.getNumElements() > 0)
            MatrixTools.removeRow(variableUpperBounds, variableIndex);
      }

      int numVars = quadraticCostQMatrix.getNumRows();
      if (linearEqualityConstraintsAMatrix.getNumElements() == 0)
         linearEqualityConstraintsAMatrix.reshape(0, numVars);
      if (linearInequalityConstraintsCMatrixO.getNumElements() == 0)
         linearInequalityConstraintsCMatrixO.reshape(0, numVars);

      removeZeroRowsFromConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);
      removeZeroRowsFromConstraints(linearInequalityConstraintsCMatrixO, linearInequalityConstraintsDVectorO);
   }

   private static void removeZeroRowsFromConstraints(DMatrixRMaj matrix, DMatrixRMaj vector)
   {
      for (int rowIndex = vector.numRows - 1; rowIndex >= 0; rowIndex--)
      {
         double sumOfRowElements = 0.0;

         for (int columnIndex = 0; columnIndex < matrix.getNumCols(); columnIndex++)
         {
            sumOfRowElements += Math.abs(matrix.get(rowIndex, columnIndex));
         }

         boolean isZeroRow = MathTools.epsilonEquals(sumOfRowElements, 0.0, 1e-12);
         if (isZeroRow)
         {
            MatrixTools.removeRow(matrix, rowIndex);
            MatrixTools.removeRow(vector, rowIndex);
         }
      }
   }

   private void copyActiveVariableSolutionToAllVariables(DMatrixRMaj solutionToPack, DMatrixRMaj activeVariableSolution)
   {
      if (MatrixTools.containsNaN(activeVariableSolution))
      {
         CommonOps_DDRM.fill(solutionToPack, Double.NaN);
         return;
      }

      int activeVariableIndex = 0;
      for (int variableIndex = 0; variableIndex < solutionToPack.getNumRows(); variableIndex++)
      {
         if (activeVariables.get(variableIndex) != 1.0)
         {
            solutionToPack.set(variableIndex, 0, 0.0);
            continue;
         }

         solutionToPack.set(variableIndex, 0, activeVariableSolution.get(activeVariableIndex, 0));
         activeVariableIndex++;
      }
   }

   @Override
   public void setLowerBounds(DMatrixRMaj variableLowerBounds)
   {
      if (variableLowerBounds.getNumRows() != originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      originalVariableLowerBounds.set(variableLowerBounds);
   }

   @Override
   public void setUpperBounds(DMatrixRMaj variableUpperBounds)
   {
      if (variableUpperBounds.getNumRows() != originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      originalVariableUpperBounds.set(variableUpperBounds);
   }

   @Override
   public void setQuadraticCostFunction(DMatrixRMaj costQuadraticMatrix, DMatrixRMaj costLinearVector, double quadraticCostScalar)
   {
      if (costLinearVector.getNumCols() != 1)
         throw new RuntimeException("costLinearVector.getNumCols() != 1");
      if (costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows()");
      if (costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols()");

      symmetricCostQuadraticMatrix.reshape(costQuadraticMatrix.getNumCols(), costQuadraticMatrix.getNumRows());
      CommonOps_DDRM.transpose(costQuadraticMatrix, symmetricCostQuadraticMatrix);

      CommonOps_DDRM.add(costQuadraticMatrix, symmetricCostQuadraticMatrix, symmetricCostQuadraticMatrix);
      CommonOps_DDRM.scale(0.5, symmetricCostQuadraticMatrix);
      originalQuadraticCostQMatrix.set(symmetricCostQuadraticMatrix);
      originalQuadraticCostQVector.set(costLinearVector);
      this.quadraticCostScalar = quadraticCostScalar;

      setAllVariablesActive();
   }

   @Override
   public double getObjectiveCost(DMatrixRMaj x)
   {
      NativeCommonOps.multQuad(x, originalQuadraticCostQMatrix, computedObjectiveFunctionValue);
      CommonOps_DDRM.scale(0.5, computedObjectiveFunctionValue);
      CommonOps_DDRM.multAddTransA(originalQuadraticCostQVector, x, computedObjectiveFunctionValue);
      return computedObjectiveFunctionValue.get(0, 0) + quadraticCostScalar;
   }

   @Override
   public void setLinearEqualityConstraints(DMatrixRMaj linearEqualityConstraintsAMatrix, DMatrixRMaj linearEqualityConstraintsBVector)
   {
      if (linearEqualityConstraintsBVector.getNumCols() != 1)
         throw new RuntimeException("linearEqualityConstraintsBVector.getNumCols() != 1");
      if (linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows()");
      if (linearEqualityConstraintsAMatrix.getNumCols() != originalQuadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      originalLinearEqualityConstraintsBVector.set(linearEqualityConstraintsBVector);
      originalLinearEqualityConstraintsAMatrix.set(linearEqualityConstraintsAMatrix);
   }

   @Override
   public void setLinearInequalityConstraints(DMatrixRMaj linearInequalityConstraintCMatrix, DMatrixRMaj linearInequalityConstraintDVector)
   {
      if (linearInequalityConstraintDVector.getNumCols() != 1)
         throw new RuntimeException("linearInequalityConstraintDVector.getNumCols() != 1");
      if (linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows()");
      if (linearInequalityConstraintCMatrix.getNumCols() != originalQuadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      originalLinearInequalityConstraintsDVectorO.set(linearInequalityConstraintDVector);
      originalLinearInequalityConstraintsCMatrixO.set(linearInequalityConstraintCMatrix);
   }

   @Override
   public void setActiveVariables(DMatrixRMaj activeVariables)
   {
      if (activeVariables.getNumRows() != originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("activeVariables.getNumRows() != quadraticCostQMatrix.getNumRows()");

      this.activeVariables.set(activeVariables);
   }

   @Override
   public void setVariableActive(int variableIndex)
   {
      if (variableIndex < 0 || variableIndex >= originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variable index is outside the number of variables: " + variableIndex);

      if (variableIndex >= activeVariables.getNumRows())
         return; // Any variable that is outside the activeVariables vector will be considered, nothing to do then.

      activeVariables.set(variableIndex, 0, 1.0);
   }

   @Override
   public void setVariableInactive(int variableIndex)
   {
      if (variableIndex < 0 || variableIndex >= originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variable index is outside the number of variables: " + variableIndex);

      if (variableIndex >= activeVariables.getNumRows())
         activeVariables.reshape(variableIndex + 1, 1, true);

      activeVariables.set(variableIndex, 0, 0.0);
   }

   @Override
   public void setAllVariablesActive()
   {
      activeVariables.reshape(originalQuadraticCostQMatrix.getNumRows(), 1);
      CommonOps_DDRM.fill(activeVariables, 1.0);
   }

   @Override
   public void clear()
   {
      super.clear();

      originalQuadraticCostQMatrix.reshape(0, 0);
      originalQuadraticCostQVector.reshape(0, 0);

      originalLinearEqualityConstraintsAMatrix.reshape(0, 0);
      originalLinearEqualityConstraintsBVector.reshape(0, 0);

      originalLinearInequalityConstraintsCMatrixO.reshape(0, 0);
      originalLinearInequalityConstraintsDVectorO.reshape(0, 0);

      originalVariableLowerBounds.reshape(0, 0);
      originalVariableUpperBounds.reshape(0, 0);

      activeVariables.reshape(0, 0);
      activeVariableSolution.reshape(0, 0);
   }

   @Override
   public int solve(DMatrixRMaj solutionToPack)
   {
      removeInactiveVariables();

      solutionToPack.reshape(originalQuadraticCostQMatrix.numRows, 1);

      int numberOfIterations = super.solve(activeVariableSolution);

      copyActiveVariableSolutionToAllVariables(solutionToPack, activeVariableSolution);

      return numberOfIterations;
   }

}
