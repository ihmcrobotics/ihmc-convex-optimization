package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import us.ihmc.commons.MathTools;
import us.ihmc.matrixlib.MatrixTools;
import us.ihmc.matrixlib.NativeMatrix;

public class SimpleEfficientActiveSetQPSolverWithInactiveVariables extends SimpleEfficientActiveSetQPSolver
      implements ActiveSetQPSolverWithInactiveVariablesInterface
{
   private final NativeMatrix originalQuadraticCostQMatrix = new NativeMatrix(0, 0);
   private final NativeMatrix originalQuadraticCostQVector = new NativeMatrix(0, 0);

   private final NativeMatrix originalLinearEqualityConstraintsAMatrix = new NativeMatrix(0, 0);
   private final NativeMatrix originalLinearEqualityConstraintsBVector = new NativeMatrix(0, 0);

   private final NativeMatrix originalLinearInequalityConstraintsCMatrixO = new NativeMatrix(0, 0);
   private final NativeMatrix originalLinearInequalityConstraintsDVectorO = new NativeMatrix(0, 0);

   private final NativeMatrix originalVariableLowerBounds = new NativeMatrix(0, 0);
   private final NativeMatrix originalVariableUpperBounds = new NativeMatrix(0, 0);

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

         quadraticCostQMatrix.removeRow(variableIndex);
         quadraticCostQMatrix.removeColumn(variableIndex);

         quadraticCostQVector.removeRow(variableIndex);

         if (linearEqualityConstraintsAMatrix.getNumElements() > 0)
            linearEqualityConstraintsAMatrix.removeColumn(variableIndex);
         if (linearInequalityConstraintsCMatrixO.getNumElements() > 0)
            linearInequalityConstraintsCMatrixO.removeColumn(variableIndex);

         if (variableLowerBounds.getNumElements() > 0)
            variableLowerBounds.removeRow(variableIndex);
         if (variableUpperBounds.getNumElements() > 0)
            variableUpperBounds.removeRow(variableIndex);
      }

      int numVars = quadraticCostQMatrix.getNumRows();
      if (linearEqualityConstraintsAMatrix.getNumElements() == 0)
         linearEqualityConstraintsAMatrix.reshape(0, numVars);
      if (linearInequalityConstraintsCMatrixO.getNumElements() == 0)
         linearInequalityConstraintsCMatrixO.reshape(0, numVars);

      removeZeroRowsFromConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);
      removeZeroRowsFromConstraints(linearInequalityConstraintsCMatrixO, linearInequalityConstraintsDVectorO);
   }

   private static void removeZeroRowsFromConstraints(NativeMatrix matrix, NativeMatrix vector)
   {
      for (int rowIndex = vector.getNumRows() - 1; rowIndex >= 0; rowIndex--)
      {
         double sumOfRowElements = 0.0;

         for (int columnIndex = 0; columnIndex < matrix.getNumCols(); columnIndex++)
         {
            sumOfRowElements += Math.abs(matrix.get(rowIndex, columnIndex));
         }

         boolean isZeroRow = MathTools.epsilonEquals(sumOfRowElements, 0.0, 1e-12);
         if (isZeroRow)
         {
            matrix.removeRow(rowIndex);
            vector.removeRow(rowIndex);
         }
      }
   }

   private void copyActiveVariableSolutionToAllVariables(DMatrix solutionToPack, DMatrixRMaj activeVariableSolution)
   {
      if (MatrixTools.containsNaN(activeVariableSolution))
      {
         for (int i = 0; i < solutionToPack.getNumRows(); i++)
            solutionToPack.set(i, 0, Double.NaN);
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
   public void setLowerBounds(DMatrix variableLowerBounds)
   {
      if (variableLowerBounds.getNumRows() != originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      originalVariableLowerBounds.set(variableLowerBounds);
   }

   @Override
   public void setUpperBounds(DMatrix variableUpperBounds)
   {
      if (variableUpperBounds.getNumRows() != originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      originalVariableUpperBounds.set(variableUpperBounds);
   }

   @Override
   public void setQuadraticCostFunction(DMatrix costQuadraticMatrix, DMatrix costLinearVector, double quadraticCostScalar)
   {
      if (costLinearVector.getNumCols() != 1)
         throw new RuntimeException("costLinearVector.getNumCols() != 1");
      if (costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costLinearVector.getNumRows()");
      if (costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols())
         throw new RuntimeException("costQuadraticMatrix.getNumRows() != costQuadraticMatrix.getNumCols()");

      this.costQuadraticMatrix.set(costQuadraticMatrix);

      symmetricCostQuadraticMatrix.transpose(this.costQuadraticMatrix);
      symmetricCostQuadraticMatrix.add(this.costQuadraticMatrix, symmetricCostQuadraticMatrix); // Note: Check for aliasing
      symmetricCostQuadraticMatrix.scale(0.5);

      originalQuadraticCostQMatrix.set(symmetricCostQuadraticMatrix);
      originalQuadraticCostQVector.set(costLinearVector);
      this.quadraticCostScalar = quadraticCostScalar;

      setAllVariablesActive();
   }

   @Override
   public double getObjectiveCost(DMatrixRMaj x)
   {
      nativexSolutionMatrix.set(x);
      computedObjectiveFunctionValue.multQuad(nativexSolutionMatrix, originalQuadraticCostQMatrix);
      computedObjectiveFunctionValue.scale(0.5);
      computedObjectiveFunctionValue.multAddTransA(originalQuadraticCostQVector, nativexSolutionMatrix);
      return computedObjectiveFunctionValue.get(0, 0) + quadraticCostScalar;
   }

   @Override
   public void setLinearEqualityConstraints(DMatrix linearEqualityConstraintsAMatrix, DMatrix linearEqualityConstraintsBVector)
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
   public void setLinearInequalityConstraints(DMatrix linearInequalityConstraintCMatrix, DMatrix linearInequalityConstraintDVector)
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
   public int solve(DMatrix solutionToPack)
   {
      removeInactiveVariables();

      activeVariableSolution.reshape(quadraticCostQVector.getNumRows(), 1);

      if (solutionToPack.getNumRows() != originalQuadraticCostQMatrix.getNumRows() || solutionToPack.getNumCols() != 1)
         throw new IllegalArgumentException("Invalid matrix dimensions.");

      int numberOfIterations = super.solve(activeVariableSolution);

      copyActiveVariableSolutionToAllVariables(solutionToPack, activeVariableSolution);

      return numberOfIterations;
   }

}
