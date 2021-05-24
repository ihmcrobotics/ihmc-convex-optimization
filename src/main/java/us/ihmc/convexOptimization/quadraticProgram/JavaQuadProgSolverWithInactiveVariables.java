package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrix1Row;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import us.ihmc.commons.MathTools;
import us.ihmc.matrixlib.MatrixTools;
import us.ihmc.matrixlib.NativeCommonOps;

public class JavaQuadProgSolverWithInactiveVariables extends JavaQuadProgSolver implements ActiveSetQPSolverWithInactiveVariablesInterface
{
   private final DMatrixRMaj originalQuadraticCostQMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalQuadraticCostQVector = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj originalLinearEqualityConstraintsAMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalLinearEqualityConstraintsBVector = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj originalLinearInequalityConstraintsCMatrixO = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalLinearInequalityConstraintsDVectorO = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj originalLowerBoundsCMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalVariableLowerBounds = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj originalUpperBoundsCMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj originalVariableUpperBounds = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj activeVariables = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj activeVariableSolution = new DMatrixRMaj(0, 0);

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
   public void setLowerBounds(DMatrix variableLowerBounds)
   {
      int numberOfLowerBounds = variableLowerBounds.getNumRows();

      if (numberOfLowerBounds != originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableLowerBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      originalLowerBoundsCMatrix.reshape(numberOfLowerBounds, numberOfLowerBounds);
      CommonOps_DDRM.setIdentity(originalLowerBoundsCMatrix);

      originalVariableLowerBounds.set(variableLowerBounds);
      CommonOps_DDRM.scale(-1.0, originalVariableLowerBounds);
   }

   @Override
   public void setUpperBounds(DMatrix variableUpperBounds)
   {
      int numberOfUpperBounds = variableUpperBounds.getNumRows();
      if (numberOfUpperBounds != originalQuadraticCostQMatrix.getNumRows())
         throw new RuntimeException("variableUpperBounds.getNumRows() != quadraticCostQMatrix.getNumRows()");

      originalUpperBoundsCMatrix.reshape(numberOfUpperBounds, numberOfUpperBounds);
      CommonOps_DDRM.setIdentity(originalUpperBoundsCMatrix);
      CommonOps_DDRM.scale(-1.0, originalUpperBoundsCMatrix);

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

      originalQuadraticCostQMatrix.set(costQuadraticMatrix);
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
   public void setLinearEqualityConstraints(DMatrix linearEqualityConstraintsAMatrix, DMatrix linearEqualityConstraintsBVector)
   {
      int numberOfEqualityConstraints = linearEqualityConstraintsBVector.getNumRows();

      if (linearEqualityConstraintsBVector.getNumCols() != 1)
         throw new RuntimeException("linearEqualityConstraintsBVector.getNumCols() != 1");
      if (linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumRows() != linearEqualityConstraintsBVector.getNumRows()");
      if (linearEqualityConstraintsAMatrix.getNumCols() != originalQuadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearEqualityConstraintsAMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      originalLinearEqualityConstraintsAMatrix.reshape(linearEqualityConstraintsAMatrix.getNumCols(), numberOfEqualityConstraints);
      standardTranspose(linearEqualityConstraintsAMatrix, originalLinearEqualityConstraintsAMatrix);
      CommonOps_DDRM.scale(-1.0, originalLinearEqualityConstraintsAMatrix);

      originalLinearEqualityConstraintsBVector.set(linearEqualityConstraintsBVector);
   }

   @Override
   public void setLinearInequalityConstraints(DMatrix linearInequalityConstraintCMatrix, DMatrix linearInequalityConstraintDVector)
   {
      int numberOfInequalityConstraints = linearInequalityConstraintDVector.getNumRows();

      if (linearInequalityConstraintDVector.getNumCols() != 1)
         throw new RuntimeException("linearInequalityConstraintDVector.getNumCols() != 1");
      if (linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumRows() != linearInequalityConstraintDVector.getNumRows()");
      if (linearInequalityConstraintCMatrix.getNumCols() != originalQuadraticCostQMatrix.getNumCols())
         throw new RuntimeException("linearInequalityConstraintCMatrix.getNumCols() != quadraticCostQMatrix.getNumCols()");

      originalLinearInequalityConstraintsCMatrixO.reshape(linearInequalityConstraintCMatrix.getNumCols(), numberOfInequalityConstraints);
      standardTranspose(linearInequalityConstraintCMatrix, originalLinearInequalityConstraintsCMatrixO);
      CommonOps_DDRM.scale(-1.0, originalLinearInequalityConstraintsCMatrixO);

      originalLinearInequalityConstraintsDVectorO.set(linearInequalityConstraintDVector);
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

      originalLowerBoundsCMatrix.reshape(0, 0);
      originalVariableLowerBounds.reshape(0, 0);

      originalUpperBoundsCMatrix.reshape(0, 0);
      originalVariableUpperBounds.reshape(0, 0);

      activeVariables.reshape(0, 0);
      activeVariableSolution.reshape(0, 0);
   }

   @Override
   public int solve(DMatrix solutionToPack)
   {
      removeInactiveVariables();

      if (solutionToPack.getNumRows() != originalQuadraticCostQMatrix.numRows || solutionToPack.getNumCols() != 1)
         throw new IllegalArgumentException("Invalid matrix dimensions.");

      int numberOfIterations = super.solve(activeVariableSolution);

      copyActiveVariableSolutionToAllVariables(solutionToPack, activeVariableSolution);

      return numberOfIterations;
   }

   private void setMatricesFromOriginal()
   {
      quadraticCostQMatrix.set(originalQuadraticCostQMatrix);
      quadraticCostQVector.set(originalQuadraticCostQVector);

      linearEqualityConstraintsAMatrix.set(originalLinearEqualityConstraintsAMatrix);
      linearEqualityConstraintsBVector.set(originalLinearEqualityConstraintsBVector);

      linearInequalityConstraintsCMatrixO.set(originalLinearInequalityConstraintsCMatrixO);
      linearInequalityConstraintsDVectorO.set(originalLinearInequalityConstraintsDVectorO);

      lowerBoundsCMatrix.set(originalLowerBoundsCMatrix);
      variableLowerBounds.set(originalVariableLowerBounds);

      upperBoundsCMatrix.set(originalUpperBoundsCMatrix);
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
            MatrixTools.removeRow(linearEqualityConstraintsAMatrix, variableIndex);
         if (linearInequalityConstraintsCMatrixO.getNumElements() > 0)
            MatrixTools.removeRow(linearInequalityConstraintsCMatrixO, variableIndex);

         if (variableLowerBounds.getNumElements() > 0)
         {
            MatrixTools.removeRow(variableLowerBounds, variableIndex);
            MatrixTools.removeRow(lowerBoundsCMatrix, variableIndex);
            MatrixTools.removeColumn(lowerBoundsCMatrix, variableIndex);
         }
         if (variableUpperBounds.getNumElements() > 0)
         {
            MatrixTools.removeRow(variableUpperBounds, variableIndex);
            MatrixTools.removeRow(upperBoundsCMatrix, variableIndex);
            MatrixTools.removeColumn(upperBoundsCMatrix, variableIndex);
         }
      }

      int numVars = quadraticCostQMatrix.getNumRows();
      if (linearEqualityConstraintsAMatrix.getNumElements() == 0)
         linearEqualityConstraintsAMatrix.reshape(numVars, 0);
      if (linearInequalityConstraintsCMatrixO.getNumElements() == 0)
         linearInequalityConstraintsCMatrixO.reshape(numVars, 0);

      removeZeroColumnsFromConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);
      removeZeroColumnsFromConstraints(linearInequalityConstraintsCMatrixO, linearInequalityConstraintsDVectorO);
   }

   private static void removeZeroColumnsFromConstraints(DMatrixRMaj matrix, DMatrixRMaj vector)
   {
      for (int rowIndex = vector.numRows - 1; rowIndex >= 0; rowIndex--)
      {
         double sumOfRowElements = 0.0;

         for (int columnIndex = 0; columnIndex < matrix.getNumRows(); columnIndex++)
         {
            sumOfRowElements += Math.abs(matrix.get(columnIndex, rowIndex));
         }

         boolean isZeroColumn = MathTools.epsilonEquals(sumOfRowElements, 0.0, 1e-12);
         if (isZeroColumn)
         {
            MatrixTools.removeColumn(matrix, rowIndex);
            MatrixTools.removeRow(vector, rowIndex);
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
            continue;

         solutionToPack.set(variableIndex, 0, activeVariableSolution.get(activeVariableIndex, 0));
         activeVariableIndex++;
      }
   }
}
