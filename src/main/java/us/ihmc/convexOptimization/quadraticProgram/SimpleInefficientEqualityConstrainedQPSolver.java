package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import us.ihmc.matrixlib.MatrixTools;

public class SimpleInefficientEqualityConstrainedQPSolver
{
   private final DMatrixRMaj quadraticCostQMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj quadraticCostQVector = new DMatrixRMaj(0, 0);

   private final DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj linearEqualityConstraintsAMatrixTranspose = new DMatrixRMaj(0, 0);
   private final DMatrixRMaj linearEqualityConstraintsBVector = new DMatrixRMaj(0, 0);

   private double quadraticCostScalar;

   private int numberOfVariablesToSolve;

   public SimpleInefficientEqualityConstrainedQPSolver()
   {
      numberOfVariablesToSolve = -1; //indicate unknown size
   }

   public void clear()
   {
      numberOfVariablesToSolve = -1;

      linearEqualityConstraintsAMatrix.reshape(0, 0);
      linearEqualityConstraintsAMatrixTranspose.reshape(0, 0);
      linearEqualityConstraintsBVector.reshape(0, 0);
      quadraticCostQMatrix.reshape(0, 0);
      quadraticCostQVector.reshape(0, 0);
   }

   //   public int getNumEqualityConstraints()
   //   {
   //      return linearEqualityConstraintAMatrix.getNumRows();
   //   }

   public void setQuadraticCostFunction(double[][] quadraticCostFunctionWMatrix, double[] quadraticCostFunctionGVector, double quadraticCostScalar)
   {
      assertCorrectSize(quadraticCostFunctionWMatrix);
      assertCorrectSize(quadraticCostFunctionGVector);
      setQuadraticCostFunction(new DMatrixRMaj(quadraticCostFunctionWMatrix), MatrixTools.createVector(quadraticCostFunctionGVector), quadraticCostScalar);
   }

   public void setQuadraticCostFunction(DMatrixRMaj quadraticCostQMatrix, DMatrixRMaj quadraticCostQVector, double quadraticCostScalar)
   {
      setAndAssertCorrectNumberOfVariablesToSolve(quadraticCostQMatrix.numCols);
      setAndAssertCorrectNumberOfVariablesToSolve(quadraticCostQMatrix.numRows);
      setAndAssertCorrectNumberOfVariablesToSolve(quadraticCostQVector.numRows);
      DMatrixRMaj symCostQuadraticMatrix = new DMatrixRMaj(quadraticCostQMatrix);
      CommonOps_DDRM.transpose(symCostQuadraticMatrix);
      CommonOps_DDRM.add(quadraticCostQMatrix, symCostQuadraticMatrix, symCostQuadraticMatrix);
      CommonOps_DDRM.scale(0.5, symCostQuadraticMatrix);
      this.quadraticCostQMatrix.set(symCostQuadraticMatrix);
      this.quadraticCostQVector.set(quadraticCostQVector);
      this.quadraticCostScalar = quadraticCostScalar;
   }

   public void setLinearEqualityConstraints(double[][] linearEqualityConstraintsAMatrix, double[] linearEqualityConstraintsBVector)
   {
      assertCorrectColumnSize(linearEqualityConstraintsAMatrix);
      if (linearEqualityConstraintsAMatrix.length != linearEqualityConstraintsBVector.length)
         throw new RuntimeException();
      setLinearEqualityConstraints(new DMatrixRMaj(linearEqualityConstraintsAMatrix), MatrixTools.createVector(linearEqualityConstraintsBVector));
   }

   public void setLinearEqualityConstraints(DMatrixRMaj linearEqualityConstraintsAMatrix, DMatrixRMaj linearEqualityConstraintsBVector)
   {
      setAndAssertCorrectNumberOfVariablesToSolve(linearEqualityConstraintsAMatrix.numCols);
      this.linearEqualityConstraintsBVector.set(linearEqualityConstraintsBVector);
      this.linearEqualityConstraintsAMatrix.set(linearEqualityConstraintsAMatrix);
      linearEqualityConstraintsAMatrixTranspose.set(CommonOps_DDRM.transpose(linearEqualityConstraintsAMatrix, null));
   }

   //   protected int getLinearEqualityConstraintsSize()
   //   {
   //      if (linearEqualityConstraintAMatrix == null)
   //         return 0;
   //
   //      return linearEqualityConstraintAMatrix.numRows;
   //   }

   private void assertCorrectSize(double[][] matrix)
   {
      if (numberOfVariablesToSolve == -1)
      {
         numberOfVariablesToSolve = matrix.length;
      }

      if (matrix.length != numberOfVariablesToSolve)
         throw new RuntimeException("matrix.length = " + matrix.length + " != numberOfVariablesToSolve = " + numberOfVariablesToSolve);
      if (matrix[0].length != numberOfVariablesToSolve)
         throw new RuntimeException("matrix[0].length = " + matrix[0].length + " != numberOfVariablesToSolve = " + numberOfVariablesToSolve);
   }

   private void assertCorrectColumnSize(double[][] matrix)
   {
      if (numberOfVariablesToSolve == -1)
      {
         numberOfVariablesToSolve = matrix[0].length;
      }

      if (matrix[0].length != numberOfVariablesToSolve)
         throw new RuntimeException("matrix[0].length = " + matrix[0].length + " != numberOfVariablesToSolve = " + numberOfVariablesToSolve);
   }

   protected void setAndAssertCorrectNumberOfVariablesToSolve(int n)
   {
      if (numberOfVariablesToSolve == -1)
      {
         numberOfVariablesToSolve = n;
      }

      if (n != numberOfVariablesToSolve)
         throw new RuntimeException("incorrect NumberOfVariables size");
   }

   private void assertCorrectSize(double[] vector)
   {
      if (numberOfVariablesToSolve == -1)
      {
         numberOfVariablesToSolve = vector.length;
      }

      if (vector.length != numberOfVariablesToSolve)
         throw new RuntimeException("vector.length = " + vector.length + " != numberOfVariablesToSolve = " + numberOfVariablesToSolve);
   }

   public void solve(double[] xSolutionToPack, double[] lagrangeMultipliersToPack)
   {
      int numberOfVariables = quadraticCostQMatrix.getNumCols();
      int numberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();

      if (xSolutionToPack.length != numberOfVariables)
         throw new RuntimeException("xSolutionToPack.length != numberOfVariables");
      if (lagrangeMultipliersToPack.length != numberOfEqualityConstraints)
         throw new RuntimeException("lagrangeMultipliersToPack.length != numberOfEqualityConstraints");

      DMatrixRMaj solution = new DMatrixRMaj(numberOfVariables, 1);
      DMatrixRMaj lagrangeMultipliers = new DMatrixRMaj(numberOfEqualityConstraints, 1);

      solve(solution, lagrangeMultipliers);

      double[] solutionData = solution.getData();

      for (int i = 0; i < numberOfVariables; i++)
      {
         xSolutionToPack[i] = solutionData[i];
      }

      double[] lagrangeMultipliersData = lagrangeMultipliers.getData();

      for (int i = 0; i < numberOfEqualityConstraints; i++)
      {
         lagrangeMultipliersToPack[i] = lagrangeMultipliersData[i];
      }
   }

   public void solve(DMatrixRMaj xSolutionToPack, DMatrixRMaj lagrangeMultipliersToPack)
   {
      int numberOfVariables = quadraticCostQMatrix.getNumCols();
      if (numberOfVariables != quadraticCostQMatrix.getNumRows())
         throw new RuntimeException("numCols != numRows");

      int numberOfEqualityConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      if (numberOfEqualityConstraints > 0)
      {
         if (linearEqualityConstraintsAMatrix.getNumCols() != numberOfVariables)
            throw new RuntimeException("linearEqualityConstraintA.getNumCols() != numberOfVariables");
      }

      if (quadraticCostQVector.getNumRows() != numberOfVariables)
         throw new RuntimeException("quadraticCostQVector.getNumRows() != numRows");
      if (quadraticCostQVector.getNumCols() != 1)
         throw new RuntimeException("quadraticCostQVector.getNumCols() != 1");

      DMatrixRMaj negativeQuadraticCostQVector = new DMatrixRMaj(quadraticCostQVector);
      CommonOps_DDRM.scale(-1.0, negativeQuadraticCostQVector);

      if (numberOfEqualityConstraints == 0)
      {
         xSolutionToPack.reshape(numberOfVariables, 1);
         CommonOps_DDRM.solve(quadraticCostQMatrix, negativeQuadraticCostQVector, xSolutionToPack);
         return;
      }

      CommonOps_DDRM.transpose(linearEqualityConstraintsAMatrix, linearEqualityConstraintsAMatrixTranspose);
      DMatrixRMaj bigMatrix = new DMatrixRMaj(numberOfVariables + numberOfEqualityConstraints, numberOfVariables + numberOfEqualityConstraints);
      DMatrixRMaj bigVector = new DMatrixRMaj(numberOfVariables + numberOfEqualityConstraints, 1);

      CommonOps_DDRM.insert(quadraticCostQMatrix, bigMatrix, 0, 0);
      CommonOps_DDRM.insert(linearEqualityConstraintsAMatrix, bigMatrix, numberOfVariables, 0);
      CommonOps_DDRM.insert(linearEqualityConstraintsAMatrixTranspose, bigMatrix, 0, numberOfVariables);

      CommonOps_DDRM.insert(negativeQuadraticCostQVector, bigVector, 0, 0);
      CommonOps_DDRM.insert(linearEqualityConstraintsBVector, bigVector, numberOfVariables, 0);

      DMatrixRMaj xAndLagrangeMultiplierSolution = new DMatrixRMaj(numberOfVariables + numberOfEqualityConstraints, 1);
      CommonOps_DDRM.solve(bigMatrix, bigVector, xAndLagrangeMultiplierSolution);

      for (int i = 0; i < numberOfVariables; i++)
      {
         xSolutionToPack.set(i, 0, xAndLagrangeMultiplierSolution.get(i, 0));
      }

      for (int i = 0; i < numberOfEqualityConstraints; i++)
      {
         lagrangeMultipliersToPack.set(i, 0, xAndLagrangeMultiplierSolution.get(numberOfVariables + i, 0));
      }
   }

   //   protected static void setPartialMatrix(double[][] fromMatrix, int startRow, int startColumn, DMatrixRMaj toMatrix)
   //   {
   //      for (int i = 0; i < fromMatrix.length; i++)
   //      {
   //         for (int j = 0; j < fromMatrix[0].length; j++)
   //         {
   //            toMatrix.set(startRow + i, startColumn + j, fromMatrix[i][j]);
   //         }
   //      }
   //   }
   //
   //   protected static void setPartialVector(double[] fromVector, int startRow, DMatrixRMaj toVector)
   //   {
   //      for (int i = 0; i < fromVector.length; i++)
   //      {
   //         toVector.set(startRow + i, 0, fromVector[i]);
   //      }
   //   }

   private final DMatrixRMaj computedObjectiveFunctionValue = new DMatrixRMaj(1, 1);

   public double getObjectiveCost(DMatrixRMaj x)
   {
      MatrixTools.multQuad(x, quadraticCostQMatrix, computedObjectiveFunctionValue);
      CommonOps_DDRM.scale(0.5, computedObjectiveFunctionValue);
      CommonOps_DDRM.multAddTransA(quadraticCostQVector, x, computedObjectiveFunctionValue);
      return computedObjectiveFunctionValue.get(0, 0) + quadraticCostScalar;
   }

   public void displayProblem()
   {
      //      setZeroSizeMatrixForNullFields();
      System.out.println("----------------------------------------------------------------------------------------------------");
      System.out.println("equalityA:" + linearEqualityConstraintsAMatrix);
      System.out.println("equalityB:" + linearEqualityConstraintsBVector);
      System.out.println("costQuadQ:" + quadraticCostQMatrix);
      System.out.println("costLinearF:" + quadraticCostQVector);
      System.out.println("costLinearScalar:" + quadraticCostScalar);
      System.out.println("----------------------------------------------------------------------------------------------------");
   }

   //   boolean isNullFieldSet = false;
   //
   //   public void setZeroSizeMatrixForNullFields()
   //   {
   //      if (isNullFieldSet)
   //         return;
   //      assert (numberOfVariablesToSolve > 0);
   //      if (linearEqualityConstraintA == null)
   //      {
   //         linearEqualityConstraintA = new DMatrixRMaj(0, numberOfVariablesToSolve);
   //         linearEqualityConstraintATranspose = new DMatrixRMaj(numberOfVariablesToSolve, 0);
   //         linearEqualityConstraintB = new DMatrixRMaj(0, 1);
   //      }
   //
   //      if (quadraticCostGMatrix == null)
   //      {
   //         quadraticCostGMatrix = new DMatrixRMaj(numberOfVariablesToSolve, numberOfVariablesToSolve);
   //      }
   //      if (quadraticCostFVector == null)
   //      {
   //         quadraticCostFVector = new DMatrixRMaj(numberOfVariablesToSolve, 1);
   //      }
   //   }

}
