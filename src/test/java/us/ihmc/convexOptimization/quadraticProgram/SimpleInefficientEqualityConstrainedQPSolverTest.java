package us.ihmc.convexOptimization.quadraticProgram;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.RandomNumbers;

public class SimpleInefficientEqualityConstrainedQPSolverTest
{
   private static final boolean VERBOSE = false;

   @Test
   public void testSimpleCases()
   {
      SimpleInefficientEqualityConstrainedQPSolver solver = new SimpleInefficientEqualityConstrainedQPSolver();

      // Minimize x^T * x
      double[][] costQuadraticMatrix = new double[][] {{2.0}};
      double[] costLinearVector = new double[] {0.0};
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      double[] solution = new double[1];
      double[] lagrangeMultipliers = new double[0];
      solver.solve(solution, lagrangeMultipliers);
      assertEquals(1, solution.length);
      assertEquals(0.0, solution[0], 1e-7);

      // Minimize (x-5) * (x-5) = x^2 - 10x + 25
      solver.clear();
      costQuadraticMatrix = new double[][] {{2.0}};
      costLinearVector = new double[] {-10.0};
      quadraticCostScalar = 25.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      solution = new double[1];
      lagrangeMultipliers = new double[0];
      solver.solve(solution, lagrangeMultipliers);

      assertEquals(1, solution.length);
      assertEquals(5.0, solution[0], 1e-7);
      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.length, 1);
      solutionMatrix.setData(solution);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(0.0, objectiveCost, 1e-7);

      // Minimize (x-5) * (x-5) + (y-3) * (y-3) = 1/2 * (2x^2 + 2y^2) - 10x -6y + 34
      solver.clear();
      costQuadraticMatrix = new double[][] {{2.0, 0.0}, {0.0, 2.0}};
      costLinearVector = new double[] {-10.0, -6.0};
      quadraticCostScalar = 34.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      solution = new double[2];
      lagrangeMultipliers = new double[0];
      solver.solve(solution, lagrangeMultipliers);

      assertEquals(2, solution.length);
      assertEquals(5.0, solution[0], 1e-7);
      assertEquals(3.0, solution[1], 1e-7);
      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.length, 1);
      solutionMatrix.setData(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(0.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 subject to x + y = 1.0
      solver.clear();
      costQuadraticMatrix = new double[][] {{2.0, 0.0}, {0.0, 2.0}};
      costLinearVector = new double[] {0.0, 0.0};
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      double[][] linearEqualityConstraintsAMatrix = new double[][] {{1.0, 1.0}};
      double[] linearEqualityConstraintsBVector = new double[] {1.0};
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      solution = new double[2];
      lagrangeMultipliers = new double[1];
      solver.solve(solution, lagrangeMultipliers);

      assertEquals(2, solution.length);
      assertEquals(0.5, solution[0], 1e-7);
      assertEquals(0.5, solution[1], 1e-7);
      assertEquals(-1.0, lagrangeMultipliers[0], 1e-7); // Lagrange multiplier is -1.0;
      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.length, 1);
      solutionMatrix.setData(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(0.5, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 subject to x + y = 2.0, 3x - 3y = 0.0
      solver.clear();
      costQuadraticMatrix = new double[][] {{2.0, 0.0}, {0.0, 2.0}};
      costLinearVector = new double[] {0.0, 0.0};
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearEqualityConstraintsAMatrix = new double[][] {{1.0, 1.0}, {3.0, -3.0}};
      linearEqualityConstraintsBVector = new double[] {2.0, 0.0};
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      solution = new double[2];
      lagrangeMultipliers = new double[2];
      solver.solve(solution, lagrangeMultipliers);

      assertEquals(2, solution.length);
      assertEquals(1.0, solution[0], 1e-7);
      assertEquals(1.0, solution[1], 1e-7);
      assertEquals(-2.0, lagrangeMultipliers[0], 1e-7); // Lagrange multiplier
      assertEquals(0.0, lagrangeMultipliers[1], 1e-7); // Lagrange multiplier
      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.length, 1);
      solutionMatrix.setData(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(2.0, objectiveCost, 1e-7);
   }

   @Test
   public void testLargeRandomProblemWithNoEqualityConstraints()
   {
      Random random = new Random(1776L);

      SimpleInefficientEqualityConstrainedQPSolver solver = new SimpleInefficientEqualityConstrainedQPSolver();

      int numberOfTests = 100;

      for (int testNumber = 0; testNumber < numberOfTests; testNumber++)
      {
         solver.clear();
         int numberOfVariables = 100;

         DMatrixRMaj costQuadraticMatrix = nextDMatrixRMaj(random, numberOfVariables, numberOfVariables);
         DMatrixRMaj identity = CommonOps_DDRM.identity(numberOfVariables, numberOfVariables); // Add n*I to make sure it is positive definite...
         CommonOps_DDRM.scale(numberOfVariables, identity);
         CommonOps_DDRM.addEquals(costQuadraticMatrix, identity);

         DMatrixRMaj costLinearVector = nextDMatrixRMaj(random, numberOfVariables, 1);
         double quadraticCostScalar = RandomNumbers.nextDouble(random, 30.0);

         solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

         double[] solution = new double[numberOfVariables];
         double[] lagrangeMultipliers = new double[0];
         solver.solve(solution, lagrangeMultipliers);

         assertEquals(numberOfVariables, solution.length);

         DMatrixRMaj solutionMatrix = new DMatrixRMaj(numberOfVariables, 1);
         solutionMatrix.setData(solution);
         double objectiveCost = solver.getObjectiveCost(solutionMatrix);

         double[] solutionWithSmallPerturbation = new double[numberOfVariables];
         for (int i = 0; i < numberOfVariables; i++)
         {
            solutionWithSmallPerturbation[i] = solution[i] + RandomNumbers.nextDouble(random, 1e-7);
         }

         solutionMatrix = new DMatrixRMaj(numberOfVariables, 1);
         solutionMatrix.setData(solutionWithSmallPerturbation);
         double objectiveCostWithSmallPerturbation = solver.getObjectiveCost(solutionMatrix);

         assertTrue(objectiveCostWithSmallPerturbation > objectiveCost,
                    "objectiveCostWithSmallPerturbation = " + objectiveCostWithSmallPerturbation + ", objectiveCost = " + objectiveCost);
      }
   }

   @Test
   public void testLargeRandomProblemWithEqualityConstraints()
   {
      Random random = new Random(1776L);

      SimpleInefficientEqualityConstrainedQPSolver solver = new SimpleInefficientEqualityConstrainedQPSolver();

      int numberOfTests = 100;

      long startTimeMillis = System.currentTimeMillis();

      for (int testNumber = 0; testNumber < numberOfTests; testNumber++)
      {
         solver.clear();
         int numberOfVariables = 80;
         int numberOfEqualityConstraints = 16;

         DMatrixRMaj costQuadraticMatrix = nextDMatrixRMaj(random, numberOfVariables, numberOfVariables);
         DMatrixRMaj identity = CommonOps_DDRM.identity(numberOfVariables, numberOfVariables); // Add n*I to make sure it is positive definite...
         CommonOps_DDRM.scale(numberOfVariables, identity);
         CommonOps_DDRM.addEquals(costQuadraticMatrix, identity);

         DMatrixRMaj costLinearVector = nextDMatrixRMaj(random, numberOfVariables, 1);
         double quadraticCostScalar = RandomNumbers.nextDouble(random, 30.0);

         solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

         DMatrixRMaj linearEqualityConstraintsAMatrix = nextDMatrixRMaj(random, numberOfEqualityConstraints, numberOfVariables);
         DMatrixRMaj linearEqualityConstraintsBVector = nextDMatrixRMaj(random, numberOfEqualityConstraints, 1);
         solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

         double[] solution = new double[numberOfVariables];
         double[] lagrangeMultipliers = new double[numberOfEqualityConstraints];
         solver.solve(solution, lagrangeMultipliers);

         assertEquals(numberOfVariables, solution.length);
         assertEquals(numberOfEqualityConstraints, lagrangeMultipliers.length);

         DMatrixRMaj solutionMatrix = new DMatrixRMaj(numberOfVariables, 1);
         solutionMatrix.setData(solution);
         double objectiveCost = solver.getObjectiveCost(solutionMatrix);

         // Verify equality constraints hold:
         verifyEqualityConstraintsHold(numberOfEqualityConstraints, linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector, solutionMatrix);

         // Verify objective is minimized by comparing to small perturbation:
         double[] solutionWithSmallPerturbation = new double[numberOfVariables];
         for (int i = 0; i < numberOfVariables; i++)
         {
            solutionWithSmallPerturbation[i] = solution[i] + RandomNumbers.nextDouble(random, 1e-4);
         }

         solutionMatrix = new DMatrixRMaj(numberOfVariables, 1);
         solutionMatrix.setData(solutionWithSmallPerturbation);

         verifyEqualityConstraintsDoNotHold(numberOfEqualityConstraints, linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector, solutionMatrix);
         DMatrixRMaj solutionMatrixProjectedOntoEqualityConstraints = projectOntoEqualityConstraints(solutionMatrix,
                                                                                                        linearEqualityConstraintsAMatrix,
                                                                                                        linearEqualityConstraintsBVector);
         verifyEqualityConstraintsHold(numberOfEqualityConstraints,
                                       linearEqualityConstraintsAMatrix,
                                       linearEqualityConstraintsBVector,
                                       solutionMatrixProjectedOntoEqualityConstraints);

         double objectiveCostWithSmallPerturbation = solver.getObjectiveCost(solutionMatrixProjectedOntoEqualityConstraints);

         assertTrue(objectiveCostWithSmallPerturbation > objectiveCost,
                    "objectiveCostWithSmallPerturbation = " + objectiveCostWithSmallPerturbation + ", objectiveCost = " + objectiveCost);
      }

      long endTimeMillis = System.currentTimeMillis();

      double timePerTest = (endTimeMillis - startTimeMillis) * 0.001 / numberOfTests;

      if (VERBOSE)
      {
         System.out.println("Time per test is " + timePerTest);
      }

   }

   private void verifyEqualityConstraintsHold(int numberOfEqualityConstraints, DMatrixRMaj linearEqualityConstraintsAMatrix,
                                              DMatrixRMaj linearEqualityConstraintsBVector, DMatrixRMaj solutionMatrix)
   {
      double maxAbsoluteError = getMaxEqualityConstraintError(numberOfEqualityConstraints,
                                                              linearEqualityConstraintsAMatrix,
                                                              linearEqualityConstraintsBVector,
                                                              solutionMatrix);
      assertEquals(0.0, maxAbsoluteError, 1e-5);
   }

   private void verifyEqualityConstraintsDoNotHold(int numberOfEqualityConstraints, DMatrixRMaj linearEqualityConstraintsAMatrix,
                                                   DMatrixRMaj linearEqualityConstraintsBVector, DMatrixRMaj solutionMatrix)
   {
      double maxAbsoluteError = getMaxEqualityConstraintError(numberOfEqualityConstraints,
                                                              linearEqualityConstraintsAMatrix,
                                                              linearEqualityConstraintsBVector,
                                                              solutionMatrix);
      assertTrue(maxAbsoluteError > 1e-5);
   }

   private double getMaxEqualityConstraintError(int numberOfEqualityConstraints, DMatrixRMaj linearEqualityConstraintsAMatrix,
                                                DMatrixRMaj linearEqualityConstraintsBVector, DMatrixRMaj solutionMatrix)
   {
      DMatrixRMaj checkMatrix = new DMatrixRMaj(numberOfEqualityConstraints, 1);
      CommonOps_DDRM.mult(linearEqualityConstraintsAMatrix, solutionMatrix, checkMatrix);
      CommonOps_DDRM.subtractEquals(checkMatrix, linearEqualityConstraintsBVector);

      return getMaxAbsoluteDataEntry(checkMatrix);
   }

   private DMatrixRMaj projectOntoEqualityConstraints(DMatrixRMaj solutionMatrix, DMatrixRMaj linearEqualityConstraintsAMatrix,
                                                         DMatrixRMaj linearEqualityConstraintsBVector)
   {
      int numberOfVariables = solutionMatrix.getNumRows();
      if (linearEqualityConstraintsAMatrix.getNumCols() != numberOfVariables)
         throw new RuntimeException();

      int numberOfConstraints = linearEqualityConstraintsAMatrix.getNumRows();
      if (linearEqualityConstraintsBVector.getNumRows() != numberOfConstraints)
         throw new RuntimeException();

      DMatrixRMaj AZMinusB = new DMatrixRMaj(numberOfConstraints, 1);
      CommonOps_DDRM.mult(linearEqualityConstraintsAMatrix, solutionMatrix, AZMinusB);
      CommonOps_DDRM.subtractEquals(AZMinusB, linearEqualityConstraintsBVector);

      DMatrixRMaj AATransposeInverse = new DMatrixRMaj(numberOfConstraints, numberOfConstraints);
      DMatrixRMaj linearEqualityConstraintsAMatrixTranspose = new DMatrixRMaj(linearEqualityConstraintsAMatrix);
      CommonOps_DDRM.transpose(linearEqualityConstraintsAMatrixTranspose);

      CommonOps_DDRM.mult(linearEqualityConstraintsAMatrix, linearEqualityConstraintsAMatrixTranspose, AATransposeInverse);
      CommonOps_DDRM.invert(AATransposeInverse);

      DMatrixRMaj ATransposeAATransposeInverse = new DMatrixRMaj(numberOfVariables, numberOfConstraints);
      CommonOps_DDRM.mult(linearEqualityConstraintsAMatrixTranspose, AATransposeInverse, ATransposeAATransposeInverse);

      DMatrixRMaj vectorToSubtract = new DMatrixRMaj(numberOfVariables, 1);
      CommonOps_DDRM.mult(ATransposeAATransposeInverse, AZMinusB, vectorToSubtract);

      DMatrixRMaj projectedSolutionMatrix = new DMatrixRMaj(solutionMatrix);
      CommonOps_DDRM.subtractEquals(projectedSolutionMatrix, vectorToSubtract);

      return projectedSolutionMatrix;
   }

   private double getMaxAbsoluteDataEntry(DMatrixRMaj matrix)
   {
      int numberOfRows = matrix.getNumRows();
      int numberOfColumns = matrix.getNumCols();

      double max = Double.NEGATIVE_INFINITY;

      for (int i = 0; i < numberOfRows; i++)
      {
         for (int j = 0; j < numberOfColumns; j++)
         {
            double absoluteValue = Math.abs(matrix.get(i, j));
            if (absoluteValue > max)
            {
               max = absoluteValue;
            }
         }
      }

      return max;
   }

   public static DMatrixRMaj nextDMatrixRMaj(Random random, int numberOfRows, int numberOfColumns)
   {
      return nextDMatrixRMaj(random, numberOfRows, numberOfColumns, 1.0);
   }

   public static DMatrixRMaj nextDMatrixRMaj(Random random, int numberOfRows, int numberOfColumns, double maxAbsoluteValue)
   {
      return RandomMatrices_DDRM.rectangle(numberOfRows, numberOfColumns, -maxAbsoluteValue, maxAbsoluteValue, random);
   }

   public static DMatrixRMaj nextDMatrixRMaj(Random random, int numberOfRows, int numberOfColumns, double boundaryOne, double boundaryTwo)
   {
      return RandomMatrices_DDRM.rectangle(numberOfRows, numberOfColumns, boundaryOne, boundaryTwo, random);
   }
}
