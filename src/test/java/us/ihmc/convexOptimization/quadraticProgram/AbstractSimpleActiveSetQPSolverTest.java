package us.ihmc.convexOptimization.quadraticProgram;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.Random;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.RandomNumbers;
import us.ihmc.convexOptimization.exceptions.NoConvergenceException;
import us.ihmc.matrixlib.MatrixTools;

public abstract class AbstractSimpleActiveSetQPSolverTest
{
   private static final boolean VERBOSE = false;

   public abstract ActiveSetQPSolver createSolverToTest();

   @Test
   public void testSimpleCasesWithNoInequalityConstraints()
   {
      ActiveSetQPSolver solver = createSolverToTest();

      // Minimize x^T * x
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj solution = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      int numberOfIterations = solver.solve(solution);
      numberOfIterations = solver.solve(solution); // Make sure ok to solve twice in a row without changing stuff.
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(0, numberOfIterations);
      assertEquals(1, solution.getNumRows());
      assertEquals(0.0, solution.get(0), 1e-7);

      // Minimize (x-5) * (x-5) = x^2 - 10x + 25
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      costLinearVector = MatrixTools.createVector(-10.0);
      quadraticCostScalar = 25.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      solution = new DMatrixRMaj(1, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(0, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(5.0, solution.get(0), 1e-7);
      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(0.0, objectiveCost, 1e-7);

      // Minimize (x-5) * (x-5) + (y-3) * (y-3) = 1/2 * (2x^2 + 2y^2) - 10x -6y + 34
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(-10.0, -6.0);
      quadraticCostScalar = 34.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(0, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(5.0, solution.get(0), 1e-7);
      assertEquals(3.0, solution.get(1), 1e-7);
      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(0.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 subject to x + y = 1.0
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0}});
      DMatrixRMaj linearEqualityConstraintsBVector = MatrixTools.createVector(1.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(0, numberOfIterations);

      assertEquals(0.5, solution.get(0), 1e-7);
      assertEquals(0.5, solution.get(1), 1e-7);
      assertEquals(-1.0, lagrangeEqualityMultipliers.get(0), 1e-7); // Lagrange multiplier is -1.0;
      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(0.5, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 subject to x + y = 2.0, 3x - 3y = 0.0
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0}, {3.0, -3.0}});
      linearEqualityConstraintsBVector = MatrixTools.createVector(2.0, 0.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(2, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(0, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(1.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(-2.0, lagrangeEqualityMultipliers.get(0), 1e-7); // Lagrange multiplier
      assertEquals(0.0, lagrangeEqualityMultipliers.get(1), 1e-7); // Lagrange multiplier
      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(2.0, objectiveCost, 1e-7);
   }

   @Test
   public void testSimpleCasesWithInequalityConstraints()
   {
      testSimpleCasesWithInequalityConstraints(1);
   }

   public void testSimpleCasesWithInequalityConstraints(int expectedNumberOfIterations)
   {
      ActiveSetQPSolver solver = createSolverToTest();

      // Minimize x^T * x subject to x <= 1
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(1.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      DMatrixRMaj solution = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);

      int numberOfIterations = solver.solve(solution);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(0.0, solution.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);

      // Minimize x^T * x subject to x >= 1 (-x <= -1)
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      costLinearVector = MatrixTools.createVector(0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{-1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(-1.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution = new DMatrixRMaj(1, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations + 1, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(1.0, solution.get(0), 1e-7);
      assertEquals(2.0, lagrangeInequalityMultipliers.get(0), 1e-7);

      // Minimize (x-5) * (x-5) = x^2 - 10x + 25 subject to x <= 3.0
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      costLinearVector = MatrixTools.createVector(-10.0);
      quadraticCostScalar = 25.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(3.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution = new DMatrixRMaj(1, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations + 1, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(3.0, solution.get(0), 1e-7);
      assertEquals(4.0, lagrangeInequalityMultipliers.get(0), 1e-7);

      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(4.0, objectiveCost, 1e-7);

      // Minimize (x-5) * (x-5) + (y-3) * (y-3) = 1/2 * (2x^2 + 2y^2) - 10x -6y + 34 subject to x <= 7 y <= 1
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(-10.0, -6.0);
      quadraticCostScalar = 34.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, 0.0}, {0.0, 1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(7.0, 1.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(2, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations + 1, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(5.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(4.0, lagrangeInequalityMultipliers.get(1), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(4.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 subject to x + y = 1.0, x <= y - 1 (x - y <= -1.0), but with y as inactive
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0}});
      DMatrixRMaj linearEqualityConstraintsBVector = MatrixTools.createVector(1.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, -1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(-1.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations + 1, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(0.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(-1.0, lagrangeEqualityMultipliers.get(0), 1e-7);
      assertEquals(1.0, lagrangeInequalityMultipliers.get(0), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(1.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 subject to x + y = 2.0, 3x - 3y = 0.0, x <= 2, x <= 10, y <= 3
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0}, {3.0, -3.0}});
      linearEqualityConstraintsBVector = MatrixTools.createVector(2.0, 0.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(2.0, 10.0, 3.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(2, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(3, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(1.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(-2.0, lagrangeEqualityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeEqualityMultipliers.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(2), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(2.0, objectiveCost, 1e-7);
   }

   @Test
   public void testSimpleCasesWithBoundsConstraints()
   {
      testSimpleCasesWithBoundsConstraints(1, 3, 3, 3, false);
   }

   public void testSimpleCasesWithBoundsConstraints(int expectedNumberOfIterations, int expectedNumberOfIterations2, int expectedNumberOfIterations3,
                                                    int expectedNumberOfIterations4, boolean ignoreLagrangeMultipliers)
   {
      ActiveSetQPSolver solver = createSolverToTest();

      // Minimize x^T * x
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj variableLowerBounds = MatrixTools.createVector(Double.NEGATIVE_INFINITY);
      DMatrixRMaj variableUpperBounds = MatrixTools.createVector(Double.POSITIVE_INFINITY);
      solver.setVariableBounds(variableLowerBounds, variableUpperBounds);

      DMatrixRMaj solution = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeLowerBoundMultipliers = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeUpperBoundMultipliers = new DMatrixRMaj(1, 1);

      int numberOfIterations = solver.solve(solution);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(0.0, solution.get(0), 1e-7);
      assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);

      // Minimize x^T * x subject to x >= 1
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      costLinearVector = MatrixTools.createVector(0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      variableLowerBounds = MatrixTools.createVector(1.0);
      variableUpperBounds = MatrixTools.createVector(Double.POSITIVE_INFINITY);
      solver.setVariableBounds(variableLowerBounds, variableUpperBounds);

      solution = new DMatrixRMaj(1, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(1, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations + 1, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(1.0, solution.get(0), 1e-7);
      assertEquals(2.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);

      // Minimize x^T * x subject to x <= -1
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      costLinearVector = MatrixTools.createVector(0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      variableLowerBounds = MatrixTools.createVector(Double.NEGATIVE_INFINITY);
      variableUpperBounds = MatrixTools.createVector(-1.0);
      solver.setVariableBounds(variableLowerBounds, variableUpperBounds);

      solution = new DMatrixRMaj(1, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(1, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations + 1, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(-1.0, solution.get(0), 1e-7);
      assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
      assertEquals(2.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);

      // Minimize x^T * x subject to 1 + 1e-12 <= x <= 1 - 1e-12 (Should give valid solution given a little epsilon to allow for roundoff)
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      costLinearVector = MatrixTools.createVector(0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      variableLowerBounds = MatrixTools.createVector(1.0 + 1e-12);
      variableUpperBounds = MatrixTools.createVector(1.0 - 1e-12);
      solver.setVariableBounds(variableLowerBounds, variableUpperBounds);

      solution = new DMatrixRMaj(1, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(1, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations + 1, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(1.0, solution.get(0), 1e-7);
      assertEquals(2.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);

      // Minimize x^T * x subject to -1 + 1e-12 <= x <= -1 - 1e-12 (Should give valid solution given a little epsilon to allow for roundoff)
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      costLinearVector = MatrixTools.createVector(0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      variableLowerBounds = MatrixTools.createVector(-1.0 + 1e-12);
      variableUpperBounds = MatrixTools.createVector(-1.0 - 1e-12);
      solver.setVariableBounds(variableLowerBounds, variableUpperBounds);

      solution = new DMatrixRMaj(1, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(1, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations + 1, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(-1.0, solution.get(0), 1e-7);
      assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
      assertEquals(2.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);

      // Minimize x^T * x subject to 1 + 1e-7 <= x <= 1 - 1e-7 (Should not give valid solution since this is too much to blame on roundoff)
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0}});
      costLinearVector = MatrixTools.createVector(0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      variableLowerBounds = MatrixTools.createVector(1.0 + 1e-7);
      variableUpperBounds = MatrixTools.createVector(1.0 - 1e-7);
      solver.setVariableBounds(variableLowerBounds, variableUpperBounds);

      solution = new DMatrixRMaj(1, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(1, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations2, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertTrue(Double.isNaN(solution.get(0)));
      assertFalse(Double.isFinite(lagrangeLowerBoundMultipliers.get(0)));
      assertFalse(Double.isFinite(lagrangeUpperBoundMultipliers.get(0)));

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, 11 <= z
      solver.clear();

      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0, 0.0}});
      DMatrixRMaj linearEqualityConstraintsBVector = MatrixTools.createVector(2.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{0.0, 1.0, -1.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(-8.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solver.setVariableBounds(getLowerBounds(), getUpperBounds());

      solution = new DMatrixRMaj(3, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(3, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(3, 1);

      numberOfIterations = solver.solve(solution);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations3, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertEquals(-4.0, solution.get(0), 1e-7);
      assertEquals(6.0, solution.get(1), 1e-7);
      assertEquals(14.0, solution.get(2), 1e-7);
      if (!ignoreLagrangeMultipliers)
      {
         assertEquals(8.0, lagrangeEqualityMultipliers.get(0), 1e-7);
         assertEquals(28.0, lagrangeInequalityMultipliers.get(0), 1e-7);

         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
         assertEquals(48.0, lagrangeLowerBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(2), 1e-7);

         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(2), 1e-7);
      }

      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(248.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, 3 <= x <= 5, 6 <= y <= 10, 11 <= z
      solver.clear();

      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0, 0.0}});
      linearEqualityConstraintsBVector = MatrixTools.createVector(2.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{0.0, 1.0, -1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(-8.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solver.setVariableBounds(MatrixTools.createVector(3.0, 6.0, 11.0), getUpperBounds());

      solution = new DMatrixRMaj(3, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(3, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(3, 1);

      numberOfIterations = solver.solve(solution);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations4, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertTrue(Double.isNaN(solution.get(0)));
      assertTrue(Double.isNaN(solution.get(1)));
      assertTrue(Double.isNaN(solution.get(2)));

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertTrue(Double.isNaN(objectiveCost));
   }

   @Test
   public void testClear()
   {
      testClear(3, 2, false);
   }

   public void testClear(int expectedNumberOfIterations1, int expectedNumberOfIterations2, boolean avoidProblematicLagrangeMultipliers)
   {
      ActiveSetQPSolver solver = createSolverToTest();

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, 11 <= z
      solver.clear();

      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0, 0.0, 0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0, 0.0}});
      DMatrixRMaj linearEqualityConstraintsBVector = MatrixTools.createVector(2.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{0.0, 1.0, -1.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(-8.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solver.setVariableBounds(getLowerBounds(), getUpperBounds());

      DMatrixRMaj solution1 = new DMatrixRMaj(3, 1);
      DMatrixRMaj solution2 = new DMatrixRMaj(3, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeLowerBoundMultipliers = new DMatrixRMaj(3, 1);
      DMatrixRMaj lagrangeUpperBoundMultipliers = new DMatrixRMaj(3, 1);

      int numberOfIterations = solver.solve(solution1);
      numberOfIterations = solver.solve(solution2);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      //assertEquals(expectedNumberOfIterations1, numberOfIterations);

      assertEquals(solution1.get(0), solution2.get(0), 1e-7);
      assertEquals(solution1.get(1), solution2.get(1), 1e-7);
      assertEquals(solution1.get(2), solution2.get(2), 1e-7);
      assertEquals(3, solution2.getNumRows());
      assertEquals(-4.0, solution2.get(0), 1e-7);
      assertEquals(6.0, solution2.get(1), 1e-7);
      assertEquals(14.0, solution2.get(2), 1e-7);

      if (!avoidProblematicLagrangeMultipliers)
      {
         assertEquals(8.0, lagrangeEqualityMultipliers.get(0), 1e-7);
         assertEquals(28.0, lagrangeInequalityMultipliers.get(0), 1e-7);

         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
         assertEquals(48.0, lagrangeLowerBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(2), 1e-7);

         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(2), 1e-7);
      }

      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution1);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(248.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8  (Remove -5 <= x <= 5, 6 <= y <= 10, 11 <= z)
      solver.clear();

      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0, 0.0}});
      linearEqualityConstraintsBVector = MatrixTools.createVector(2.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{0.0, 1.0, -1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(-8.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution1 = new DMatrixRMaj(3, 1);
      solution2 = new DMatrixRMaj(3, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 1);

      numberOfIterations = solver.solve(solution1);
      numberOfIterations = solver.solve(solution2);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations2, numberOfIterations);

      assertEquals(3, solution1.getNumRows());
      assertEquals(solution1.get(0), solution2.get(0), 1e-7);
      assertEquals(solution1.get(1), solution2.get(1), 1e-7);
      assertEquals(solution1.get(2), solution2.get(2), 1e-7);
      assertEquals(4.0, solution1.get(0), 1e-7);
      assertEquals(-2.0, solution1.get(1), 1e-7);
      assertEquals(6.0, solution1.get(2), 1e-7);
      assertEquals(-8.0, lagrangeEqualityMultipliers.get(0), 1e-7);
      assertEquals(12.0, lagrangeInequalityMultipliers.get(0), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution1);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(56.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, (Remove y - z <= -8)
      solver.clear();

      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0, 0.0}});
      linearEqualityConstraintsBVector = MatrixTools.createVector(2.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      solution1 = new DMatrixRMaj(3, 1);
      solution2 = new DMatrixRMaj(3, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 1);

      numberOfIterations = solver.solve(solution1);
      numberOfIterations = solver.solve(solution2);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(0, numberOfIterations);

      assertEquals(3, solution1.getNumRows());
      assertEquals(solution1.get(0), solution2.get(0), 1e-7);
      assertEquals(solution1.get(1), solution2.get(1), 1e-7);
      assertEquals(solution1.get(2), solution2.get(2), 1e-7);
      assertEquals(1.0, solution1.get(0), 1e-7);
      assertEquals(1.0, solution1.get(1), 1e-7);
      assertEquals(0.0, solution1.get(2), 1e-7);
      assertEquals(-2.0, lagrangeEqualityMultipliers.get(0), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution1);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(2.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 (Remove subject to x + y = 2.0)
      solver.clear();

      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      solution1 = new DMatrixRMaj(3, 1);
      solution2 = new DMatrixRMaj(3, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 1);

      numberOfIterations = solver.solve(solution1);
      numberOfIterations = solver.solve(solution2);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(0, numberOfIterations);

      assertEquals(3, solution1.getNumRows());
      assertEquals(solution1.get(0), solution2.get(0), 1e-7);
      assertEquals(solution1.get(1), solution2.get(1), 1e-7);
      assertEquals(solution1.get(2), solution2.get(2), 1e-7);
      assertEquals(0.0, solution1.get(0), 1e-7);
      assertEquals(0.0, solution1.get(1), 1e-7);
      assertEquals(0.0, solution1.get(2), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution1);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(0.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0 (Added without clearing)
      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0, 0.0}});
      linearEqualityConstraintsBVector = MatrixTools.createVector(2.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      solution1 = new DMatrixRMaj(3, 1);
      solution2 = new DMatrixRMaj(3, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 1);

      numberOfIterations = solver.solve(solution1);
      numberOfIterations = solver.solve(solution2);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(0, numberOfIterations);

      assertEquals(3, solution1.getNumRows());
      assertEquals(solution1.get(0), solution2.get(0), 1e-7);
      assertEquals(solution1.get(1), solution2.get(1), 1e-7);
      assertEquals(solution1.get(2), solution2.get(2), 1e-7);
      assertEquals(1.0, solution1.get(0), 1e-7);
      assertEquals(1.0, solution1.get(1), 1e-7);
      assertEquals(0.0, solution1.get(2), 1e-7);
      assertEquals(-2.0, lagrangeEqualityMultipliers.get(0), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution1);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(2.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8  (Added without clearing)
      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{0.0, 1.0, -1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(-8.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution1 = new DMatrixRMaj(3, 1);
      solution2 = new DMatrixRMaj(3, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 1);

      numberOfIterations = solver.solve(solution1);
      numberOfIterations = solver.solve(solution2);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations2, numberOfIterations);

      assertEquals(3, solution1.getNumRows());
      assertEquals(solution1.get(0), solution2.get(0), 1e-7);
      assertEquals(solution1.get(1), solution2.get(1), 1e-7);
      assertEquals(solution1.get(2), solution2.get(2), 1e-7);
      assertEquals(4.0, solution1.get(0), 1e-7);
      assertEquals(-2.0, solution1.get(1), 1e-7);
      assertEquals(6.0, solution1.get(2), 1e-7);
      assertEquals(-8.0, lagrangeEqualityMultipliers.get(0), 1e-7);
      assertEquals(12.0, lagrangeInequalityMultipliers.get(0), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution1);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(56.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, 11 <= z (Added without clearing)
      solver.setVariableBounds(getLowerBounds(), getUpperBounds());

      solution1 = new DMatrixRMaj(3, 1);
      solution2 = new DMatrixRMaj(3, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeLowerBoundMultipliers = new DMatrixRMaj(3, 1);
      lagrangeUpperBoundMultipliers = new DMatrixRMaj(3, 1);

      numberOfIterations = solver.solve(solution1);
      numberOfIterations = solver.solve(solution2);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(expectedNumberOfIterations1, numberOfIterations);

      assertEquals(3, solution1.getNumRows());
      assertEquals(solution1.get(0), solution2.get(0), 1e-7);
      assertEquals(solution1.get(1), solution2.get(1), 1e-7);
      assertEquals(solution1.get(2), solution2.get(2), 1e-7);
      assertEquals(-4.0, solution1.get(0), 1e-7);
      assertEquals(6.0, solution1.get(1), 1e-7);
      assertEquals(14.0, solution1.get(2), 1e-7);

      if (!avoidProblematicLagrangeMultipliers)
      {
         assertEquals(8.0, lagrangeEqualityMultipliers.get(0), 1e-7);
         assertEquals(28.0, lagrangeInequalityMultipliers.get(0), 1e-7);

         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
         assertEquals(48.0, lagrangeLowerBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(2), 1e-7);

         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(2), 1e-7);
      }

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution1);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(248.0, objectiveCost, 1e-7);
   }

   @Test
   public void testSolutionMethodsAreAllConsistent() throws NoConvergenceException
   {
      testSolutionMethodsAreAllConsistent(2);
   }

   public void testSolutionMethodsAreAllConsistent(int expectedNumberOfIterations) throws NoConvergenceException
   {
      ActiveSetQPSolver solver = createSolverToTest();

      // Minimize x^2 + y^2 subject to x + y = 2.0, y >= 0.5, y >= 3.0, y >= x-3  (-y <= -0.5, -y <= -3.0, x - y <= 3
      solver.clear();
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0, 0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0}});
      DMatrixRMaj linearEqualityConstraintsBVector = MatrixTools.createVector(2.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{0.0, -1.0}, {0.0, -1.0}, {1.0, -1.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(-0.5, -3.0, 3.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      DMatrixRMaj solution = new DMatrixRMaj(2, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(3, 1);
      int numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(-1.0, solution.get(0), 1e-7);
      assertEquals(3.0, solution.get(1), 1e-7);
      assertEquals(2.0, lagrangeEqualityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(8.0, lagrangeInequalityMultipliers.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(2), 1e-7);

      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(10.0, objectiveCost, 1e-7);

      // Try with other solve method:
      solver.clear();
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);
      numberOfIterations = solver.solve(solution);

      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(-1.0, solution.get(0), 1e-7);
      assertEquals(3.0, solution.get(1), 1e-7);
      assertEquals(2.0, lagrangeEqualityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(8.0, lagrangeInequalityMultipliers.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(2), 1e-7);

      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(10.0, objectiveCost, 1e-7);

      // Try with other solve method:
      solver.clear();
      DMatrixRMaj quadraticCostMatrix64F = new DMatrixRMaj(costQuadraticMatrix);
      DMatrixRMaj linearCostVector64F = new DMatrixRMaj(costLinearVector.getNumRows(), 1);
      linearCostVector64F.set(costLinearVector);

      solver.setQuadraticCostFunction(quadraticCostMatrix64F, linearCostVector64F, quadraticCostScalar);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      DMatrixRMaj solutionMatrix64F = new DMatrixRMaj(quadraticCostMatrix64F.getNumRows(), 1);
      DMatrixRMaj lagrangeEqualityMultipliers64F = new DMatrixRMaj(linearEqualityConstraintsAMatrix.getNumRows(), 1);
      DMatrixRMaj lagrangeInequalityMultipliers64F = new DMatrixRMaj(linearInequalityConstraintsCMatrix.getNumRows(), 1);
      numberOfIterations = solver.solve(solutionMatrix64F);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers64F);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers64F);

      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solutionMatrix64F.getNumRows());
      assertEquals(-1.0, solutionMatrix64F.get(0, 0), 1e-7);
      assertEquals(3.0, solutionMatrix64F.get(1, 0), 1e-7);
      assertEquals(2.0, lagrangeEqualityMultipliers64F.get(0, 0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers64F.get(0, 0), 1e-7);
      assertEquals(8.0, lagrangeInequalityMultipliers64F.get(1, 0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers64F.get(2, 0), 1e-7);

      objectiveCost = solver.getObjectiveCost(solutionMatrix64F);
      assertEquals(10.0, objectiveCost, 1e-7);

      // Try with other solve method:
      solver = createSolverToTest();

      solver.setQuadraticCostFunction(quadraticCostMatrix64F, linearCostVector64F, quadraticCostScalar);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      DMatrixRMaj linearInequalityConstraintsCMatrix64F = new DMatrixRMaj(linearInequalityConstraintsCMatrix);
      DMatrixRMaj linearInqualityConstraintsDVector64F = new DMatrixRMaj(linearInqualityConstraintsDVector.getNumRows(), 1);
      linearInqualityConstraintsDVector64F.set(linearInqualityConstraintsDVector);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix64F, linearInqualityConstraintsDVector64F);

      solutionMatrix64F.zero();
      numberOfIterations = solver.solve(solutionMatrix64F);

      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solutionMatrix64F.getNumRows());
      assertEquals(-1.0, solutionMatrix64F.get(0, 0), 1e-7);
      assertEquals(3.0, solutionMatrix64F.get(1, 0), 1e-7);

      objectiveCost = solver.getObjectiveCost(solutionMatrix64F);
      assertEquals(10.0, objectiveCost, 1e-7);
   }

   @Test
   public void test2DCasesWithPolygonConstraints()
   {
      test2DCasesWithPolygonConstraints(2, 3);
   }

   public void test2DCasesWithPolygonConstraints(int firstExpectedNumberOfIterations, int secondExpectedNumberOfIterations)
   {
      ActiveSetQPSolver solver = createSolverToTest();

      // Minimize x^2 + y^2 subject to 3 <= x <= 5, 2 <= y <= 4
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0, 0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, 0.0}, {-1.0, 0.0}, {0.0, 1.0}, {0.0, -1.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(5.0, -3.0, 4.0, -2.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      DMatrixRMaj solution = new DMatrixRMaj(2, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(4, 1);
      int numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(firstExpectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(3.0, solution.get(0), 1e-7);
      assertEquals(2.0, solution.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(6.0, lagrangeInequalityMultipliers.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(2), 1e-7);
      assertEquals(4.0, lagrangeInequalityMultipliers.get(3), 1e-7);

      // Minimize x^2 + y^2 subject to x + y >= 2 (-x -y <= -2), y <= 10x - 2 (-10x + y <= -2)
      // Equality solution will violate both constraints, but optimal only has the first constraint active.
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{-1.0, -1.0}, {-10.0, 1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(-2.0, -2.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(2, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(secondExpectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(1.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(2.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(1), 1e-7);
   }

   @Disabled // This should pass with a good solver. But a simple one has trouble on it.
   @Test
   public void testChallengingCasesWithPolygonConstraints()
   {
      testChallengingCasesWithPolygonConstraints(3, 3);
   }

   public void testChallengingCasesWithPolygonConstraints(int expectedNumberOfIterations1, int expectedNumberOfIterations2)
   {
      ActiveSetQPSolver solver = createSolverToTest();
      solver.setMaxNumberOfIterations(10);

      // Minimize x^2 + y^2 subject to x + y >= 2 (-x -y <= -2), y <= 10x - 2 (-10x + y <= -2), x <= 10y - 2 (x - 10y <= -2),
      // Equality solution will violate all three constraints, but optimal only has the first constraint active.
      // However, if you set all three constraints active, there is no solution.
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0, 0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{-1.0, -1.0}, {-10.0, 1.0}, {1.0, -10.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(-2.0, -2.0, -2.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      DMatrixRMaj solution = new DMatrixRMaj(2, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(3, 1);
      int numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations1, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(1.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(2.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(2), 1e-7);

      // Reorder and make sure it works:
      // Minimize x^2 + y^2 subject to x + y >= 2 (-x -y <= -2), y <= 10x - 2 (-10x + y <= -2), x <= 10y - 2 (x - 10y <= -2),
      // Equality solution will violate all three constraints, but optimal only has the first constraint active.
      // However, if you set all three constraints active, there is no solution.
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{-10.0, 1.0}, {-1.0, -1.0}, {1.0, -10.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(-2.0, -2.0, -2.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(3, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations2, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(1.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(2.0, lagrangeInequalityMultipliers.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(2), 1e-7);
   }

   // This should pass with a good solver. But a simple one has trouble on it.
   @Test
   public void testChallengingCasesWithPolygonConstraintsCheckFailsWithSimpleSolver()
   {
      ActiveSetQPSolver solver = createSolverToTest();
      solver.setMaxNumberOfIterations(10);

      // Minimize x^2 + y^2 subject to x + y >= 2 (-x -y <= -2), y <= 10x - 2 (-10x + y <= -2), x <= 10y - 2 (x - 10y <= -2),
      // Equality solution will violate all three constraints, but optimal only has the first constraint active.
      // However, if you set all three constraints active, there is no solution.
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0, 0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{-1.0, -1.0}, {-10.0, 1.0}, {1.0, -10.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(-2.0, -2.0, -2.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      DMatrixRMaj solution = new DMatrixRMaj(2, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(3, 1);
      solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(2, solution.getNumRows());
      assertTrue(Double.isNaN(solution.get(0)));
      assertTrue(Double.isNaN(solution.get(1)));
      assertTrue(Double.isInfinite(lagrangeInequalityMultipliers.get(0)) || Double.isNaN(lagrangeInequalityMultipliers.get(0)));
      assertTrue(Double.isInfinite(lagrangeInequalityMultipliers.get(1)) || Double.isNaN(lagrangeInequalityMultipliers.get(1)));
      assertTrue(Double.isInfinite(lagrangeInequalityMultipliers.get(2)) || Double.isNaN(lagrangeInequalityMultipliers.get(2)));
   }

   @Disabled /**
              * we can set this to be valid, via
              * {@link JavaQuadProgSolver.setRequireInequalityConstraintsSatisfied(boolean)} to true. But this
              * does not, by default require that
              */
   @Test
   public void testCaseWithNoSolution()
   {
      ActiveSetQPSolver solver = createSolverToTest();
      int maxNumberOfIterations = 10;
      solver.setMaxNumberOfIterations(maxNumberOfIterations);

      // Minimize x^2 + y^2 subject to x + y = 5, x + y <= 2
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0, 0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0}});
      DMatrixRMaj linearEqualityConstraintsBVector = MatrixTools.createVector(5.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(2.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      DMatrixRMaj solution = new DMatrixRMaj(2, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(2, solution.getNumRows());
      assertEquals(Double.NaN, solution.get(0), 1e-7);
      assertEquals(Double.NaN, solution.get(1), 1e-7);
      assertTrue(Double.isInfinite(lagrangeEqualityMultipliers.get(0)));
      assertTrue(Double.isInfinite(lagrangeInequalityMultipliers.get(0)));
   }

   @Test
   public void testLargeRandomProblemWithInequalityConstraints()
   {
      Random random = new Random(1776L);

      ActiveSetQPSolver solver = createSolverToTest();

      int numberOfTests = 100;

      long startTimeMillis = System.currentTimeMillis();
      int maxNumberOfIterations = 0;

      int numberOfVariables = 80;
      int numberOfEqualityConstraints = 10;
      int numberOfInequalityConstraints = 36;

      DMatrixRMaj solution = new DMatrixRMaj(numberOfVariables, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(numberOfEqualityConstraints, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(numberOfInequalityConstraints, 1);
      double[] solutionWithSmallPerturbation = new double[numberOfVariables];

      DMatrixRMaj augmentedLinearEqualityConstraintsAMatrix = new DMatrixRMaj(0, 0);
      DMatrixRMaj augmentedLinearEqualityConstraintsBVector = new DMatrixRMaj(0, 0);

      for (int testNumber = 0; testNumber < numberOfTests; testNumber++)
      {
         solver.clear();

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

         DMatrixRMaj linearInequalityConstraintsCMatrix = nextDMatrixRMaj(random, numberOfInequalityConstraints, numberOfVariables);
         DMatrixRMaj linearInequalityConstraintsDVector = nextDMatrixRMaj(random, numberOfInequalityConstraints, 1);
         solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector);

         int numberOfIterations = solver.solve(solution);
         solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
         solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
         if (numberOfIterations > maxNumberOfIterations)
            maxNumberOfIterations = numberOfIterations;
         //         System.out.println("numberOfIterations = " + numberOfIterations);

         assertEquals(numberOfVariables, solution.getNumRows());
         assertEquals(numberOfEqualityConstraints, lagrangeEqualityMultipliers.getNumRows());

         double objectiveCost = solver.getObjectiveCost(solution);

         // Verify constraints hold:
         verifyEqualityConstraintsHold(numberOfEqualityConstraints, linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector, solution);
         verifyInequalityConstraintsHold(numberOfInequalityConstraints, linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector, solution);

         // Verify objective is minimized by comparing to small perturbation:
         for (int i = 0; i < numberOfVariables; i++)
         {
            solutionWithSmallPerturbation[i] = solution.get(i, 0) + RandomNumbers.nextDouble(random, 1e-4);
         }

         solution.zero();
         solution.setData(solutionWithSmallPerturbation);

         verifyEqualityConstraintsDoNotHold(numberOfEqualityConstraints, linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector, solution);

         // Equality constraints usually do not hold. Sometimes they do, so if you run with lots of numberOfTests, comment out the following:
         verifyInequalityConstraintsDoNotHold(numberOfInequalityConstraints, linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector, solution);

         int activeSetSize = 0;
         for (int i = 0; i < numberOfInequalityConstraints; i++)
         {
            double lagrangeMultiplier = lagrangeInequalityMultipliers.get(i, 0);

            if (lagrangeMultiplier < 0.0)
            {
               throw new RuntimeException("Received a negative lagrange multiplier.");
            }
            if (lagrangeMultiplier > 0.0)
            {
               activeSetSize++;
            }
         }

         augmentedLinearEqualityConstraintsAMatrix.reshape(numberOfEqualityConstraints + activeSetSize, numberOfVariables);
         augmentedLinearEqualityConstraintsBVector.reshape(numberOfEqualityConstraints + activeSetSize, 1);

         CommonOps_DDRM.extract(linearEqualityConstraintsAMatrix,
                           0,
                           numberOfEqualityConstraints,
                           0,
                           numberOfVariables,
                           augmentedLinearEqualityConstraintsAMatrix,
                           0,
                           0);
         CommonOps_DDRM.extract(linearEqualityConstraintsBVector, 0, numberOfEqualityConstraints, 0, 1, augmentedLinearEqualityConstraintsBVector, 0, 0);

         int index = 0;
         for (int i = 0; i < numberOfInequalityConstraints; i++)
         {
            double lagrangeMultiplier = lagrangeInequalityMultipliers.get(i, 0);

            if (lagrangeMultiplier < 0.0)
            {
               throw new RuntimeException();
            }
            if (lagrangeMultiplier > 0.0)
            {
               CommonOps_DDRM.extract(linearInequalityConstraintsCMatrix,
                                 i,
                                 i + 1,
                                 0,
                                 numberOfVariables,
                                 augmentedLinearEqualityConstraintsAMatrix,
                                 numberOfEqualityConstraints + index,
                                 0);
               CommonOps_DDRM.extract(linearInequalityConstraintsDVector,
                                 i,
                                 i + 1,
                                 0,
                                 1,
                                 augmentedLinearEqualityConstraintsBVector,
                                 numberOfEqualityConstraints + index,
                                 0);
               index++;
            }
         }

         DMatrixRMaj solutionMatrixProjectedOntoEqualityConstraints = projectOntoEqualityConstraints(solution,
                                                                                                        augmentedLinearEqualityConstraintsAMatrix,
                                                                                                        augmentedLinearEqualityConstraintsBVector);
         verifyEqualityConstraintsHold(numberOfEqualityConstraints + activeSetSize,
                                       augmentedLinearEqualityConstraintsAMatrix,
                                       augmentedLinearEqualityConstraintsBVector,
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
         System.out.println("maxNumberOfIterations is " + maxNumberOfIterations);
      }
   }

   @Test
   public void testLargeRandomProblemWithInequalityAndBoundsConstraints()
   {
      Random random = new Random(1776L);

      ActiveSetQPSolver solver = createSolverToTest();

      int numberOfTests = 100;

      long startTimeMillis = System.currentTimeMillis();
      int maxNumberOfIterations = 0;

      int numberOfVariables = 80;
      int numberOfEqualityConstraints = 10;
      int numberOfInequalityConstraints = 36;

      DMatrixRMaj solution = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeLowerBoundMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj lagrangeUpperBoundMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj solutionWithSmallPerturbation = new DMatrixRMaj(numberOfVariables, 1);

      DMatrixRMaj augmentedLinearEqualityConstraintsAMatrix = new DMatrixRMaj(0, 1);
      DMatrixRMaj augmentedLinearEqualityConstraintsBVector = new DMatrixRMaj(0, 1);

      int numberOfNaNSolutions = 0;
      for (int testNumber = 0; testNumber < numberOfTests; testNumber++)
      {
         //         System.out.println("testNumber = " + testNumber);
         solver.clear();

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

         DMatrixRMaj linearInequalityConstraintsCMatrix = nextDMatrixRMaj(random, numberOfInequalityConstraints, numberOfVariables);
         DMatrixRMaj linearInequalityConstraintsDVector = nextDMatrixRMaj(random, numberOfInequalityConstraints, 1);
         solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector);

         DMatrixRMaj variableLowerBounds = nextDMatrixRMaj(random, numberOfVariables, 1, -5.0, -0.01);
         DMatrixRMaj variableUpperBounds = nextDMatrixRMaj(random, numberOfVariables, 1, 0.01, 5.0);
         solver.setVariableBounds(variableLowerBounds, variableUpperBounds);

         solution.reshape(numberOfVariables, 1);
         int numberOfIterations = solver.solve(solution);
         solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
         solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
         solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
         solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);

         if (numberOfIterations > maxNumberOfIterations)
            maxNumberOfIterations = numberOfIterations;

         //         System.out.println(solution);
         //         System.out.println("numberOfIterations = " + numberOfIterations);

         assertEquals(numberOfVariables, solution.getNumRows());
         assertEquals(numberOfEqualityConstraints, lagrangeEqualityMultipliers.getNumRows());
         assertEquals(numberOfInequalityConstraints, lagrangeInequalityMultipliers.getNumRows());
         assertEquals(variableLowerBounds.getNumRows(), lagrangeLowerBoundMultipliers.getNumRows());
         assertEquals(variableUpperBounds.getNumRows(), lagrangeUpperBoundMultipliers.getNumRows());

         if (Double.isNaN(solution.get(0)))
         {
            numberOfNaNSolutions++;
            continue;
         }

         double objectiveCost = solver.getObjectiveCost(solution);

         // Verify constraints hold:
         verifyEqualityConstraintsHold(numberOfEqualityConstraints, linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector, solution);
         verifyInequalityConstraintsHold(numberOfInequalityConstraints, linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector, solution);
         verifyVariableBoundsHold(testNumber, variableLowerBounds, variableUpperBounds, solution);

         // Verify objective is minimized by comparing to small perturbation:
         for (int i = 0; i < numberOfVariables; i++)
         {
            solutionWithSmallPerturbation.set(i, solution.get(i, 0) + RandomNumbers.nextDouble(random, 5e-3));
         }

         solution.zero();
         solution.set(solutionWithSmallPerturbation);

         verifyEqualityConstraintsDoNotHold(numberOfEqualityConstraints, linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector, solution);

         // Equality constraints usually do not hold. Sometimes they do, so if you run with lots of numberOfTests, comment out the following:
         verifyInequalityConstraintsDoNotHold(numberOfInequalityConstraints, linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector, solution);

         int activeInequalitiesSize = 0;
         for (int i = 0; i < numberOfInequalityConstraints; i++)
         {
            double lagrangeMultiplier = lagrangeInequalityMultipliers.get(i, 0);

            if (lagrangeMultiplier < 0.0)
            {
               throw new RuntimeException("Received a negative lagrange multiplier.");
            }
            if (lagrangeMultiplier > 0.0)
            {
               activeInequalitiesSize++;
            }
         }

         int activeLowerBoundsSize = 0;
         for (int i = 0; i < variableLowerBounds.getNumRows(); i++)
         {
            double lagrangeMultiplier = lagrangeLowerBoundMultipliers.get(i, 0);

            if (lagrangeMultiplier < 0.0)
            {
               throw new RuntimeException();
            }
            if (lagrangeMultiplier > 0.0)
            {
               activeLowerBoundsSize++;
            }
         }

         int activeUpperBoundsSize = 0;
         for (int i = 0; i < variableUpperBounds.getNumRows(); i++)
         {
            double lagrangeMultiplier = lagrangeUpperBoundMultipliers.get(i, 0);

            if (lagrangeMultiplier < 0.0)
            {
               throw new RuntimeException();
            }
            if (lagrangeMultiplier > 0.0)
            {
               activeUpperBoundsSize++;
            }
         }

         augmentedLinearEqualityConstraintsAMatrix.reshape(numberOfEqualityConstraints + activeInequalitiesSize + activeLowerBoundsSize + activeUpperBoundsSize,
                                                           numberOfVariables);
         augmentedLinearEqualityConstraintsBVector.reshape(numberOfEqualityConstraints + activeInequalitiesSize + activeLowerBoundsSize + activeUpperBoundsSize,
                                                           1);
         augmentedLinearEqualityConstraintsAMatrix.zero();
         augmentedLinearEqualityConstraintsBVector.zero();

         CommonOps_DDRM.extract(linearEqualityConstraintsAMatrix,
                           0,
                           numberOfEqualityConstraints,
                           0,
                           numberOfVariables,
                           augmentedLinearEqualityConstraintsAMatrix,
                           0,
                           0);
         CommonOps_DDRM.extract(linearEqualityConstraintsBVector, 0, numberOfEqualityConstraints, 0, 1, augmentedLinearEqualityConstraintsBVector, 0, 0);

         int index = 0;
         for (int i = 0; i < numberOfInequalityConstraints; i++)
         {
            double lagrangeMultiplier = lagrangeInequalityMultipliers.get(i, 0);

            if (lagrangeMultiplier < 0.0)
            {
               throw new RuntimeException();
            }
            if (lagrangeMultiplier > 0.0)
            {
               CommonOps_DDRM.extract(linearInequalityConstraintsCMatrix,
                                 i,
                                 i + 1,
                                 0,
                                 numberOfVariables,
                                 augmentedLinearEqualityConstraintsAMatrix,
                                 numberOfEqualityConstraints + index,
                                 0);
               CommonOps_DDRM.extract(linearInequalityConstraintsDVector,
                                 i,
                                 i + 1,
                                 0,
                                 1,
                                 augmentedLinearEqualityConstraintsBVector,
                                 numberOfEqualityConstraints + index,
                                 0);
               index++;
            }
         }

         for (int i = 0; i < variableLowerBounds.getNumRows(); i++)
         {
            double lagrangeMultiplier = lagrangeLowerBoundMultipliers.get(i, 0);

            if (lagrangeMultiplier > 0.0)
            {
               augmentedLinearEqualityConstraintsAMatrix.set(numberOfEqualityConstraints + index, i, -1.0);
               augmentedLinearEqualityConstraintsBVector.set(numberOfEqualityConstraints + index, -variableLowerBounds.get(i));
               index++;
            }
         }

         for (int i = 0; i < variableUpperBounds.getNumRows(); i++)
         {
            double lagrangeMultiplier = lagrangeUpperBoundMultipliers.get(i, 0);

            if (lagrangeMultiplier > 0.0)
            {
               augmentedLinearEqualityConstraintsAMatrix.set(numberOfEqualityConstraints + index, i, 1.0);
               augmentedLinearEqualityConstraintsBVector.set(numberOfEqualityConstraints + index, variableUpperBounds.get(i));
               index++;
            }
         }

         assertTrue(index == activeInequalitiesSize + activeLowerBoundsSize + activeUpperBoundsSize);

         DMatrixRMaj solutionMatrixProjectedOntoEqualityConstraints = projectOntoEqualityConstraints(solution,
                                                                                                        augmentedLinearEqualityConstraintsAMatrix,
                                                                                                        augmentedLinearEqualityConstraintsBVector);
         verifyEqualityConstraintsHold(numberOfEqualityConstraints + activeInequalitiesSize + activeLowerBoundsSize + activeUpperBoundsSize,
                                       augmentedLinearEqualityConstraintsAMatrix,
                                       augmentedLinearEqualityConstraintsBVector,
                                       solutionMatrixProjectedOntoEqualityConstraints);

         double maxSignedError = getMaxInequalityConstraintError(numberOfInequalityConstraints,
                                                                 linearInequalityConstraintsCMatrix,
                                                                 linearInequalityConstraintsDVector,
                                                                 solutionMatrixProjectedOntoEqualityConstraints);

         double objectiveCostWithSmallPerturbation = solver.getObjectiveCost(solutionMatrixProjectedOntoEqualityConstraints);

         if (maxSignedError < 1.0e-7) // Java quad prog does not necessarily include the correct form of equality constraints, so this must be considered.
         {
            assertTrue(objectiveCostWithSmallPerturbation > objectiveCost,
                       "objectiveCostWithSmallPerturbation = " + objectiveCostWithSmallPerturbation + ", objectiveCost = " + objectiveCost);
         }
      }

      assertTrue(numberOfNaNSolutions < 0.05 * numberOfTests);

      long endTimeMillis = System.currentTimeMillis();

      double timePerTest = (endTimeMillis - startTimeMillis) * 0.001 / numberOfTests;
      if (VERBOSE)
      {
         System.out.println("Time per test is " + timePerTest);
         System.out.println("maxNumberOfIterations is " + maxNumberOfIterations);
         System.out.println("numberOfNaNSolutions = " + numberOfNaNSolutions);
         System.out.println("numberOfTests = " + numberOfTests);

      }
   }

   /**
    * Test with dataset from sim that revealed a bug with the variable lower/upper bounds handling.
    */
   @Test
   public void testFindValidSolutionForDataset20160319()
   {
      ActualDatasetFrom20160319 dataset = new ActualDatasetFrom20160319();
      ActiveSetQPSolver solver = createSolverToTest();
      solver.clear();
      solver.setQuadraticCostFunction(dataset.getCostQuadraticMatrix(), dataset.getCostLinearVector(), 0.0);
      solver.setVariableBounds(dataset.getVariableLowerBounds(), dataset.getVariableUpperBounds());
      DMatrixRMaj solution = new DMatrixRMaj(dataset.getProblemSize(), 1);
      solver.solve(solution);

      assertFalse(MatrixTools.containsNaN(solution));
   }

   /**
    * Test with dataset of a Kiwi simulation performing a fast pace gait. It seems that the problem is
    * related to the fact that the robot has 6 contact points per foot. The solver still fails when
    * increasing the max number of iterations.
    */
   @Test
   public void testFindValidSolutionForKiwiDataset20170712()
   {
      ActualDatasetFromKiwi20170712 dataset = new ActualDatasetFromKiwi20170712();
      ActiveSetQPSolver solver = createSolverToTest();
      solver.clear();
      solver.setQuadraticCostFunction(dataset.getCostQuadraticMatrix(), dataset.getCostLinearVector(), 0.0);
      solver.setVariableBounds(dataset.getVariableLowerBounds(), dataset.getVariableUpperBounds());
      DMatrixRMaj solution = new DMatrixRMaj(dataset.getProblemSize(), 1);
      solver.solve(solution);

      assertFalse(MatrixTools.containsNaN(solution));
   }

   /**
    * Test with dataset of a Kiwi simulation walking backward. It seems that the problem is related to
    * the fact that the robot has 6 contact points per foot. The solver still fails when increasing the
    * max number of iterations.
    */
   @Test
   public void testFindValidSolutionForKiwiDataset20171013()
   {
      ActualDatasetFromKiwi20171013 dataset = new ActualDatasetFromKiwi20171013();
      ActiveSetQPSolver solver = createSolverToTest();
      solver.clear();
      solver.setQuadraticCostFunction(dataset.getCostQuadraticMatrix(), dataset.getCostLinearVector(), 0.0);
      solver.setVariableBounds(dataset.getVariableLowerBounds(), dataset.getVariableUpperBounds());
      DMatrixRMaj solution = new DMatrixRMaj(dataset.getProblemSize(), 1);
      solver.solve(solution);

      assertFalse(MatrixTools.containsNaN(solution));
   }

   @Test
   public void testMaxIterations()
   {
      testMaxIterations(3, true);
   }

   public void testMaxIterations(int maxForSolution, boolean checkLagrangeMultipliers)
   {
      ActiveSetQPSolver solver = createSolverToTest();

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, 11 <= z
      solver.clear();

      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0, 0.0, 0.0);
      double quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0, 0.0}});
      DMatrixRMaj linearEqualityConstraintsBVector = MatrixTools.createVector(2.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{0.0, 1.0, -1.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(-8.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solver.setVariableBounds(getLowerBounds(), getUpperBounds());

      DMatrixRMaj solution = new DMatrixRMaj(3, 1);
      DMatrixRMaj lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      DMatrixRMaj lagrangeLowerBoundMultipliers = new DMatrixRMaj(3, 1);
      DMatrixRMaj lagrangeUpperBoundMultipliers = new DMatrixRMaj(3, 1);

      solver.setMaxNumberOfIterations(maxForSolution - 1);
      int numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(maxForSolution - 1, numberOfIterations);

      assertTrue(Double.isNaN(solution.get(0)));
      assertTrue(Double.isNaN(solution.get(1)));
      assertTrue(Double.isNaN(solution.get(2)));

      solver.setMaxNumberOfIterations(maxForSolution);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      solver.getLagrangeLowerBoundsMultipliers(lagrangeLowerBoundMultipliers);
      solver.getLagrangeUpperBoundsMultipliers(lagrangeUpperBoundMultipliers);
      assertEquals(maxForSolution, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertEquals(-4.0, solution.get(0), 1e-7);
      assertEquals(6.0, solution.get(1), 1e-7);
      assertEquals(14.0, solution.get(2), 1e-7);
      /** These lagrange multipliers cause problems */
      if (checkLagrangeMultipliers)
      {
         assertEquals(8.0, lagrangeEqualityMultipliers.get(0), 1e-7);
         assertEquals(28.0, lagrangeInequalityMultipliers.get(0), 1e-7);

         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
         assertEquals(48.0, lagrangeLowerBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(2), 1e-7);

         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(2), 1e-7);
      }

      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(248.0, objectiveCost, 1e-7);
   }

   public DMatrixRMaj getUpperBounds()
   {
      return MatrixTools.createVector(5.0, 10.0, Double.POSITIVE_INFINITY);
   }

   public DMatrixRMaj getLowerBounds()
   {
      return MatrixTools.createVector(-5.0, 6.0, 11.0);
   }

   @Test
   public void testSomeExceptions()
   {
      ActiveSetQPSolver solver = createSolverToTest();

      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0);
      double quadraticCostScalar = 0.0;

      try
      {
         solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);
         fail("costQuadraticMatrix needs to be square!");
      }
      catch (RuntimeException e)
      {
      }

      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      try
      {
         solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);
         fail("costQuadraticMatrix needs to be same length as costLinearVector!");
      }
      catch (RuntimeException e)
      {
      }

      try
      {
         DMatrixRMaj costQuadraticMatrix64F = new DMatrixRMaj(2, 2);
         DMatrixRMaj costLinearVector64F = new DMatrixRMaj(1, 2);

         solver.setQuadraticCostFunction(costQuadraticMatrix64F, costLinearVector64F, quadraticCostScalar);
         fail("Wrong size for costLinearVector64F.");
      }
      catch (RuntimeException e)
      {
      }

      costLinearVector = MatrixTools.createVector(0.0, 0.0);
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      // Variable Bounds
      DMatrixRMaj variableLowerBounds = MatrixTools.createVector(0.0);
      DMatrixRMaj variableUpperBounds = MatrixTools.createVector(0.0, 0.0);

      try
      {
         solver.setVariableBounds(variableLowerBounds, variableUpperBounds);
         fail("Wrong lower bounds size");
      }
      catch (RuntimeException e)
      {
      }

      variableLowerBounds = MatrixTools.createVector(0.0, 0.0);
      variableUpperBounds = MatrixTools.createVector(0.0);

      try
      {
         solver.setVariableBounds(variableLowerBounds, variableUpperBounds);
         fail("Wrong upper bounds size");
      }
      catch (RuntimeException e)
      {
      }

      variableLowerBounds = MatrixTools.createVector(0.0, 0.0);
      variableUpperBounds = MatrixTools.createVector(0.0, 0.0);

      solver.setVariableBounds(variableLowerBounds, variableUpperBounds);

      // Equality Constraints
      DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 0.0}});
      DMatrixRMaj linearEqualityConstraintsBVector = MatrixTools.createVector(1.0, 2.0);

      try
      {
         solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);
         fail("Wrong size for linearEqualityConstraintsBVector.");
      }
      catch (RuntimeException e)
      {
      }

      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0}});
      linearEqualityConstraintsBVector = MatrixTools.createVector(1.0);
      try
      {
         solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);
         fail("Wrong size for linearEqualityConstraintsBVector.");
      }
      catch (RuntimeException e)
      {
      }

      try
      {
         DMatrixRMaj linearEqualityConstraintsAMatrix64F = new DMatrixRMaj(2, 2);
         DMatrixRMaj linearEqualityConstraintsBVector64F = new DMatrixRMaj(1, 2);

         solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix64F, linearEqualityConstraintsBVector64F);
         fail("Wrong size for linearEqualityConstraintsBVector.");
      }
      catch (RuntimeException e)
      {
      }

      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 0.0}});
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      // Inequality Constraints
      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, 0.0}});
      DMatrixRMaj linearInequalityConstraintsDVector = MatrixTools.createVector(1.0, 2.0);

      try
      {
         solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector);
         fail("Wrong size for linearInqualityConstraintsDVector.");
      }
      catch (RuntimeException e)
      {
      }

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0}});
      linearInequalityConstraintsDVector = MatrixTools.createVector(1.0);
      try
      {
         solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector);
         fail("Wrong size for linearInqualityConstraintsDVector.");
      }
      catch (RuntimeException e)
      {
      }

      try
      {
         DMatrixRMaj linearInequalityConstraintsCMatrix64F = new DMatrixRMaj(2, 2);
         DMatrixRMaj linearInequalityConstraintsDVector64F = new DMatrixRMaj(1, 2);

         solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix64F, linearInequalityConstraintsDVector64F);
         fail("Wrong size for linearInequalityConstraintsDVector64F.");
      }
      catch (RuntimeException e)
      {
      }

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, 0.0}});
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInequalityConstraintsDVector);
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

   private void verifyInequalityConstraintsHold(int numberOfEqualityConstraints, DMatrixRMaj linearInequalityConstraintsCMatrix,
                                                DMatrixRMaj linearInequalityConstraintsDVector, DMatrixRMaj solutionMatrix)
   {
      double maxSignedError = getMaxInequalityConstraintError(numberOfEqualityConstraints,
                                                              linearInequalityConstraintsCMatrix,
                                                              linearInequalityConstraintsDVector,
                                                              solutionMatrix);
      assertTrue(maxSignedError < 1e-10);
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

   private void verifyInequalityConstraintsDoNotHold(int numberOfInequalityConstraints, DMatrixRMaj linearInequalityConstraintsCMatrix,
                                                     DMatrixRMaj linearInequalityConstraintsDVector, DMatrixRMaj solutionMatrix)
   {
      double maxError = getMaxInequalityConstraintError(numberOfInequalityConstraints,
                                                        linearInequalityConstraintsCMatrix,
                                                        linearInequalityConstraintsDVector,
                                                        solutionMatrix);
      assertTrue(maxError > 1e-5);
   }

   private void verifyVariableBoundsHold(int testNumber, DMatrixRMaj variableLowerBounds, DMatrixRMaj variableUpperBounds, DMatrixRMaj solution)
   {
      for (int i = 0; i < variableLowerBounds.getNumRows(); i++)
      {
         assertTrue(solution.get(i, 0) >= variableLowerBounds.get(i, 0) - 1e-7,
                    "In test number " + testNumber + " the solution " + solution.get(i, 0) + " is less than the lower bound " + variableLowerBounds.get(i, 0));
      }

      for (int i = 0; i < variableUpperBounds.getNumRows(); i++)
      {
         assertTrue(solution.get(i, 0) <= variableUpperBounds.get(i, 0) + 1e-7);
      }
   }

   private double getMaxEqualityConstraintError(int numberOfEqualityConstraints, DMatrixRMaj linearEqualityConstraintsAMatrix,
                                                DMatrixRMaj linearEqualityConstraintsBVector, DMatrixRMaj solutionMatrix)
   {
      DMatrixRMaj checkMatrix = new DMatrixRMaj(numberOfEqualityConstraints, 1);
      CommonOps_DDRM.mult(linearEqualityConstraintsAMatrix, solutionMatrix, checkMatrix);
      CommonOps_DDRM.subtractEquals(checkMatrix, linearEqualityConstraintsBVector);

      return getMaxAbsoluteDataEntry(checkMatrix);
   }

   private double getMaxInequalityConstraintError(int numberOfInequalityConstraints, DMatrixRMaj linearInequalityConstraintsCMatrix,
                                                  DMatrixRMaj linearInequalityConstraintsDVector, DMatrixRMaj solutionMatrix)
   {
      DMatrixRMaj checkMatrix = new DMatrixRMaj(numberOfInequalityConstraints, 1);
      CommonOps_DDRM.mult(linearInequalityConstraintsCMatrix, solutionMatrix, checkMatrix);
      CommonOps_DDRM.subtractEquals(checkMatrix, linearInequalityConstraintsDVector);

      return getMaxSignedDataEntry(checkMatrix);
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

   private double getMaxSignedDataEntry(DMatrixRMaj matrix)
   {
      int numberOfRows = matrix.getNumRows();
      int numberOfColumns = matrix.getNumCols();

      double max = Double.NEGATIVE_INFINITY;

      for (int i = 0; i < numberOfRows; i++)
      {
         for (int j = 0; j < numberOfColumns; j++)
         {
            double value = matrix.get(i, j);
            if (value > max)
            {
               max = value;
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
