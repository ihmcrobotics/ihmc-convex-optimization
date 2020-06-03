package us.ihmc.convexOptimization.quadraticProgram;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.jupiter.api.Test;

import us.ihmc.matrixlib.MatrixTools;

public abstract class AbstractSimpleActiveSetQPSolverWithInactiveVariablesTest extends AbstractSimpleActiveSetQPSolverTest
{
   @Override
   public abstract ActiveSetQPSolverWithInactiveVariablesInterface createSolverToTest();

   @Test
   public void testSimpleCasesWithInequalityConstraintsAndInactiveVariables()
   {
      testSimpleCasesWithInequalityConstraintsAndInactiveVariables(1);
   }

   public void testSimpleCasesWithInequalityConstraintsAndInactiveVariables(int expectedNumberOfIterations)
   {
      ActiveSetQPSolverWithInactiveVariablesInterface solver = createSolverToTest();

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
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);

      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(1, solution.getNumRows());
      assertEquals(0.0, solution.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);

      // Minimize (x-5) * (x-5) + (y-3) * (y-3) = 1/2 * (2x^2 + 2y^2) - 10x -6y + 34 subject to x <= 7 y <= 1, with y inactive
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(-10.0, -6.0);
      quadraticCostScalar = 34.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      DMatrixRMaj activeVariables = MatrixTools.createVector(1.0, 0.0);
      solver.setActiveVariables(activeVariables);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, 0.0}, {0.0, 1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(7.0, 1.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(2, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(5.0, solution.get(0), 1e-7);
      assertEquals(0.0, solution.get(1), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(1), 1e-7);

      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(9.0, objectiveCost, 1e-7);

      // Minimize (x-5) * (x-5) + (y-3) * (y-3) = 1/2 * (2x^2 + 2y^2) - 10x -6y + 34 subject to x <= 7 y <= 1, with x inactive
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(-10.0, -6.0);
      quadraticCostScalar = 34.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      activeVariables = MatrixTools.createVector(0.0, 1.0);
      solver.setActiveVariables(activeVariables);

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
      assertEquals(0.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(4.0, lagrangeInequalityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(1), 1e-7);

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(29.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 subject to x + y = 1.0, x <= y - 1 (x - y <= -1.0), x inactive
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

      activeVariables = MatrixTools.createVector(1.0, 0.0);
      solver.setActiveVariables(activeVariables);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);

      assertEquals(2, solution.getNumRows());
      assertTrue(Double.isNaN(solution.get(0)));
      assertTrue(Double.isNaN(solution.get(1)));

      // Minimize x^2 + y^2 subject to x + y = 1.0, x <= y - 1 (x - y <= -1.0), y inactive
      solver.clear();
      costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      costLinearVector = MatrixTools.createVector(0.0, 0.0);
      quadraticCostScalar = 0.0;
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);

      linearEqualityConstraintsAMatrix = new DMatrixRMaj(new double[][] {{1.0, 1.0}});
      linearEqualityConstraintsBVector = MatrixTools.createVector(1.0);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{1.0, -1.0}});
      linearInqualityConstraintsDVector = MatrixTools.createVector(-1.0);
      solver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

      activeVariables = MatrixTools.createVector(0.0, 1.0);
      solver.setActiveVariables(activeVariables);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(1, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(1, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(2, solution.getNumRows());
      assertEquals(0.0, solution.get(0), 1e-7);
      assertEquals(1.0, solution.get(1), 1e-7);
      assertEquals(-2.0, lagrangeEqualityMultipliers.get(0), 1e-7);
      assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);

      // Minimize x^2 + y^2 subject to x + y = 2.0, 3x - 3y = 0.0, x <= 2, x <= 10, y <= 3, x active
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

      activeVariables = MatrixTools.createVector(1.0, 0.0);
      solver.setActiveVariables(activeVariables);

      solution = new DMatrixRMaj(2, 1);
      lagrangeEqualityMultipliers = new DMatrixRMaj(2, 1);
      lagrangeInequalityMultipliers = new DMatrixRMaj(3, 1);
      numberOfIterations = solver.solve(solution);
      solver.getLagrangeEqualityConstraintMultipliers(lagrangeEqualityMultipliers);
      solver.getLagrangeInequalityConstraintMultipliers(lagrangeInequalityMultipliers);
      assertTrue(numberOfIterations <= 1);

      assertEquals(2, solution.getNumRows());
      assertTrue(Double.isNaN(solution.get(0)));
      assertTrue(Double.isNaN(solution.get(1)));
   }

   @Test
   public void testSimpleCasesWithBoundsConstraintsAndInactiveVariables()
   {
      testSimpleCasesWithBoundsConstraintsAndInactiveVariables(1, 2, 3, 2, false);
   }

   public void testSimpleCasesWithBoundsConstraintsAndInactiveVariables(int expectedNumberOfIterations, int expectedNumberOfIterations2,
                                                                        int expectedNubmerOfIterations3, int expectedNumberOfIterations4,
                                                                        boolean ignoreLagrangeMultipliers)
   {
      ActiveSetQPSolverWithInactiveVariablesInterface solver = createSolverToTest();

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

      // minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 1 <= y <= 10, -2 <= z, y and z active
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

      solver.setVariableBounds(MatrixTools.createVector(-5.0, 1.0, -2.0), MatrixTools.createVector(5.0, 10.0, Double.POSITIVE_INFINITY));

      DMatrixRMaj activeVariables = MatrixTools.createVector(0.0, 1.0, 1.0);
      solver.setActiveVariables(activeVariables);

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
      assertEquals(expectedNumberOfIterations2, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertEquals(0.0, solution.get(0), 1e-7);
      assertEquals(2.0, solution.get(1), 1e-7);
      assertEquals(10.0, solution.get(2), 1e-7);
      if (!ignoreLagrangeMultipliers)
      {
         assertEquals(-24.0, lagrangeEqualityMultipliers.get(0), 1e-7);
         assertEquals(20.0, lagrangeInequalityMultipliers.get(0), 1e-7);

         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(2), 1e-7);

         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(2), 1e-7);
      }

      DMatrixRMaj solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      double objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(104.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, -2 <= z, x and z active
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

      solver.setVariableBounds(MatrixTools.createVector(-5.0, 6.0, -2.0), MatrixTools.createVector(5.0, 10.0, Double.POSITIVE_INFINITY));

      activeVariables = MatrixTools.createVector(1.0, 0.0, 1.0);
      solver.setActiveVariables(activeVariables);

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
      assertEquals(expectedNumberOfIterations2, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertEquals(2.0, solution.get(0), 1e-7);
      assertEquals(0.0, solution.get(1), 1e-7);
      assertEquals(8.0, solution.get(2), 1e-7);
      if (!ignoreLagrangeMultipliers)
      {
         assertEquals(-4.0, lagrangeEqualityMultipliers.get(0), 1e-7);
         assertEquals(16.0, lagrangeInequalityMultipliers.get(0), 1e-7);

         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(2), 1e-7);

         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(2), 1e-7);
      }

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(68.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, -2 <= z, x and y active
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

      solver.setVariableBounds(MatrixTools.createVector(-5.0, 6.0, -2.0), MatrixTools.createVector(5.0, 10.0, Double.POSITIVE_INFINITY));

      activeVariables = MatrixTools.createVector(1.0, 1.0, 0.0);
      solver.setActiveVariables(activeVariables);

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
      assertEquals(expectedNumberOfIterations2, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertTrue(Double.isNaN(solution.get(0)));
      assertTrue(Double.isNaN(solution.get(1)));
      assertTrue(Double.isNaN(solution.get(2)));

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, -2 <= z, x active
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

      solver.setVariableBounds(MatrixTools.createVector(-5.0, 6.0, -2.0), MatrixTools.createVector(5.0, 10.0, Double.POSITIVE_INFINITY));

      activeVariables = MatrixTools.createVector(1.0, 0.0, 0.0);
      solver.setActiveVariables(activeVariables);

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
      assertEquals(expectedNumberOfIterations, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertEquals(2.0, solution.get(0), 1e-7);
      assertEquals(0.0, solution.get(1), 1e-7);
      assertEquals(0.0, solution.get(2), 1e-7);
      if (!ignoreLagrangeMultipliers)
      {
         assertEquals(-4.0, lagrangeEqualityMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeInequalityMultipliers.get(0), 1e-7);

         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(2), 1e-7);

         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(2), 1e-7);
      }

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(4.0, objectiveCost, 1e-7);

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, -2 <= z, y active
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

      solver.setVariableBounds(MatrixTools.createVector(-5.0, 6.0, -2.0), MatrixTools.createVector(5.0, 10.0, Double.POSITIVE_INFINITY));

      activeVariables = MatrixTools.createVector(0.0, 1.0, 0.0);
      solver.setActiveVariables(activeVariables);

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

      // Minimize x^2 + y^2 + z^2 subject to x + y = 2.0, y - z <= -8, -5 <= x <= 5, 6 <= y <= 10, -2 <= z, z active
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

      solver.setVariableBounds(MatrixTools.createVector(-5.0, 6.0, -2.0), MatrixTools.createVector(5.0, 10.0, Double.POSITIVE_INFINITY));

      activeVariables = MatrixTools.createVector(0.0, 0.0, 1.0);
      solver.setActiveVariables(activeVariables);

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
      assertEquals(expectedNumberOfIterations2, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertEquals(0.0, solution.get(0), 1e-7);
      assertEquals(0.0, solution.get(1), 1e-7);
      assertEquals(8.0, solution.get(2), 1e-7);
      if (!ignoreLagrangeMultipliers)
      {
         assertEquals(0.0, lagrangeEqualityMultipliers.get(0), 1e-7);
         assertEquals(16.0, lagrangeInequalityMultipliers.get(0), 1e-7);

         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeLowerBoundMultipliers.get(2), 1e-7);

         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(0), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(1), 1e-7);
         assertEquals(0.0, lagrangeUpperBoundMultipliers.get(2), 1e-7);
      }

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertEquals(64.0, objectiveCost, 1e-7);

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

      solver.setVariableBounds(MatrixTools.createVector(3.0, 6.0, 11.0), MatrixTools.createVector(5.0, 10.0, Double.POSITIVE_INFINITY));

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
      assertEquals(expectedNubmerOfIterations3, numberOfIterations);

      assertEquals(3, solution.getNumRows());
      assertTrue(Double.isNaN(solution.get(0)));
      assertTrue(Double.isNaN(solution.get(1)));
      assertTrue(Double.isNaN(solution.get(2)));

      solutionMatrix = new DMatrixRMaj(costQuadraticMatrix.getNumRows(), 1);
      solutionMatrix.set(solution);
      objectiveCost = solver.getObjectiveCost(solutionMatrix);
      assertTrue(Double.isNaN(objectiveCost));
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

}
