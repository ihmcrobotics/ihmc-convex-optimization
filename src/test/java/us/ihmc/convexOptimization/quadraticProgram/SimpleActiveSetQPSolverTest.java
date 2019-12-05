package us.ihmc.convexOptimization.quadraticProgram;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

public class SimpleActiveSetQPSolverTest
{

   @Test
   public void testSimpleUnconstrainedOptimization()
   {
      // Minimize 1/2 * (x1^2 + x2^2) - x1 - 2.0 * x2; (Solution = [1, 2])
      AbstractActiveSetQPSolver solver = new SimpleActiveSetQPSolver();

      double[][] quadraticCostFunctionPMatrix = new double[][] {{1.0, 0.0}, {0.0, 1.0}};
      double[] quadraticCostFunctionQVector = new double[] {-1.0, -2.0};
      double quadraticCostFunctionR = 0.0;

      solver.setQuadraticCostFunction(quadraticCostFunctionPMatrix, quadraticCostFunctionQVector, quadraticCostFunctionR);

      double[] solution = solver.solve();
      assertEquals(2, solution.length);

      assertEquals(1.0, solution[0], 1e-7);
      assertEquals(2.0, solution[1], 1e-7);
   }

   @Test
   public void testSimpleEqualityConstrainedOptimization()
   {
      // Minimize 1/2 * (x1^2 + x2^2) subject to x1 + x2 = 2; (Solution = [1, 1])
      AbstractActiveSetQPSolver solver = new SimpleActiveSetQPSolver();

      double[][] quadraticCostFunctionPMatrix = new double[][] {{1.0, 0.0}, {0.0, 1.0}};
      double[] quadraticCostFunctionQVector = new double[] {0.0, 0.0};
      double quadraticCostFunctionR = 0.0;

      double[][] linearEqualityConstraintsAMatrix = new double[][] {{1.0, 1.0}};
      double[] linearEqualityConstraintsBVector = new double[] {2.0};

      solver.setQuadraticCostFunction(quadraticCostFunctionPMatrix, quadraticCostFunctionQVector, quadraticCostFunctionR);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      double[] solution = solver.solve();
      assertEquals(2, solution.length);

      assertEquals(1.0, solution[0], 1e-7);
      assertEquals(1.0, solution[1], 1e-7);
   }

   @Test
   public void testSimpleInequalityConstrainedOptimizationWithActiveConstraint()
   {
      // Minimize 1/2 * (x1^2 + x2^2) subject to x1 + x2 <= -2; (Solution = [-1, -1])
      AbstractActiveSetQPSolver solver = new SimpleActiveSetQPSolver();

      double[][] quadraticCostFunctionPMatrix = new double[][] {{1.0, 0.0}, {0.0, 1.0}};
      double[] quadraticCostFunctionQVector = new double[] {0.0, 0.0};
      double quadraticCostFunctionR = 0.0;

      double[][] linearInequalityConstraintCVectors = new double[][] {{1.0, 1.0}};
      double[] linearInequalityConstraintFs = new double[] {-2.0};

      solver.setQuadraticCostFunction(quadraticCostFunctionPMatrix, quadraticCostFunctionQVector, quadraticCostFunctionR);
      solver.setLinearInequalityConstraints(linearInequalityConstraintCVectors, linearInequalityConstraintFs);

      double[] solution = solver.solve();
      assertEquals(2, solution.length);

      assertEquals(-1.0, solution[0], 1e-7);
      assertEquals(-1.0, solution[1], 1e-7);
   }

   @Test
   public void testSimpleInequalityConstrainedOptimizationWithInactiveConstraint()
   {
      // Minimize 1/2 * (x1^2 + x2^2) - x1 - 2.0 * x2 subject to x1 + x2 <= 4; (Solution = [1, 2])
      AbstractActiveSetQPSolver solver = new SimpleActiveSetQPSolver();

      double[][] quadraticCostFunctionPMatrix = new double[][] {{1.0, 0.0}, {0.0, 1.0}};
      double[] quadraticCostFunctionQVector = new double[] {-1.0, -2.0};
      double quadraticCostFunctionR = 0.0;

      double[][] linearInequalityConstraintCVectors = new double[][] {{1.0, 1.0}};
      double[] linearInequalityConstraintFs = new double[] {4.0};

      solver.setQuadraticCostFunction(quadraticCostFunctionPMatrix, quadraticCostFunctionQVector, quadraticCostFunctionR);
      solver.setLinearInequalityConstraints(linearInequalityConstraintCVectors, linearInequalityConstraintFs);

      double[] solution = solver.solve();
      assertEquals(2, solution.length);

      assertEquals(1.0, solution[0], 1e-7);
      assertEquals(2.0, solution[1], 1e-7);
   }

   @Test
   public void testNoValidSolutionDueToNonSolvableEqualityConstraints()
   {
      // Minimize 1/2 * (x1^2 + x2^2) - x1 - 2.0 * x2 subject to x1 + x2 = 1; x1 + x2 = 2; (No solution)
      AbstractActiveSetQPSolver solver = new SimpleActiveSetQPSolver();

      double[][] quadraticCostFunctionPMatrix = new double[][] {{1.0, 0.0}, {0.0, 1.0}};
      double[] quadraticCostFunctionQVector = new double[] {-1.0, -2.0};
      double quadraticCostFunctionR = 0.0;

      double[][] linearEqualityConstraintsAMatrix = new double[][] {{1.0, 1.0}, {1.0, 1.0}};
      double[] linearEqualityConstraintsBVector = new double[] {1.0, 2.0};

      solver.setQuadraticCostFunction(quadraticCostFunctionPMatrix, quadraticCostFunctionQVector, quadraticCostFunctionR);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);

      double[] solution = solver.solve();
      assertEquals(2, solution.length);

      assertTrue(Double.isNaN(solution[0]));
      assertTrue(Double.isNaN(solution[1]));
   }

   // Fails when conflicting constraints are active... Need to fix this case.
   @Disabled
   @Test
   public void testConflictingInequalityAndEqualityConstraintsIfActive()
   {
      // Minimize 1/2 * (x1^2 + x2^2) - x1 - 2.0 * x2 subject to x1 = -4; x1 < 4; (Solution = [-4, 2])
      AbstractActiveSetQPSolver solver = new SimpleActiveSetQPSolver();

      double[][] quadraticCostFunctionPMatrix = new double[][] {{1.0, 0.0}, {0.0, 1.0}};
      double[] quadraticCostFunctionQVector = new double[] {-1.0, -2.0};
      double quadraticCostFunctionR = 0.0;

      double[][] linearEqualityConstraintsAMatrix = new double[][] {{1.0, 0.0},};
      double[] linearEqualityConstraintsBVector = new double[] {-4.0};

      double[][] linearInequalityConstraintCVectors = new double[][] {{1.0, 0.0}};
      double[] linearInequalityConstraintFs = new double[] {4.0};

      solver.setQuadraticCostFunction(quadraticCostFunctionPMatrix, quadraticCostFunctionQVector, quadraticCostFunctionR);
      solver.setLinearEqualityConstraints(linearEqualityConstraintsAMatrix, linearEqualityConstraintsBVector);
      solver.setLinearInequalityConstraints(linearInequalityConstraintCVectors, linearInequalityConstraintFs);

      solver.setLinearInequalityActiveSet(new boolean[] {true});

      double[] solution = solver.solve();
      assertEquals(2, solution.length);

      assertEquals(-4.0, solution[0], 1e-7);
      assertEquals(2.0, solution[1], 1e-7);
   }

}
