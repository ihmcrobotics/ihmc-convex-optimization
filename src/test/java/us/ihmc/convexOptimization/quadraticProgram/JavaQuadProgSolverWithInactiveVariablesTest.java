package us.ihmc.convexOptimization.quadraticProgram;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.ejml.data.DMatrixRMaj;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import us.ihmc.convexOptimization.exceptions.NoConvergenceException;
import us.ihmc.matrixlib.MatrixTools;

public class JavaQuadProgSolverWithInactiveVariablesTest extends AbstractSimpleActiveSetQPSolverWithInactiveVariablesTest
{
   private static final double epsilon = 1e-4;

   @Override
   public ActiveSetQPSolverWithInactiveVariablesInterface createSolverToTest()
   {
      JavaQuadProgSolverWithInactiveVariables solver = new JavaQuadProgSolverWithInactiveVariables();
      solver.setUseWarmStart(false);
      return solver;
   }

   @Override /** have to override because quad prog uses fewer iterations */
   @Test
   public void testSolutionMethodsAreAllConsistent() throws NoConvergenceException
   {
      testSolutionMethodsAreAllConsistent(1);
   }

   @Override /** have to override because quad prog uses fewer iterations */
   @Test
   public void testSimpleCasesWithInequalityConstraints()
   {
      testSimpleCasesWithInequalityConstraints(0);
   }

   @Override /** have to override because quad prog uses fewer iterations */
   @Test
   public void testSimpleCasesWithBoundsConstraints()
   {
      testSimpleCasesWithBoundsConstraints(0, 1, 6, 2, true);
   }

   @Override /** have to override because quad prog uses different iterations */
   @Test
   public void testClear()
   {
      testClear(6, 1, true);
   }

   @Override
   @Test
   public void testMaxIterations()
   {
      testMaxIterations(6, false);
   }

   @Override
   @Test
   public void test2DCasesWithPolygonConstraints()
   {
      test2DCasesWithPolygonConstraints(2, 1);
   }

   @Override
   @Disabled
   @Test
   public void testChallengingCasesWithPolygonConstraints()
   {
      testChallengingCasesWithPolygonConstraints(1, 5);
   }

   @Override
   @Test
   public void testSimpleCasesWithInequalityConstraintsAndInactiveVariables()
   {
      testSimpleCasesWithInequalityConstraintsAndInactiveVariables(0);
   }

   @Override
   @Test
   public void testSimpleCasesWithBoundsConstraintsAndInactiveVariables()
   {
      testSimpleCasesWithBoundsConstraintsAndInactiveVariables(0, 1, 2, 0, false);
   }

   @Override /** This IS a good solver **/
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
      assertEquals(solution.get(0), 1.0, epsilon);
      assertEquals(solution.get(1), 1.0, epsilon);
      assertEquals(lagrangeInequalityMultipliers.get(0), 2.0, epsilon);
      assertEquals(lagrangeInequalityMultipliers.get(1), 0.0, epsilon);
      assertEquals(lagrangeInequalityMultipliers.get(2), 0.0, epsilon);
   }
}
