package us.ihmc.convexOptimization.quadraticProgram;

import org.junit.jupiter.api.Test;

public class SimpleDiagonalActiveSetQPSolverTest extends AbstractSimpleActiveSetQPSolverTest
{
   @Override
   public ActiveSetQPSolver createSolverToTest()
   {
      return new SimpleDiagonalActiveSetQPSolver();
   }

   @Override
   @Test
   public void testSimpleCasesWithBoundsConstraints()
   {
      testSimpleCasesWithBoundsConstraints(1, 3, 3, 2, false);
   }
}
