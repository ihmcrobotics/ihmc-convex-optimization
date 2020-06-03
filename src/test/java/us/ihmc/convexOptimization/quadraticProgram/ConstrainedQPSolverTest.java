package us.ihmc.convexOptimization.quadraticProgram;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.MathTools;
import us.ihmc.convexOptimization.exceptions.NoConvergenceException;
import us.ihmc.log.LogTools;
import us.ihmc.matrixlib.MatrixTestTools;

public class ConstrainedQPSolverTest
{
   @Test
   public void testSolveContrainedQP() throws NoConvergenceException
   {
      int numberOfInequalityConstraints = 1;
      int numberOfEqualityConstraints = 1;
      int numberOfVariables = 2;

      DMatrixRMaj Q = new DMatrixRMaj(numberOfVariables, numberOfVariables, true, 1, 0, 0, 1);
      DMatrixRMaj f = new DMatrixRMaj(numberOfVariables, 1, true, 1, 0);
      DMatrixRMaj Aeq = new DMatrixRMaj(numberOfEqualityConstraints, numberOfVariables, true, 1, 1);
      DMatrixRMaj beq = new DMatrixRMaj(numberOfEqualityConstraints, 1, true, 0);
      DMatrixRMaj Ain = new DMatrixRMaj(numberOfInequalityConstraints, numberOfVariables, true, 2, 1);
      DMatrixRMaj bin = new DMatrixRMaj(numberOfInequalityConstraints, 1, true, 0);

      ConstrainedQPSolver[] optimizers = createSolvers();
      JavaQuadProgSolver solver = new JavaQuadProgSolver();

      for (int repeat = 0; repeat < 10000; repeat++)
      {
         for (int i = 0; i < optimizers.length; i++)
         {
            DMatrixRMaj x = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1);
            optimizers[i].solve(Q, f, Aeq, beq, Ain, bin, x, false);
            assertArrayEquals(x.getData(), new double[] {-0.5, 0.5}, 1e-10);
         }
         DMatrixRMaj x = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1);

         solver.clear();
         solver.setQuadraticCostFunction(Q, f, 0.0);
         solver.setLinearInequalityConstraints(Ain, bin);
         solver.setLinearEqualityConstraints(Aeq, beq);
         solver.solve(x);
         //solver.solve(Q, f, 0.0, Aeq, beq, Ain, bin, x, false);

         assertArrayEquals(x.getData(), new double[] {-0.5, 0.5}, 1e-10);
      }

      //TODO: Need more test cases. Can't trust these QP solvers without them...
      optimizers = new ConstrainedQPSolver[] { //new JOptimizerConstrainedQPSolver(), new CompositeActiveSetQPSolver(registry)
            //            new OASESConstrainedQPSolver(registry),
            //            new QuadProgSolver(registry),

      };

      numberOfInequalityConstraints = 1;
      numberOfEqualityConstraints = 2;
      numberOfVariables = 3;

      Q = new DMatrixRMaj(numberOfVariables, numberOfVariables, true, 1, 0, 1, 0, 1, 2, 1, 3, 7);
      f = new DMatrixRMaj(numberOfVariables, 1, true, 1, 0, 9);
      Aeq = new DMatrixRMaj(numberOfEqualityConstraints, numberOfVariables, true, 1, 1, 1, 2, 3, 4);
      beq = new DMatrixRMaj(numberOfEqualityConstraints, 1, true, 0, 7);
      Ain = new DMatrixRMaj(numberOfInequalityConstraints, numberOfVariables, true, 2, 1, 3);
      bin = new DMatrixRMaj(numberOfInequalityConstraints, 1, true, 0);

      for (int repeat = 0; repeat < 10000; repeat++)
      {
         for (int i = 0; i < optimizers.length; i++)
         {
            DMatrixRMaj x = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1, 3);
            optimizers[i].solve(Q, f, Aeq, beq, Ain, bin, x, false);
            assertArrayEquals(x.getData(), new double[] {-7.75, 8.5, -0.75}, 1e-10, "repeat = " + repeat + ", optimizer = " + i);

            DMatrixRMaj bEqualityVerify = new DMatrixRMaj(numberOfEqualityConstraints, 1);
            CommonOps_DDRM.mult(Aeq, x, bEqualityVerify);

            // Verify Ax=b Equality constraints hold:
            MatrixTestTools.assertMatrixEquals(bEqualityVerify, beq, 1e-7);

            // Verify Ax<b Inequality constraints hold:
            DMatrixRMaj bInequalityVerify = new DMatrixRMaj(numberOfInequalityConstraints, 1);
            CommonOps_DDRM.mult(Ain, x, bInequalityVerify);

            for (int j = 0; j < bInequalityVerify.getNumRows(); j++)
            {
               assertTrue(bInequalityVerify.get(j, 0) < beq.get(j, 0));
            }
         }

         if (optimizers.length > 0)
         {
            DMatrixRMaj x = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1, 3);

            solver.clear();
            solver.setQuadraticCostFunction(Q, f, 0.0);
            solver.setLinearInequalityConstraints(Ain, bin);
            solver.setLinearEqualityConstraints(Aeq, beq);
            solver.solve(x);
            //solver.solve(Q, f, 0.0, Aeq, beq, Ain, bin, x, false);
            assertArrayEquals(x.getData(), new double[] {-7.75, 8.5, -0.75}, 1e-10, "repeat = " + repeat + ", Java solver");

            DMatrixRMaj bEqualityVerify = new DMatrixRMaj(numberOfEqualityConstraints, 1);
            CommonOps_DDRM.mult(Aeq, x, bEqualityVerify);

            // Verify Ax=b Equality constraints hold:
            MatrixTestTools.assertMatrixEquals(bEqualityVerify, beq, 1e-7);

            // Verify Ax<b Inequality constraints hold:
            DMatrixRMaj bInequalityVerify = new DMatrixRMaj(numberOfInequalityConstraints, 1);
            CommonOps_DDRM.mult(Ain, x, bInequalityVerify);

            for (int j = 0; j < bInequalityVerify.getNumRows(); j++)
            {
               assertTrue(bInequalityVerify.get(j, 0) < beq.get(j, 0));
            }
         }
      }
   }

   @Test
   public void testSolveProblemWithParallelConstraints() throws NoConvergenceException
   {
      // our simple active set solver can not solve this:
      // test problem: x <= -1 and x <= -2
      DMatrixRMaj Q = new DMatrixRMaj(1, 1);
      DMatrixRMaj Ain = new DMatrixRMaj(2, 1);
      DMatrixRMaj bin = new DMatrixRMaj(2, 1);
      DMatrixRMaj x = new DMatrixRMaj(1, 1);

      Q.set(0, 0, 1.0);
      Ain.set(0, 0, 1.0);
      Ain.set(1, 0, 1.0);
      bin.set(0, -1.0);
      bin.set(0, -2.0);

      DMatrixRMaj f = new DMatrixRMaj(1, 1);
      DMatrixRMaj Aeq = new DMatrixRMaj(0, 1);
      DMatrixRMaj beq = new DMatrixRMaj(0, 1);

      ConstrainedQPSolver[] solvers = createSolvers();
      for (ConstrainedQPSolver solver : solvers)
      {
         LogTools.info("Attempting to solve problem with: " + solver.getClass().getSimpleName());
         solver.solve(Q, f, Aeq, beq, Ain, bin, x, true);
         boolean correct = MathTools.epsilonEquals(-2.0, x.get(0), 10E-10);
         if (!correct)
            LogTools.info("Failed. Result was " + x.get(0) + ", expected -2.0");
         assertTrue(correct);
      }

      JavaQuadProgSolver solver = new JavaQuadProgSolver();
      LogTools.info("Attempting to solve problem with: " + solver.getClass().getSimpleName());

      solver.clear();
      solver.setQuadraticCostFunction(Q, f, 0.0);
      solver.setLinearInequalityConstraints(Ain, bin);
      solver.setLinearEqualityConstraints(Aeq, beq);
      solver.solve(x);

      //solver.solve(Q, f, 0.0, Aeq, beq, Ain, bin, x, false);
      boolean correct = MathTools.epsilonEquals(-2.0, x.get(0), 10E-10);
      if (!correct)
         LogTools.info("Failed. Result was " + x.get(0) + ", expected -2.0");
      assertTrue(correct);
   }

   private ConstrainedQPSolver[] createSolvers()
   {
      ConstrainedQPSolver[] optimizers = { //new JOptimizerConstrainedQPSolver(),
            new OASESConstrainedQPSolver(), new QuadProgSolver()};
      return optimizers;
   }
}
