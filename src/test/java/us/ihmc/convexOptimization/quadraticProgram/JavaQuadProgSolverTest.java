package us.ihmc.convexOptimization.quadraticProgram;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.apache.commons.lang3.time.StopWatch;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import us.ihmc.commons.Conversions;
import us.ihmc.commons.MathTools;
import us.ihmc.convexOptimization.exceptions.NoConvergenceException;
import us.ihmc.log.LogTools;
import us.ihmc.matrixlib.MatrixTestTools;
import us.ihmc.matrixlib.MatrixTools;

public class JavaQuadProgSolverTest extends AbstractSimpleActiveSetQPSolverTest
{
   private static final double epsilon = 1e-4;

   @Override
   public ActiveSetQPSolver createSolverToTest()
   {
      JavaQuadProgSolver solver = new JavaQuadProgSolver();
      solver.setUseWarmStart(false);
      return solver;
   }

   @Test
   public void testTimingAgainstStandardQuadProg() throws NoConvergenceException
   {
      SolverTimer javaSolverTimer = new SolverTimer();
      SolverTimer wrapperSolverTimer = new SolverTimer();

      int numberOfInequalityConstraints = 1;
      int numberOfEqualityConstraints = 2;
      int numberOfVariables = 3;

      DMatrixRMaj Q = new DMatrixRMaj(numberOfVariables, numberOfVariables, true, 1, 0, 1, 0, 1, 2, 1, 3, 7);
      DMatrixRMaj f = new DMatrixRMaj(numberOfVariables, 1, true, 1, 0, 9);
      DMatrixRMaj Aeq = new DMatrixRMaj(numberOfEqualityConstraints, numberOfVariables, true, 1, 1, 1, 2, 3, 4);
      DMatrixRMaj beq = new DMatrixRMaj(numberOfEqualityConstraints, 1, true, 0, 7);
      DMatrixRMaj Ain = new DMatrixRMaj(numberOfInequalityConstraints, numberOfVariables, true, 2, 1, 3);
      DMatrixRMaj bin = new DMatrixRMaj(numberOfInequalityConstraints, 1, true, 0);

      int numberOfIterations = 10000;

      for (int repeat = 0; repeat < numberOfIterations; repeat++)
      {
         DMatrixRMaj x = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1, 3);
         DMatrixRMaj xWrapper = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1, 3);

         JavaQuadProgSolver javaSolver = new JavaQuadProgSolver();
         javaSolverTimer.startMeasurement();

         javaSolver.clear();
         javaSolver.setQuadraticCostFunction(Q, f, 0.0);
         javaSolver.setLinearInequalityConstraints(Ain, bin);
         javaSolver.setLinearEqualityConstraints(Aeq, beq);
         javaSolver.solve(x);

         javaSolverTimer.stopMeasurement();

         QuadProgSolver solver = new QuadProgSolver();
         wrapperSolverTimer.startMeasurement();
         solver.solve(Q, f, Aeq, beq, Ain, bin, xWrapper, false);
         wrapperSolverTimer.stopMeasurement();

         DMatrixRMaj bEqualityVerify = new DMatrixRMaj(numberOfEqualityConstraints, 1);
         CommonOps_DDRM.mult(Aeq, x, bEqualityVerify);

         // Verify Ax=b Equality constraints hold:
         MatrixTestTools.assertMatrixEquals(bEqualityVerify, beq, epsilon);

         // Verify Ax<b Inequality constraints hold:
         DMatrixRMaj bInequalityVerify = new DMatrixRMaj(numberOfInequalityConstraints, 1);
         CommonOps_DDRM.mult(Ain, x, bInequalityVerify);

         for (int j = 0; j < bInequalityVerify.getNumRows(); j++)
         {
            assertTrue(bInequalityVerify.get(j, 0) < beq.get(j, 0));
         }

         // Verify solution is as expected
         assertArrayEquals(x.getData(), xWrapper.getData(), 1e-10, "repeat = " + repeat);
      }

      LogTools.info("Wrapper solve time : " + wrapperSolverTimer.getAverageTime());
      LogTools.info("Java solve time : " + javaSolverTimer.getAverageTime());
   }

   @Test
   public void testTimingAgainstSimpleSolver()
   {
      SolverTimer quadProgTotalTimer = new SolverTimer();
      SolverTimer simpleTotalTimer = new SolverTimer();
      SolverTimer quadProgTimer = new SolverTimer();
      SolverTimer simpleTimer = new SolverTimer();

      // Minimize x^2 + y^2 subject to x + y >= 2 (-x -y <= -2), y <= 10x - 2 (-10x + y <= -2), x <= 10y - 2 (x - 10y <= -2),
      // Equality solution will violate all three constraints, but optimal only has the first constraint active.
      // However, if you set all three constraints active, there is no solution.
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(new double[][] {{2.0, 0.0}, {0.0, 2.0}});
      DMatrixRMaj costLinearVector = MatrixTools.createVector(0.0, 0.0);
      double quadraticCostScalar = 0.0;

      DMatrixRMaj linearInequalityConstraintsCMatrix = new DMatrixRMaj(new double[][] {{-1.0, -1.0}, {-10.0, 1.0}, {1.0, -10.0}});
      DMatrixRMaj linearInqualityConstraintsDVector = MatrixTools.createVector(-2.0, -2.0, -2.0);

      JavaQuadProgSolver quadProg = new JavaQuadProgSolver();
      SimpleEfficientActiveSetQPSolver simpleSolver = new SimpleEfficientActiveSetQPSolver();

      DMatrixRMaj quadProgSolution = new DMatrixRMaj(2, 1);
      DMatrixRMaj quadProgLagrangeEqualityMultipliers = new DMatrixRMaj(0, 1);
      DMatrixRMaj quadProgLagrangeInequalityMultipliers = new DMatrixRMaj(3, 1);
      DMatrixRMaj simpleSolution = new DMatrixRMaj(2, 1);

      for (int repeat = 0; repeat < 5000; repeat++)
      {
         quadProgTotalTimer.startMeasurement();

         quadProg.clear();
         quadProg.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);
         quadProg.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

         quadProgTimer.startMeasurement();
         quadProg.solve(quadProgSolution);
         quadProg.getLagrangeEqualityConstraintMultipliers(quadProgLagrangeEqualityMultipliers);
         quadProg.getLagrangeInequalityConstraintMultipliers(quadProgLagrangeInequalityMultipliers);
         quadProgTimer.stopMeasurement();

         quadProgTotalTimer.stopMeasurement();

         simpleTotalTimer.startMeasurement();

         simpleSolver.clear();
         simpleSolver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar);
         simpleSolver.setLinearInequalityConstraints(linearInequalityConstraintsCMatrix, linearInqualityConstraintsDVector);

         simpleTimer.startMeasurement();
         simpleSolver.solve(simpleSolution);
         simpleTimer.stopMeasurement();

         simpleTotalTimer.stopMeasurement();
      }

      LogTools.info("Quad Prog total time : " + quadProgTotalTimer.getAverageTime());
      LogTools.info("Simple total time : " + simpleTotalTimer.getAverageTime());
      LogTools.info("Quad Prog solve time : " + quadProgTimer.getAverageTime());
      LogTools.info("Simple solve time : " + simpleTimer.getAverageTime());
   }

   @Test
   public void testAgainstStandardQuadProg() throws NoConvergenceException
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

      JavaQuadProgSolver javaSolver = new JavaQuadProgSolver();
      QuadProgSolver solver = new QuadProgSolver();

      for (int repeat = 0; repeat < 10000; repeat++)
      {
         DMatrixRMaj x = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1);
         javaSolver.clear();
         javaSolver.setQuadraticCostFunction(Q, f, 0.0);
         javaSolver.setLinearInequalityConstraints(Ain, bin);
         javaSolver.setLinearEqualityConstraints(Aeq, beq);
         javaSolver.solve(x);
         assertArrayEquals(x.getData(), new double[] {-0.5, 0.5}, 1e-10);
      }

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
         DMatrixRMaj x = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1, 3);
         DMatrixRMaj xWrapper = new DMatrixRMaj(numberOfVariables, 1, true, -1, 1, 3);

         javaSolver.clear();
         javaSolver.setQuadraticCostFunction(Q, f, 0.0);
         javaSolver.setLinearEqualityConstraints(Aeq, beq);
         javaSolver.setLinearInequalityConstraints(Ain, bin);
         javaSolver.solve(x);
         solver.solve(Q, f, Aeq, beq, Ain, bin, xWrapper, false);

         DMatrixRMaj bEqualityVerify = new DMatrixRMaj(numberOfEqualityConstraints, 1);
         CommonOps_DDRM.mult(Aeq, x, bEqualityVerify);

         // Verify Ax=b Equality constraints hold:
         MatrixTestTools.assertMatrixEquals(bEqualityVerify, beq, epsilon);

         // Verify Ax<b Inequality constraints hold:
         DMatrixRMaj bInequalityVerify = new DMatrixRMaj(numberOfInequalityConstraints, 1);
         CommonOps_DDRM.mult(Ain, x, bInequalityVerify);

         for (int j = 0; j < bInequalityVerify.getNumRows(); j++)
         {
            assertTrue(bInequalityVerify.get(j, 0) < beq.get(j, 0));
         }

         // Verify solution is as expected
         assertArrayEquals(x.getData(), xWrapper.getData(), 1e-10, "repeat = " + repeat);
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

      JavaQuadProgSolver solver = new JavaQuadProgSolver();

      solver.clear();
      solver.setQuadraticCostFunction(Q, f, 0.0);
      solver.setLinearEqualityConstraints(Aeq, beq);
      solver.setLinearInequalityConstraints(Ain, bin);
      solver.solve(x);

      LogTools.info("Attempting to solve problem with: " + solver.getClass().getSimpleName());
      solver.clear();
      solver.setQuadraticCostFunction(Q, f, 0.0);
      solver.setLinearEqualityConstraints(Aeq, beq);
      solver.setLinearInequalityConstraints(Ain, bin);
      solver.solve(x);

      boolean correct = MathTools.epsilonEquals(-2.0, x.get(0), 10E-10);
      if (!correct)
      {
         LogTools.info("Failed. Java Result was " + x.get(0) + ", expected -2.0");
      }
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

   /**
    * GW: exported this from a failing Atlas unit test 05/23/2019 Not sure if this should be solvable
    * but in any case it should fail gracefully.
    */
   @Test
   public void testCaseFromSimulation()
   {
      DMatrixRMaj costQuadraticMatrix = new DMatrixRMaj(6, 6);
      costQuadraticMatrix.data = new double[] {993.9053988041245, 327.83942494534944, 993.556655887893, 327.83942494534944, 2308.09243287179, 354.1845700419416,
            327.83942494534944, 1423.124867640583, 327.83942494534944, 1422.7761247243516, 354.1845700419416, 2771.803937517132, 993.556655887893,
            327.83942494534944, 1009.1964870941272, 327.83942494534944, 2308.09243287179, 354.1845700419416, 327.83942494534944, 1422.7761247243516,
            327.83942494534944, 1438.4159559305858, 354.1845700419416, 2771.803937517132, 2308.09243287179, 354.1845700419416, 2308.09243287179,
            354.1845700419416, 5508.124706211761, 0.06581329118532331, 354.1845700419416, 2771.803937517132, 354.1845700419416, 2771.803937517132,
            0.06581329118532508, 5507.8812912972435};

      DMatrixRMaj costLinearVector = new DMatrixRMaj(6, 1);
      costLinearVector.data = new double[] {20222.5613016018, 5592.963753999038, 20222.5613016018, 5592.963753999038, 47486.04338938162, 5042.8181779521055};

      DMatrixRMaj quadraticCostScalar = new DMatrixRMaj(1, 1);
      quadraticCostScalar.data = new double[] {206999.16716064143};

      DMatrixRMaj linearInequalityConstraintCMatrix = new DMatrixRMaj(17, 6);
      linearInequalityConstraintCMatrix.data = new double[] {-0.532490759103742, 0.8464358165089191, 0.0, 0.0, 0.0, 0.0, 0.8781297702936439, 0.4784225188304082,
            0.0, 0.0, 0.0, 0.0, 0.4314826988044296, -0.9021212117184951, 0.0, 0.0, 0.0, 0.0, -0.8674617517025186, -0.4975038787117122, 0.0, 0.0, 0.0, 0.0,
            -0.532490759103742, 0.8464358165089191, -0.532490759103742, 0.8464358165089191, 0.0, 0.0, 0.8781297702936439, 0.4784225188304082,
            0.8781297702936439, 0.4784225188304082, 0.0, 0.0, 0.4314826988044296, -0.9021212117184951, 0.4314826988044296, -0.9021212117184951, 0.0, 0.0,
            -0.8674617517025186, -0.4975038787117122, -0.8674617517025186, -0.4975038787117122, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9838237285970943,
            0.17913925044308668, 0.0, 0.0, 0.0, 0.0, -0.6119239483029338, 0.7909166084318549, 0.0, 0.0, 0.0, 0.0, -0.15126410326159784, 0.9884933844313095, 0.0,
            0.0, 0.0, 0.0, 0.5483141711347475, 0.836272425548526, 0.0, 0.0, 0.0, 0.0, 0.39417400487132703, -0.9190358284004487, 0.0, 0.0, 0.0, 0.0, -0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0};

      DMatrixRMaj linearInequalityConstraintDVector = new DMatrixRMaj(17, 1);
      linearInequalityConstraintDVector.data = new double[] {0.03209312106775908, 0.12675512800740218, 0.05477548585328318, 0.07957996960652647,
            0.07200972933235494, 0.15848182583418868, 0.09792941203582961, 0.1298875084279425, 8.725615888588424, 4.899796215838858, 0.8792506851639368,
            -4.6465724005639855, -2.4847821341512306, -0.8057687026687546, -9.103397051640995, 1.0993716410559957, 9.292818302213409};

      JavaQuadProgSolver solver = new JavaQuadProgSolver();
      solver.setMaxNumberOfIterations(100);
      solver.setUseWarmStart(true);
      solver.clear();
      solver.resetActiveSet();
      solver.setQuadraticCostFunction(costQuadraticMatrix, costLinearVector, quadraticCostScalar.get(0, 0));
      solver.setLinearInequalityConstraints(linearInequalityConstraintCMatrix, linearInequalityConstraintDVector);

      DMatrixRMaj solution = new DMatrixRMaj(6, 1);
      solver.solve(solution);
   }

   private static class SolverTimer
   {
      private final StopWatch stopWatch = new StopWatch();
      private int lapCounter = 0;

      public SolverTimer()
      {
      }

      public void startMeasurement()
      {
         if (!stopWatch.isStarted())
            stopWatch.start();
         else
            stopWatch.resume();
      }

      public void stopMeasurement()
      {
         stopWatch.suspend();
         lapCounter++;
      }

      public double getAverageTime()
      {
         return Conversions.nanosecondsToSeconds(stopWatch.getNanoTime() / lapCounter);
      }
   }
}
