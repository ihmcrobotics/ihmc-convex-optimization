package us.ihmc.convexOptimization.experimental;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import com.joptimizer.functions.ConvexMultivariateRealFunction;
import com.joptimizer.functions.LinearMultivariateRealFunction;

public class ExperimentalSOCPSolverUsingJOptimizerTest
{
   public static final boolean VERBOSE = false;

   // At time of test writing JOptimizer wasn't working for SOCPs...
   @Disabled
   @Test
   public void testASimpleSecondOrderConeProblem()
   {
      // Minimize -(x + y) subject to z <= sqrt(18) and ||(x, y, 0)|| <= z. Answer should be (3, 3, sqrt(18))

      ExperimentalSOCPSolverUsingJOptimizer specialSOCPSolverUsingJOptimizer = new ExperimentalSOCPSolverUsingJOptimizer();

      double[] minimizeF = new double[] {-1.0, -1.0, 0.0};
      specialSOCPSolverUsingJOptimizer.setOptimizationFunctionVectorF(minimizeF);

      // inequalities
      double[][] coneInequalityMatrixB = new double[][] {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}};
      double[] coneInequalityVectorU = new double[] {0.0, 0.0, 1.0};

      ArrayList<ConvexMultivariateRealFunction> otherInequalities = new ArrayList<>();

      double[] qVector = new double[] {0.0, 0.0, 1.0};
      double r = -Math.sqrt(18.0);
      otherInequalities.add(new LinearMultivariateRealFunction(qVector, r));

      specialSOCPSolverUsingJOptimizer.setSpecialSecondOrderConeInequality(coneInequalityMatrixB, coneInequalityVectorU, otherInequalities);

      double[] solution = specialSOCPSolverUsingJOptimizer.solveAndReturnOptimalVector();
      if (VERBOSE)
         System.out.println("solution = (" + solution[0] + ", " + solution[1] + ", " + solution[2] + ")");

      assertEquals(3.0, solution[0], 1e-5);
      assertEquals(3.0, solution[1], 1e-5);
      assertEquals(Math.sqrt(18.0), solution[2], 1e-5);
   }

   //At time of test writing JOptimizer wasn't working for SOCPs...
   @Disabled
   @Test
   public void testARotatedSecondOrderConeProblem()
   {
      // Cone constraint: Given cone tilted to a normal vector and friction like constraints, with mu
      // Linear constraint: Given Ax = b
      // Find feasible solution that matches those two constraints

      ExperimentalSOCPSolverUsingJOptimizer specialSOCPSolverUsingJOptimizer = new ExperimentalSOCPSolverUsingJOptimizer();

      double[] minimizeF = new double[] {0.0, 0.0, 1.0};
      specialSOCPSolverUsingJOptimizer.setOptimizationFunctionVectorF(minimizeF);

      // Equality constraints:
      double[][] linearEqualityAMatrix = new double[][] {{0.0, 1.0, 0.0}};
      double[] linearEqualityBVector = new double[] {0.0};

      specialSOCPSolverUsingJOptimizer.setLinearEqualityConstraints(linearEqualityAMatrix, linearEqualityBVector);

      // Inequalities
      DMatrixRMaj surfaceNormal = new DMatrixRMaj(new double[][] {{0.0}, {0.0}, {1.0}});
      DMatrixRMaj identityMatrix = CommonOps_DDRM.identity(3);
      DMatrixRMaj coneInequalityMatrixB = new DMatrixRMaj(3, 3);

      DMatrixRMaj normalNormalTranspose = new DMatrixRMaj(3, 3);
      CommonOps_DDRM.multTransB(surfaceNormal, surfaceNormal, normalNormalTranspose);

      CommonOps_DDRM.subtract(identityMatrix, normalNormalTranspose, coneInequalityMatrixB);

      double mu = 1.0;
      DMatrixRMaj coneInequalityVectorU = new DMatrixRMaj(surfaceNormal);
      CommonOps_DDRM.scale(mu, coneInequalityVectorU);

      System.out.println("coneInequalityMatrixB = " + coneInequalityMatrixB);
      System.out.println("coneInequalityVectorU = " + coneInequalityVectorU);

      ArrayList<ConvexMultivariateRealFunction> otherInequalities = new ArrayList<>();

      //      double[] qVector = new double[]{0.0, 0.0, 1.0};
      //      double r = -Math.sqrt(18.0);
      //      otherInequalities.add(new LinearMultivariateRealFunction(qVector, r));    

      specialSOCPSolverUsingJOptimizer.setSpecialSecondOrderConeInequality(coneInequalityMatrixB, coneInequalityVectorU, otherInequalities);

      double[] solution = specialSOCPSolverUsingJOptimizer.solveAndReturnOptimalVector();
      System.out.println("solution = (" + solution[0] + ", " + solution[1] + ", " + solution[2] + ")");

      assertEquals(3.0, solution[0], 1e-5);
      assertEquals(3.0, solution[1], 1e-5);
      assertEquals(Math.sqrt(18.0), solution[2], 1e-5);
   }
}
